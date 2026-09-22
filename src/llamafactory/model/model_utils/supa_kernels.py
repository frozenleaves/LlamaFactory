# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Fused operators for the supa (SUPA PrivateUse1) accelerator.

The eager RMSNorm path upcasts activations to float32, and float32 elementwise
math (``pow``/``mean``/``rsqrt``) is orders of magnitude slower than bfloat16 on
supa, which manifests as an apparent training hang. supa ships a fast fused
forward kernel (``torch.ops.sudnn.rms_norm_func``) but it is not autograd-aware,
so it cannot be dropped into a trainable model directly.

This module wraps the fused forward in a :class:`torch.autograd.Function` whose
backward is computed with bfloat16-friendly elementwise kernels, and patches the
``forward`` of a model's RMSNorm modules to use it when tensors live on supa.
"""

import functools
from typing import TYPE_CHECKING

import torch

from ...extras import logging
from ...extras.misc import get_device_name, is_env_enabled, is_torch_supa_available


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


@functools.lru_cache
def _is_sudnn_rms_norm_available() -> bool:
    r"""Check whether the fused ``sudnn.rms_norm_func`` operator is registered."""
    if not is_torch_supa_available():
        return False

    try:
        import torch_supa_ext  # noqa: F401  # registers torch.ops.sudnn.*
    except Exception:
        pass

    return hasattr(torch.ops, "sudnn") and hasattr(torch.ops.sudnn, "rms_norm_func")


class SupaRMSNormFunction(torch.autograd.Function):
    r"""Autograd wrapper around the fused supa RMSNorm forward kernel.

    Forward uses ``torch.ops.sudnn.rms_norm_func``; backward recomputes the
    reciprocal RMS in the activation dtype (bfloat16 on supa) and derives the
    input/weight gradients analytically, avoiding the slow float32 path.

    Only the standard RMSNorm definition ``out = weight * x / rms(x)`` is
    supported. Variants that use ``(1 + weight)`` (e.g. Gemma) must not be
    routed here (see :func:`patch_rmsnorm_for_supa`).
    """

    @staticmethod
    def forward(ctx, hidden_states: "torch.Tensor", weight: "torch.Tensor", eps: float) -> "torch.Tensor":
        weight = weight.to(dtype=hidden_states.dtype)
        bias = torch.zeros_like(weight)
        output = torch.ops.sudnn.rms_norm_func(hidden_states, weight, bias, eps, False, False)
        ctx.save_for_backward(hidden_states, weight)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        hidden_states, weight = ctx.saved_tensors
        eps = ctx.eps
        hidden_size = hidden_states.shape[-1]

        # recompute the reciprocal RMS in the activation dtype (fast on supa)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        rrms = torch.rsqrt(variance + eps)  # (..., 1)
        x_norm = hidden_states * rrms  # normalized activations (without weight)

        grad_input = grad_weight = None

        if ctx.needs_input_grad[1]:
            grad_weight = (grad_output * x_norm).reshape(-1, hidden_size).sum(0).to(weight.dtype)

        if ctx.needs_input_grad[0]:
            grad_norm = grad_output * weight  # d L / d x_norm
            # d L / d x = rrms * grad_norm - (rrms^3 / H) * x * sum_k(grad_norm_k * x_k)
            dot = (grad_norm * hidden_states).sum(-1, keepdim=True)
            grad_input = rrms * grad_norm - (rrms.pow(3) / hidden_size) * hidden_states * dot
            grad_input = grad_input.to(hidden_states.dtype)

        return grad_input, grad_weight, None


def _patch_rmsnorm_class(cls: type) -> bool:
    r"""Patch a RMSNorm class ``forward`` to route supa tensors to the fused kernel."""
    if getattr(cls, "_supa_rmsnorm_patched", False):
        return False

    original_forward = cls.forward

    def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        if hidden_states.device.type != "supa":
            return original_forward(self, hidden_states)

        return SupaRMSNormFunction.apply(hidden_states, self.weight, self.variance_epsilon)

    cls.forward = forward
    cls._supa_rmsnorm_patched = True
    cls._supa_rmsnorm_original_forward = original_forward
    return True


def patch_rmsnorm_for_supa(model: "PreTrainedModel") -> None:
    r"""Route the model's RMSNorm modules to the fused supa kernel when on supa.

    No-op unless supa is the active device. Controlled by the ``SUPA_FUSED_RMSNORM``
    environment variable (enabled by default).
    """
    if get_device_name() != "supa":
        return

    if not is_env_enabled("SUPA_FUSED_RMSNORM", default="1"):
        logger.info_rank0("Fused supa RMSNorm is disabled via SUPA_FUSED_RMSNORM=0.")
        return

    if not _is_sudnn_rms_norm_available():
        logger.warning_rank0(
            "Fused supa RMSNorm (torch.ops.sudnn.rms_norm_func) is unavailable; "
            "the eager float32 RMSNorm path may be extremely slow on supa. "
            "Ensure torch_supa_ext is installed."
        )
        return

    patched_classes = set()
    for module in model.modules():
        cls = module.__class__
        if cls in patched_classes or getattr(cls, "_supa_rmsnorm_patched", False):
            continue

        # heuristic: only standard RMSNorm modules exposing weight + variance_epsilon
        if "RMSNorm" not in cls.__name__:
            continue

        if "Gemma" in cls.__name__:  # gemma uses the (1 + weight) convention, not standard RMSNorm
            logger.warning_rank0_once(
                f"Skipping fused supa RMSNorm for {cls.__name__} (non-standard weight convention)."
            )
            continue

        if not hasattr(module, "weight") or not hasattr(module, "variance_epsilon"):
            continue

        if _patch_rmsnorm_class(cls):
            patched_classes.add(cls)

    if patched_classes:
        logger.info_rank0(
            "Applied fused supa RMSNorm to: {}.".format(", ".join(sorted(c.__name__ for c in patched_classes)))
        )
