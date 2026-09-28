# Biren SUPA

本页介绍壁仞（Biren）SUPA 加速卡的接入方式、功能范围与融合算子加速能力。

SUPA 通过 PyTorch 的 PrivateUse1 机制注册为 `supa` 设备。`import torch_supa` 后即可使用 `torch.supa.*` 接口（`is_available`、`device_count`、`mem_get_info`、`is_bf16_supported` 等），用法与 `torch.cuda` 基本一致。

## 安装

SUPA 需要壁仞提供的驱动、运行时以及配套的 PyTorch 插件包：

| 组件 | 说明 |
|------|------|
| `torch_supa` | 注册 `supa` 设备，`import torch` 时自动加载 |
| `torch_supa_ext` | 提供融合算子（`torch.ops.sudnn.*`，如 `rms_norm_func`），需显式安装 |

安装 LlamaFactory 及其依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

再按壁仞官方指引安装与硬件匹配的 `torch_supa` / `torch_supa_ext`。

### 验证 PyTorch SUPA

```bash
python -c "import torch, torch_supa; print(torch.supa.is_available())"
```

输出 `True` 表示 PyTorch 已识别 SUPA 设备。也可以通过下面命令查看被识别的设备信息：

```bash
llamafactory-cli env
```

当检测到 SUPA 时，会额外打印 `SUPA type` / `SUPA number` / `SUPA memory` 等字段。

## 代码接入链路

SUPA 复用了各后端共用的 accelerator 抽象，主要改动集中在以下位置：

| 环节 | 位置 | 作用 |
|------|------|------|
| 设备探测 | `extras/misc.py:is_torch_supa_available` | 统一入口，其余判断均基于它 |
| 设备/内存/GC | `extras/misc.py` | `get_current_device`、`get_device_count`、`get_current_memory`、`torch_gc` 等增加 `supa` 分支 |
| bf16 / fp16 | `extras/misc.py`、`hparams/parser.py` | 通过 `torch.supa.is_bf16_supported()` 判定 `pure_bf16` 可用性 |
| 环境变量 | `hparams/parser.py:_set_env_vars` | 导入 `torch_supa_ext` 注册融合算子，并设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| 注意力实现 | `model/model_utils/attention.py` | SUPA 无 FlashAttention，`auto/fa2/fa3` 统一回退到 SDPA |
| 梯度检查点 | `model/model_utils/checkpointing.py` | 反向按原始设备（cuda/npu/supa）搬运张量，不再硬编码 `cuda` |
| Profiler | `train/callbacks.py` | SUPA 虽报告为 cuda 但不支持 CUDA profiler 后端，故跳过 `ProfilerActivity.CUDA` |
| Ray | `train/tuner.py` | 识别 `RAY_EXPERIMENTAL_NOSET_SUPA_VISIBLE_DEVICES` |
| v1 accelerator | `v1/accelerator/helper.py:DeviceType.SUPA` | v1 设备枚举 |
| 融合算子 | `v1/plugins/model_plugins/kernels/ops/rms_norm/supa_rms_norm.py` | 融合 RMSNorm，`kernel_config` 中名称为 `supa_fused_rmsnorm` |

## 功能范围

| 功能 | 状态 | 说明 |
|------|:----:|------|
| SFT 全参训练 | 支持 | 使用通用训练路径 |
| LoRA / Freeze | 支持 | 使用通用 PEFT 路径 |
| QLoRA | 不支持 | 当前量化路径依赖 bitsandbytes |
| Unsloth | 不支持 | `use_unsloth` 在 SUPA 上会直接报错 |
| FlashAttention | 不支持 | 自动回退到 SDPA |
| 融合 RMSNorm | 支持 | 需安装 `torch_supa_ext`，否则回退到 eager |
| `pure_bf16` | 依赖硬件 | 由 `torch.supa.is_bf16_supported()` 决定 |

## 融合算子加速

在 v1（`USE_V1=1`）下，`kernel_config.name: auto` 在 SUPA 上会尝试：

- `supa_fused_rmsnorm`

也可以显式指定：

```yaml
kernel_config:
  name: supa_fused_rmsnorm
```

该 Kernel 仅替换标准约定（`y = x * rsqrt(mean(x^2) + eps) * weight`）的 RMSNorm 模块，支持的类名维护在 `supa_rms_norm.py` 的 `_SUPPORTED_RMSNORM_CLASSES` 白名单中。对于非标准约定（如 Gemma 的 `1 + weight`、门控/残差 RMSNorm）或未列入白名单的类，会跳过替换并回退到 eager 路径，同时打印一次告警。若未安装 `torch_supa_ext`，同样回退到 eager 路径。
