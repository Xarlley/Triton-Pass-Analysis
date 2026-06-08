# Spikingformer-8-768 — 实验脚本说明

A100 上 Spikingformer-8-768 的实验脚本快照。脚本中的路径是 **A100 绝对路径**，
原样保留，用于在 A100 复跑。

> ⚠️ **同步规则（给 Claude）**：今后在 A100 上新增/修改本模型相关脚本，必须同步回本目录
> 与 `examples/snn_triton_pipeline/_a100_backup/`。详见 [`../../README.md`](../../README.md)。

## 模型与环境

| 项 | 值 |
|---|---|
| 模型 | Spikingformer-8-768（脉冲 Transformer，embed=768, depths=8, heads=8, patch16, T=4, 多步 LIF τ=2） |
| checkpoint | `/home/liushifeng/lsf/checkpoints/spikingformer_8_768/checkpoint-284.pth.tar`（key `state_dict`） |
| 模型构造 | `repos/Spikingformer/imagenet/model.py::vit_snn(...)` |
| 输入变换 | `r224_bicubic`（ckpt 训练 crop_pct=1.0, bicubic） |
| 参考精度 | top-1 ≈ **75.9%** |
| 环境 | eager：`snn-infer`；整网 Triton：`triton-src` |

## 模型专属脚本

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| **`run_spikingformer.py`** | **eager 基线**。加载 ckpt → eager 推理 → ImageNet val top-1/5。 | `python run_spikingformer.py --n 2000 --bs 50` |
| **`run_spikingformer_triton.py`** | **整网走 Triton**。`timm_compat`+`sj_compat`（强制 triton 神经元后端）；`--compile` 用 inductor max-autotune（关 cudnn、GEMM/conv→Triton、`compile_threads=1`）；`--profile` 统计 kernel 占比；`--amp bf16`。注意力矩阵乘 + 1×1 卷积都走 Triton 模板。 | `python run_spikingformer_triton.py --n 2000 --bs 50 --compile --amp bf16 --triton-conv` |

> 本模型没有 snn_compiler 加速实验（snn_compiler 当前主打 IF/LIF 卷积型 SNN；
> Spikingformer 的注意力/矩阵乘不在其融合 pattern 内）。

## 共享基础设施（`_common/`）

`sj_compat.py` / `timm_compat.py` / `snn_eval_lib_triton.py` / `snn_eval_lib.py` /
`capture_ir.py` / `capture_both.sh` / `run_all_bf16_2k.sh` /
`spikingjelly_triton_utils.elementty.patch` / `PIPELINE_OVERVIEW.md`
——作用与三模型一致，详细说明见
[`../../sew_resnet34/scripts/README.md`](../../sew_resnet34/scripts/README.md) 的「共享基础设施」一节。

## 来源（A100）

`run_spikingformer.py`, `snn_eval_lib.py` ← `~/charlley/snn_infer/`；
`run_spikingformer_triton.py` 及 `_common/` ← `~/charlley/snn_infer_triton/`；
`.elementty.patch` ← `~/lsf/inference_code_triton/`。
