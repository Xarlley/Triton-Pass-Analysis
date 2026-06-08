# Meta-SpikeFormer-55M (SDT-V2) — 实验脚本说明

A100 上 Meta-SpikeFormer-55M（Spike-Driven-Transformer-V2）的实验脚本快照。
脚本中的路径是 **A100 绝对路径**，原样保留，用于在 A100 复跑。

> ⚠️ **同步规则（给 Claude）**：今后在 A100 上新增/修改本模型相关脚本，必须同步回本目录
> 与 `examples/snn_triton_pipeline/_a100_backup/`。详见 [`../../README.md`](../../README.md)。

## 模型与环境

| 项 | 值 |
|---|---|
| 模型 | Meta-SpikeFormer-55M / SDT-V2（脉冲 Transformer + SepConv，T=4, 多步 LIF） |
| checkpoint | `/home/liushifeng/lsf/checkpoints/spike_driven_v2_metaspikeformer/55M_kd_T4.pth`（key `model`，含 KD 头） |
| 模型构造 | `repos/Spike-Driven-Transformer-V2/classification/models.py::metaspikformer_8_512(kd=True)`，构造后置 `m.T=4` |
| 输入变换 | `r256_bicubic` |
| 参考精度 | top-1 ≈ **79.9%** |
| 环境 | eager：`snn-infer`；整网 Triton：`triton-src` |

## 模型专属脚本

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| **`run_sdtv2.py`** | **eager 基线**。加载 ckpt → eager 推理 → ImageNet val top-1/5。 | `python run_sdtv2.py --n 2000 --bs 32` |
| **`run_sdtv2_triton.py`** | **整网走 Triton**。`timm_compat`+`sj_compat`（强制 triton 神经元后端）；`--compile` 用 inductor max-autotune（关 cudnn、GEMM/conv→Triton、`compile_threads=1`）；`--profile` 统计 kernel 占比；`--amp bf16`。残留 ATEN 部分是 SepConv 的 7×7 depthwise 卷积（inductor 无对应 Triton 模板，非 cudnn/cublas）。 | `python run_sdtv2_triton.py --n 2000 --bs 32 --compile --amp bf16 --triton-conv` |

> 本模型没有 snn_compiler 加速实验（同上：注意力/矩阵乘/depthwise 不在 snn_compiler 融合 pattern 内）。

## 共享基础设施（`_common/`）

`sj_compat.py` / `timm_compat.py` / `snn_eval_lib_triton.py` / `snn_eval_lib.py` /
`capture_ir.py` / `capture_both.sh` / `run_all_bf16_2k.sh` /
`spikingjelly_triton_utils.elementty.patch` / `PIPELINE_OVERVIEW.md`
——作用与三模型一致，详细说明见
[`../../sew_resnet34/scripts/README.md`](../../sew_resnet34/scripts/README.md) 的「共享基础设施」一节。

## 来源（A100）

`run_sdtv2.py`, `snn_eval_lib.py` ← `~/charlley/snn_infer/`；
`run_sdtv2_triton.py` 及 `_common/` ← `~/charlley/snn_infer_triton/`；
`.elementty.patch` ← `~/lsf/inference_code_triton/`。
