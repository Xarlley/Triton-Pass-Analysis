# `_a100_backup/` — A100 实验工作区脚本的原始镜像

本目录是 A100（`liushifeng@a100`）上几个 SNN 推理 / Triton / snn_compiler 工作区的
**逐文件原样备份**（只含脚本与文档，不含数据集、checkpoint、triton/inductor 缓存、IR 产物）。
保持原始目录结构以保留出处；各模型经过整理、带文档的副本见
`../<model>/scripts/`。

## 子目录 = A100 路径

| 本目录 | A100 路径 | 内容 |
|---|---|---|
| `snn_infer/` | `~/charlley/snn_infer/` | **eager 基线**推理工作区：`run_{sew,spikingformer,sdtv2,sdtv3}.py` + `snn_eval_lib.py`。env `snn-infer`。 |
| `snn_infer_triton/` | `~/charlley/snn_infer_triton/` | **整网走 Triton** 流水线（torch.compile + inductor + 强制 Triton 神经元）：`run_*_triton.py`、`capture_ir.py`、`sj_compat.py`、`timm_compat.py`、`snn_eval_lib_triton.py`、`*.sh`、`README.md`。env `triton-src`。 |
| `snn_compiler_test/` | `~/charlley/snn_compiler_test/` | **snn_compiler 加速实验**（仅 SEW-ResNet-34）：`sew_exp.py`、`accuracy.py`、`smoke.py`、`refload.py`、`debug*.py`。env `triton-src`。（注：A100 上该目录还含一份 `snn_compiler/` 包副本——即仓库根的 `snn_compiler/`，本备份未重复收录。） |
| `inference_code_triton/` | `~/lsf/inference_code_triton/` | `snn_infer_triton/` 的 lsf 镜像，**额外含** `spikingjelly_triton_utils.elementty.patch`（SJ triton 神经元 kernel 的 triton-3.7 编译修复）。 |

## 关于 `run_sdtv3.py`

`snn_infer/run_sdtv3.py` 跑的是 **E-SpikeFormer-83M / SDT-V3**（83M, ImageNet top-1 ≈ 84%）。
它**不在** `examples/snn_triton_pipeline/` 的三个示例模型之列（没有对应的 `<model>/` 目录），
故只保留在本备份中，未整理进 per-model `scripts/`。

## 同步规则（给 Claude）

A100 上这些脚本是"真源"。每当在 A100 **新增 / 修改 / 删除**脚本：
1. 更新本备份对应文件（保持与 A100 逐字节一致）；
2. 同步更新 `../<model>/scripts/` 中该模型的整理副本与其 `README.md`。

连接：`ssh -p 3004 -o "ProxyJump charlley@180.76.139.31:40022" liushifeng@172.18.23.247`
（详见 `remote_gpu_servers/a100-liushifeng.md`）。
