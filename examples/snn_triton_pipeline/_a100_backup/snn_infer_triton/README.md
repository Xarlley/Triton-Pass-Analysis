# SNN Inference on the spikingjelly **Triton** backend (triton-src env)

2026-06-05. 用 `triton-src` conda 环境里的 spikingjelly（源码版，activation_based API + Triton kernel）复现三个 SNN 权重的 ImageNet 推理，并尽量把整模型算子都交给 Triton 编译（关 cudnn、GEMM 强制 Triton），不走 torch eager / 不调 cudnn/cublas。SDT-V3 不在范围内。

## 结果

### 阶段A — 脉冲神经元走 Triton（eager；conv/Linear 仍是 cudnn/cublas），ImageNet val 前 2000 张
| 模型 | top-1 | top-5 | 对比（snn-infer 的 torch 后端） |
|------|-------|-------|------|
| SEW-ResNet-34 (ADD,T=4)        | 67.75% | 87.45% | 完全一致 |
| Spikingformer-8-768 (T=4)      | 75.95% | 93.30% | 完全一致 |
| Meta-SpikeFormer-55M (SDT-V2)  | 79.90% | 94.85% | 完全一致 |

神经元 `backend='triton'`（多步模式）与 `backend='torch'` 逐位一致（smoke test maxdiff=0）。

### 阶段B — 整模型 Triton（torch.compile/inductor，关 cudnn，GEMM=Triton，bf16），前 2000 张
| 模型 | top-1 | top-5 | Triton 占比* | cublas | cudnn | 其它 |
|------|-------|-------|-------------|--------|-------|------|
| SEW-ResNet-34       | 67.10% | 87.55% | **100%** | 0 | 0 | 0 |
| Spikingformer-8-768 | 76.55% | 93.15% | **100%** | 0 | 0 | 0 |
| Meta-SpikeFormer-55M| 79.85% | 95.10% | 64% | **0** | **0** | 36% |

\* 单 batch(bs=50,bf16) CUDA kernel 自时间占比。**三个模型 cublas/cudnn 均为 0。** SDT-V2 残留 36% 是 ATEN 的 **depthwise 卷积**原生 kernel（`at::native::conv_depthwise2d_*`，SepConv 的 7×7 深度可分离卷积）——inductor 没有 depthwise/grouped 卷积的 Triton 模板，只能回退 ATEN 原生 kernel（注意：它既不是 cudnn 也不是 cublas）。

## 关键工程点（按重要性）

1. **修复 spikingjelly 自身 Triton kernel 的 bug**（否则 triton 后端根本编不过）：
   `spikingjelly/activation_based/triton_kernel/triton_utils.py` 的 `convert_and_store` 里
   `value.to(pointer.dtype.element_ty.element_ty)` → `pointer.dtype.element_ty`（triton 3.7 的 block-pointer 类型 API：`block_ptr.dtype.element_ty` 已是标量类型，多写一层会报 `'dtype' has no attribute 'element_ty'`）。备份在同目录 `triton_utils.py.bak`。
2. **inductor 必须 `compile_threads=1`**：源码编译的 triton 在 inductor 子进程编译 worker 里初始化不了驱动（`RuntimeError: 0 active drivers`），所有 triton 模板都会编译失败并回退 ATEN。设为 1 即在主进程内编译，triton 模板正常。
3. **去 cudnn/cublas 的 inductor 配置**：`torch.backends.cudnn.enabled=False`、`max_autotune=True`、`max_autotune_gemm_backends="TRITON"`、`max_autotune_conv_backends="ATEN,TRITON"`、`conv_1x1_as_mm=True`、`mode="max-autotune-no-cudagraphs"`。
4. **bf16 解决共享内存溢出（核心）**：fp32 下 triton 的 conv/mm 模板大 block 配置需 256–288KB 共享内存 > A100 上限 163KB → 全部 OOM 回退 ATEN（关 cudnn 时即 im2col+cublas sgemm）。**改 bf16（`torch.autocast`）后单元素 2 字节，足迹减半 → triton 模板放得下 → conv/GEMM 全变 Triton，cublas/cudnn 清零。** 精度无损（bf16≈fp32，且原仓库本就用 AMP 训练/评测）。

## 文件
- `sj_compat.py` —— 把仓库的旧 `spikingjelly.clock_driven` / `cext` 导入映射到新的 `activation_based` API，并把所有神经元强制成 `step_mode='m', backend='triton'`（`SJ_NEURON_BACKEND=torch` 可切回 torch 后端做对照）。
- `timm_compat.py` —— 只补 transformer 仓库 import 的少量 timm 符号（to_2tuple/trunc_normal_/DropPath/register_model/_cfg），不动 triton-src 环境。einops 是 triton-src 真实安装的（spikingjelly 依赖），不 shim。
- `snn_eval_lib_triton.py` —— 数据/评测循环（每 batch `functional.reset_net`）+ CUDA kernel 分类 profiler（triton / cublas / cudnn / other）+ `--amp` autocast。
- `run_{sew,spikingformer,sdtv2}_triton.py` —— 各模型 driver，参数：`--n --bs --compile --profile --amp {none,bf16,fp16} --triton-conv`。
- `run_all_bf16_2k.sh` —— 三模型 n=2000 bf16+compile 一键复现。
- 复用 `snn-infer` 那次的数据集与标注（`~/charlley/snn_infer/data/`）和已 patch 的仓库（`~/charlley/snn_infer/repos/`）。

## 运行
```bash
conda activate triton-src
cd ~/charlley/snn_infer_triton
# 阶段A：神经元走 triton（eager），复现精度
SJ_NEURON_BACKEND=triton python run_sew_triton.py --n 2000 --bs 100
# 阶段B：整模型 triton（bf16，关 cudnn，强制 triton GEMM/conv）+ 算子归类 profile
SJ_NEURON_BACKEND=triton python run_sew_triton.py --n 2000 --bs 50 --compile --amp bf16 --triton-conv --profile
```
环境（triton-src）：python 3.11、torch 2.12.0+cu130、**triton 3.7.0（源码编译）**、spikingjelly 0.0.0.0.15（源码 editable，含 triton_kernel）、einops 0.8.2、timm 用 shim。
