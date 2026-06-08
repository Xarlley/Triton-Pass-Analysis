# SEW-ResNet-34 — 实验脚本说明

本目录是 **A100 机器上 SEW-ResNet-34 全部实验脚本的快照**（来源见文末）。
脚本里的路径（checkpoint / 数据集 / repo）都是 **A100 上的绝对路径**，原样保留，
用于在 A100 复跑，不能直接在本地跑。

> ⚠️ **同步规则（给 Claude）**：今后若在 A100 上**新增或修改**了与本模型相关的脚本，
> 必须把改动**同步回本目录**（以及 `examples/snn_triton_pipeline/_a100_backup/` 中对应的
> 原始镜像），保持文档与服务器实际状态一致。详见
> [`../../README.md`](../../README.md) 的「A100 脚本同步」一节。

## 模型与环境

| 项 | 值 |
|---|---|
| 模型 | SEW-ResNet-34（ImageNet-1K, T=4, connect_f=ADD, IF 神经元, 硬复位 θ=1.0） |
| checkpoint | `/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth` |
| 输入变换 | `r256_bilinear`（Resize256 → CenterCrop224 → Normalize） |
| 参考精度 | eager top-1 ≈ **67.7%** / top-5 ≈ 87.8% |
| 三条路径用到的 conda 环境 | 见下 |

- **eager 基线**（`run_sew.py`）：A100 env `snn-infer`（torch 2.4.1 + spikingjelly 0.0.0.0.12）。
- **整网 Triton 流水线**（`run_sew_triton.py`）：env `triton-src`（torch 2.12 + 源码 triton 3.7 + spikingjelly 源码）。
- **snn_compiler 加速实验**（`sew_exp.py` 等）：env `triton-src`，需要 `snn_compiler` 包在
  `PYTHONPATH`（A100 上为 `~/charlley/snn_compiler_test/snn_compiler`）。

---

## 模型专属脚本

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| **`run_sew.py`** | **eager 基线推理**。加载 ckpt → SpikingJelly eager（torch 神经元后端）→ ImageNet val 测 top-1/5。SEW 的 `.module.` key 改名以适配新版 SeqToANNContainer。 | `python run_sew.py --n 2000 --bs 100` |
| **`run_sew_triton.py`** | **整网走 Triton**。`sj_compat` 强制神经元 triton 后端；可选 `--compile` 用 inductor max-autotune（关 cudnn、GEMM/conv 限定 Triton、`compile_threads=1`）；`--profile` 统计 CUDA kernel 中 triton/cublas/cudnn 占比；`--amp bf16`。 | `python run_sew_triton.py --n 2000 --bs 50 --compile --amp bf16 --triton-conv` |
| **`sew_exp.py`** ★ | **snn_compiler 加速主实验**。构建参考网（eager）+ 两个加速变体：**EXACT**（conv+BN 仍 eager，仅把 IF 换成 snn_compiler 融合 Triton kernel → **逐位精确**）与 **FOLD**（完整 `FusedConvBNNeuron`，折 BN，可 bf16+NHWC）。`--mode correctness` 比对输出，`--mode speed` 带预热的中位数测速。定义的 `SEWExact`/`SEWFold` 被 `accuracy.py` 复用。**关键：SEW 残差加在两个神经元之后，且 downsample 自带神经元，所以用 `FusedConvBNNeuron`+普通 `+`，不能用 `FusedConvBNAddNeuron`。** | `python sew_exp.py --mode correctness`<br>`python sew_exp.py --mode speed --bs 64` |
| **`accuracy.py`** | 真实 ImageNet 上对比 **ref / EXACT / FOLD-bf16** 的 top-1/5 与逐样本 argmax 一致率（验证 EXACT 逐位、量化 FOLD 的精度影响）。 | `python accuracy.py --num 5000 --bs 50` |
| **`smoke.py`** | 冒烟测试：snn_compiler IF kernel vs naive 参考（maxdiff 应为 0）+ `FusedConvBNNeuron` 能跑通；打印 torch/triton 版本。 | `python smoke.py` |
| **`refload.py`** | 最小化验证：加载参考 SEW 模型并前向一次，确认 ckpt 加载（missing/unexpected=0）。 | `python refload.py` |
| **`debug_layers.py`** | 单神经元等价性（SJ `IFNode` vs `if_lif`，maxdiff=0）+ block / downsample 结构 introspection。 | 定位用 |
| **`debug2.py`** | **逐层散度定位**：对参考网挂 forward hook，逐 stage（stem/maxpool/layer1-4）对比 fused 中间激活，确认拓扑映射正确（各级发放率一致）。 | 定位用 |
| **`debug3.py`** | **机理定位**：证明 (A) 把参考 pre-activation 喂进 `if_lif` 逐位一致；(B) BN 折叠后的 pre-activation 偏差 ~1e-3；(C) conv+BN 分开算则逐位一致 → 锁定"BN 折叠在硬阈值下翻转脉冲"。 | 定位用 |

### 这次 snn_compiler 实验的结论（A100, bs=64, T=4）

| 配置 | ms/img | 加速比 | 正确性 |
|---|--:|--:|---|
| ref eager fp32 | 0.718 | 1.00× | 基线 |
| **EXACT（Triton IF，逐位精确）** | 0.410 | **1.75×** | max\|Δ\|=0，真实 top-1 与 ref 完全一致 |
| FOLD bf16+NHWC | 0.312 | **2.30×** | 总体精度不降（5000 张 67.34% vs 67.70%），但逐样本预测 80% 一致（脉冲洗牌） |

详见仓库根 memory `snn-compiler-bnfold-not-bitexact` 与对话记录。

---

## 共享基础设施（`_common/`，三模型相同）

| 文件 | 作用 |
|---|---|
| `sj_compat.py` | 把旧 SpikingJelly API（`clock_driven`/`cext`）shim 到 `activation_based`，并按 env `SJ_NEURON_BACKEND`（`triton`/`torch`）把神经元设为多步 + 指定后端。**必须在 import repo 模型之前 import**。 |
| `timm_compat.py` | 提供最小 timm 符号 shim（`to_2tuple`/`trunc_normal_`/`DropPath`/`register_model`/`_cfg`），让 transformer repo 不依赖真 timm（SEW 不需要，留着无害）。 |
| `snn_eval_lib_triton.py` | Triton 流水线版 eval：ImageNet val Dataset/loader、`evaluate()`（top-1/5）、`profile_kernels()`（把 CUDA kernel 归类 triton/cublas/cudnn）、amp 上下文、`reset_net`。`run_*_triton.py` 用它。 |
| `snn_eval_lib.py` | eager 版 eval（更简单），`run_*.py` 用它。 |
| `capture_ir.py` | 构建模型（sew/sf/sdtv2）→ `torch.compile` max-autotune → 跑一个 bf16 batch，触发 `TRITON_CACHE_DIR` / inductor debug 的 IR 落盘（`.ttir/.ttgir/.llir/.ptx` + `output_code.py`）。 |
| `capture_both.sh` | 用 IR 捕获 env 跑 `capture_ir.py`（sf + sdtv2）。 |
| `run_all_bf16_2k.sh` | 一键跑三模型 n=2000 bf16 `--compile` 评测。 |
| `spikingjelly_triton_utils.elementty.patch` | 修 SJ 自带手写 triton 神经元 kernel（`convert_and_store` 多写一层 `.element_ty`），否则其 triton 后端在 triton 3.7 下编不过。用 `SJ_NEURON_BACKEND=triton` 前需对 spikingjelly 源码打此补丁。 |
| `PIPELINE_OVERVIEW.md` | A100 `snn_infer_triton/README.md` 原文（整条 Triton 流水线方法论）。 |

---

## 来源（A100, `liushifeng@a100`）

- `run_sew.py`, `snn_eval_lib.py` ← `~/charlley/snn_infer/`
- `run_sew_triton.py` 及 `_common/` 多数 ← `~/charlley/snn_infer_triton/`
- `sew_exp.py`/`accuracy.py`/`smoke.py`/`refload.py`/`debug*.py` ← `~/charlley/snn_compiler_test/`
- `.elementty.patch` ← `~/lsf/inference_code_triton/`
- 连接：`ssh -p 3004 -o "ProxyJump charlley@180.76.139.31:40022" liushifeng@172.18.23.247`
