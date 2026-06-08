# SNN → SpikingJelly → IR → GPU：三个预训练模型的真实下降流程

本目录用**真实运行截获**的代码与各级 IR，说明本轮验证的三个 SNN 预训练模型，如何从 SpikingJelly 框架开始、经过 `torch.compile`/Inductor/Triton 的逐层降级，最终落到 A100 GPU 上执行。

三个子目录对应三个模型：

| 子目录 | 模型 | 类型 | 脉冲神经元 | 主算力 |
|--------|------|------|-----------|--------|
| [`sew_resnet34/`](./sew_resnet34/) | SEW-ResNet-34 (T=4, ADD) | 脉冲 **CNN** | 多步 **IF**（无 τ） | 3×3 卷积 |
| [`spikingformer_8_768/`](./spikingformer_8_768/) | Spikingformer-8-768 (T=4) | 脉冲 **Transformer** | 多步 **LIF**（τ=2） | 1×1 卷积 + 注意力矩阵乘 |
| [`metaspikeformer_55m/`](./metaspikeformer_55m/) | Meta-SpikeFormer-55M / SDT-V2 (T=4) | 脉冲 **Transformer**（含 SepConv） | 多步 **LIF** | 矩阵乘 + 卷积 + depthwise 卷积 |

> 与既有 [`examples/spikingjelly_triton/analysis/`](../spikingjelly_triton/analysis/)（单层 `SimpleSNN`、训练+反向的玩具示例）相比：本目录是**真实预训练权重的整网推理**，且目标是把**整个网络**（卷积/BN/残差/注意力/脉冲神经元）尽量都下降为 Triton kernel、不调用 cudnn/cublas。本目录复用并印证了那套「用户代码 → FX → Inductor → TTIR → TTGIR → LLVM IR → PTX → CUBIN」的方法论。

---

## 实验脚本（A100 快照）

A100 上每个模型的全部实验脚本，已整理进各自的 `scripts/`，并附逐脚本说明文档：

| 模型 | 脚本目录 | 含 eager 基线 | 含整网 Triton | 含 snn_compiler 加速实验 |
|---|---|:--:|:--:|:--:|
| SEW-ResNet-34 | [`sew_resnet34/scripts/`](./sew_resnet34/scripts/) | ✅ | ✅ | ✅（`sew_exp.py` 等，EXACT 1.75× 逐位精确 / FOLD 2.3×） |
| Spikingformer-8-768 | [`spikingformer_8_768/scripts/`](./spikingformer_8_768/scripts/) | ✅ | ✅ | — |
| Meta-SpikeFormer-55M | [`metaspikeformer_55m/scripts/`](./metaspikeformer_55m/scripts/) | ✅ | ✅ | — |

- 每个 `scripts/` 含模型专属脚本 + `_common/`（三模型共享的 `sj_compat.py` / `snn_eval_lib*.py` /
  `timm_compat.py` / `capture_ir.py` / 补丁 等，各副本内容相同），并有 `README.md` 逐文件说明。
- [`_a100_backup/`](./_a100_backup/) 是 A100 工作区（`snn_infer/`、`snn_infer_triton/`、
  `snn_compiler_test/`、`inference_code_triton/`）的**逐文件原始镜像**（单一备份），保留出处与
  `run_sdtv3.py` 等未归入三模型的脚本。
- 脚本中的路径是 **A100 绝对路径**，仅供在 A100 复跑（见
  [`remote_gpu_servers/a100-liushifeng.md`](../../remote_gpu_servers/a100-liushifeng.md)）。

### ⚠️ A100 脚本同步（给 Claude 的指示）

A100 上的脚本是"真源"。**每当在 A100 新增 / 修改 / 删除任何实验脚本，必须同步本仓库**：

1. 更新 `_a100_backup/<workspace>/…` 中对应文件，使其与 A100 逐字节一致；
2. 更新对应模型 `<model>/scripts/`（共享脚本则更新所有三个模型的 `_common/` 副本）；
3. 若脚本作用变化，更新对应 `scripts/README.md`。

反之，从本仓库改了脚本要在 A100 上跑，也要先推回 A100 对应路径。

---

## 统一的下降管线

```
L1  用户 Python（SpikingJelly 模型，神经元 backend='triton', step_mode='m'）
        │  torch.compile → TorchDynamo 截获字节码
L2  ATen FX 图（脉冲神经元 = 不透明 custom op：multistep_{if,lif}_inference）
        │  TorchInductor：算子融合 + 代码生成
L3  Inductor 输出 output_code.py（调度 wrapper `call()` + 各 Triton kernel 源码 + 启动序列）
        │  triton.compile()  →  Triton MLIR pipeline
L5  TTIR（tt dialect，逻辑张量，无 GPU 布局）
        │  convert-triton-to-triton-gpu
L6  TTGIR（ttg dialect，插入 #blocked 线程块布局 → 合并访存）
        │  convert-triton-gpu-to-llvm
L7  LLVM IR
        │  LLVM NVPTX 后端 → target sm_80 (A100)
L8  PTX
        │  ptxas
L9  CUBIN → GPU 执行
```

两类 kernel 来源：
- **脉冲神经元 kernel**（`_multistep_if/lif_forward_kernel`）：SpikingJelly **自带的手写 Triton kernel**，把整个时间维 `tl.static_range(0,T)` 融合在一个 kernel 内；被 Inductor 包裹并 autotune。
- **ANN 算子 kernel**（卷积/矩阵乘/BN/逐元素）：由 **TorchInductor codegen + Triton 模板**生成（`triton_tem_*`/`triton_poi_*`/`triton_mm`）。

---

## 本轮关键配置（让"整网走 Triton、不碰 cudnn/cublas"成立）

在 `triton-src` 环境（torch 2.12 + 源码编译 triton 3.7 + spikingjelly 源码版）下：
```python
torch.backends.cudnn.enabled = False                 # 关 cudnn
ic.max_autotune = True
ic.max_autotune_gemm_backends = "TRITON"             # 矩阵乘只用 Triton（排除 cublas）
ic.max_autotune_conv_backends = "ATEN,TRITON"        # 卷积优先 Triton 模板
ic.conv_1x1_as_mm = True                             # 1×1 卷积当矩阵乘 → Triton
ic.compile_threads = 1                               # 源码 triton 在 inductor 子进程编不了，必须主进程内编
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
with torch.autocast("cuda", dtype=torch.bfloat16): model(x)   # bf16：把 triton 模板共享内存占用减半 → 放得进 A100
```

三个踩坑（详见各模型 README 与 `remote_gpu_servers/a100-liushifeng.md`）：
1. **SpikingJelly 自带 triton kernel 的 bug**：`convert_and_store` 多写一层 `.element_ty`，triton 3.7 下其神经元 triton 后端根本编不过 → 已修。
2. **inductor 子进程编译 worker 找不到源码 triton 驱动**（`0 active drivers`）→ `compile_threads=1`。
3. **共享内存溢出**：fp32 下 triton conv/mm 模板大 block 需 256–288KB > A100 上限 163KB → OOM 回退 ATEN（=cudnn/cublas）。**改 bf16 占用减半即放得下**，cublas/cudnn 清零。

---

## 结果（ImageNet val 子集；profile = 单 batch CUDA kernel 时间占比）

| 模型 | top-1（triton 神经元, eager） | top-1（整网 triton, bf16+compile） | Triton 占比 | cublas | cudnn |
|------|------|------|------|------|------|
| SEW-ResNet-34 | 67.75% | 67.10% | **100%** | 0 | 0 |
| Spikingformer-8-768 | 75.95% | 76.55% | **100%** | 0 | 0 |
| Meta-SpikeFormer-55M | 79.90% | 79.85% | 64% | **0** | **0** |

- 神经元 Triton 后端与 torch 后端逐位一致（maxdiff=0），精度全部复现。
- 三模型 **cublas/cudnn 均为 0**。Meta-SpikeFormer 残留 36% 是 ATEN 的 **depthwise 卷积**原生 kernel（SepConv 的 7×7 深度可分离卷积，inductor 无对应 Triton 模板）——既非 cudnn 也非 cublas。

---

## 复现 / 重新截获

捕获脚本与运行代码在 a100 `~/charlley/snn_infer_triton/`（亦存 `lsf/inference_code_triton/`）：
```bash
conda activate triton-src
cd ~/charlley/snn_infer_triton
# 截获某模型的全部 IR 到 capture/<model>/（triton_cache + inductor debug）
env SJ_NEURON_BACKEND=triton TRITON_ALWAYS_COMPILE=1 \
    TRITON_CACHE_DIR=$PWD/capture/sew/triton_cache \
    TORCHINDUCTOR_CACHE_DIR=$PWD/capture/sew/inductor_cache \
    TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=$PWD/capture/sew/debug \
    python capture_ir.py sew --bs 8
```
- 各 kernel 的 `.ttir/.ttgir/.llir/.ptx/.cubin/.source` 在 `TRITON_CACHE_DIR/<hash>/`。
- `fx_graph_readable.py / ir_pre_fusion.txt / ir_post_fusion.txt / output_code.py` 在 inductor debug 目录。
- 各子目录 `artifacts/` 是从上述真实产物中精选的片段；完整大文件留在 a100 `capture/<model>/`。
