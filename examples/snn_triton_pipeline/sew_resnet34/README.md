# SEW-ResNet-34：从 SpikingJelly 到 GPU 的完整下降流程

> 本目录用**真实运行截获**的代码/IR 说明 SEW-ResNet-34（ImageNet-1K，T=4，connect_f=ADD）这一轮推理从 SpikingJelly 框架一路下降到 GPU 执行的每一步。
> 所有 `artifacts/` 文件都来自一次真实的编译运行（`triton-src` 环境，A100，bf16 + `torch.compile`），从 `TRITON_CACHE_DIR` / Inductor `TORCH_COMPILE_DEBUG` 目录直接提取，**非人工构造**。
> 捕获脚本：本仓库 `examples/snn_infer_triton/`（即 a100 上 `~/charlley/snn_infer_triton/capture_ir.py`）。完整大文件（fx_graph/output_code 等）见 a100 `~/charlley/snn_infer_triton/capture/sew/`。

本轮与既有 `examples/spikingjelly_triton/analysis/`（单层 SimpleSNN）的区别：那里是**训练 + 反向**的玩具模型；这里是**真实预训练权重的整网推理**，且通过 `torch.compile + inductor`（关 cudnn、强制 Triton GEMM/conv、bf16）把**整个网络**——包括卷积、BN、残差加、脉冲神经元——全部下降为 Triton kernel。该模型 profile 实测 **100% Triton，0 cublas/0 cudnn**。

---

## 模型结构与脉冲神经元（L1 用户代码）

SEW-ResNet-34 = 标准 ResNet-34 骨干 + 用脉冲神经元替换 ReLU + SEW 残差（脉冲逐元素相加）。每个 BasicBlock：

```
conv3x3 → BN → IF神经元(sn1) → conv3x3 → BN → IF神经元(sn2) → out += identity(脉冲)
```

源码：`Spike-Element-Wise-ResNet/imagenet/sew_resnet.py`（`SEWResNet`/`BasicBlock`）。
神经元用 SpikingJelly 的**多步 IF 节点**。本轮通过 `sj_compat.py` 把原仓库的旧 API 映射成新 `activation_based` API，并**强制 Triton 后端**：

```python
# sj_compat.py：原仓库 cext.MultiStepIFNode(detach_reset=True) →
class MultiStepIFNode(activation_based.neuron.IFNode):
    def __init__(self, ...):
        super().__init__(..., step_mode='m', backend='triton')   # ← 关键：多步 + Triton 后端
```

IF 充电/发放/重置（与 LIF 不同点：充电无 τ 衰减，直接 `H=V+X`）：
```
H[t] = V[t-1] + X[t]                 # 充电（积分）
S[t] = (H[t] >= v_threshold)         # 发放（Heaviside）
V[t] = S[t]*v_reset + (1-S[t])*H[t]  # 硬重置
```

整网前向把输入沿时间维复制 T=4 份 `x.unsqueeze(0).repeat(T,1,1,1,1)`，最后 `fc(x.mean(dim=0))` 对时间求平均。

---

## L4 SpikingJelly 的 Triton 神经元 kernel（手写）

神经元动力学不是交给 inductor 生成，而是 SpikingJelly **自带的手写 Triton kernel**
（`spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py`），整段时间循环 `tl.static_range(0,T)` 融合在**一个** kernel 内（避免逐时间步的全局内存往返）：

```python
@triton.jit
def _multistep_if_forward_kernel(x_seq_ptr, v_init_ptr, s_seq_ptr, h_seq_ptr, v_seq_ptr,
                                 v_threshold, v_reset, T: tl.constexpr, NCL, BLOCK_NCL, ...):
    v = tl.load(v_init_ptrs, ...)                 # 膜电位初值 (=0)
    for t in tl.static_range(0, T, 1):            # 时间维在 kernel 内展开
        x = tl.load(x_ptrs, ...)
        h = v + x                                 # 充电
        s = (h >= v_threshold).to(dtype)          # 发放
        v = s * v_reset + (1.0 - s) * h           # 硬重置
        convert_and_store(s_ptrs, s, ...)         # 写回脉冲
```

> 注：本轮还修了这个 kernel 在 triton 3.7 下的一个 bug（`convert_and_store` 多写一层 `.element_ty`，否则该后端编译不过），见 `lsf/inference_code_triton/spikingjelly_triton_utils.elementty.patch`。

---

## 完整下降链（每一层都有 `artifacts/` 真实文件）

```
L1 用户代码 (sew_resnet.py + sj_compat 强制 triton 神经元)
      │  torch.compile  →  TorchDynamo 截获字节码
L2 ATen FX 图  ............................  artifacts/fx_graph.excerpt.py
      │  TorchInductor：算子融合 + 代码生成（max-autotune, conv/GEMM 限定 Triton, cudnn 关）
L3 Inductor 输出（调度 wrapper + 各 Triton kernel 源码 + 启动序列）
      │                                       artifacts/output_code.excerpt.py
      ├─ 神经元 kernel 被 inductor 包裹并 autotune（BLOCK_NCL∈{128,256,512}）
      ├─ triton 卷积模板 triton_tem_fused__to_copy_convolution
      └─ 融合逐元素 kernel triton_poi_fused__to_copy_add_convolution_view（含 SEW 残差加）
      │  triton.compile()  →  MLIR pipeline
L5 TTIR (tt dialect) ......................  artifacts/neuron_if_kernel.ttir
      │  convert-triton-to-triton-gpu（插入 GPU 线程块布局）
L6 TTGIR (ttg dialect) ....................  artifacts/neuron_if_kernel.ttgir
      │  convert-triton-gpu-to-llvm
L7 LLVM IR ................................  artifacts/neuron_if_kernel.llir
      │  LLVM NVPTX 后端（target sm_80 = A100）
L8 PTX ....................................  artifacts/neuron_if_kernel.ptx
      │  ptxas
L9 CUBIN → GPU 执行
```

### L2：ATen FX 图（`artifacts/fx_graph.excerpt.py`）
TorchDynamo + AOTAutograd 把整网展平成 ATen 算子 DAG；脉冲神经元作为**不透明 custom op** 出现（不被拆成 aten 算子），其来源行号被忠实记录：
```python
# File: .../spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py:453
#   in multistep_if, code: s_seq, v_seq = multistep_if_inference(...)
```
ResNet-34 共有多处 IF 节点，故图中出现多次这样的 custom-op 调用。

### L3：Inductor 输出 + 真实 GPU 启动序列（`artifacts/output_code.excerpt.py`）
Inductor 把神经元 kernel **嵌入并 autotune**（`inductor_meta` 里 `declared_constexpr_names=['T','NCL','BLOCK_NCL','dtype','soft_reset','save_intermediates']`，候选 `BLOCK_NCL∈{128,256,512}, num_warps∈{4,8}`）。`call()` 是真实的逐 kernel 启动序列，stem 部分：
```python
triton_poi_fused__to_copy_0.run(arg1_1, buf0, ...)            # 输入 fp32→bf16 (NHWC)
triton_tem_fused__to_copy_convolution_2.run(buf0, buf1, buf2) # 7×7 stem 卷积（Triton 模板）
triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf3, ...)          # BN（推理）
triton_poi_fused_..._full_like_repeat_unsqueeze_4.run(buf3, buf6, ...)          # 沿 T 复制 + v 初值
_multistep_if_forward_kernel_0.run(buf6, buf7, buf4, buf5, buf5, 1.0, 0.0, ...) # 脉冲神经元
triton_poi_fused_max_pool2d_with_indices_view_6.run(buf4, buf10, ...)           # maxpool
... 下一个 BasicBlock ...
```
全部是 `triton_poi_* / triton_tem_* / _multistep_if_forward_kernel`——没有 `extern_kernels.convolution`(cudnn)、没有 `extern_kernels.mm`(cublas)。SEW 的脉冲残差加 `out += identity` 被融进 `triton_poi_fused__to_copy_add_convolution_view`。

### L5：TTIR — IF 动力学（`artifacts/neuron_if_kernel.ttir`，bf16）
时间维 T=4 已被**完全展开**，每步是充电/发放/重置三段（真实片段）：
```mlir
%h    = arith.addf %v_14, %x_16 : tensor<1x256xbf16>      // H = V + X  充电
%s_19 = arith.cmpf oge, %s_17, %s_18 : tensor<1x256xf64>  // S = (H >= v_threshold) 发放
%v_25 = arith.subf %cst_3, %s_20                          // (1 - S)
%v_26 = arith.mulf %v_25, %h                              // (1-S)*H
%v_24 = arith.mulf %s, %v_reset                           // S*v_reset
%v_29 = arith.addf %v_24, %v_28                           // V = S*v_reset + (1-S)*H 硬重置
tt.store %1, %s_20, ...                                   // 写脉冲
tt.store %3, %value, ...                                  // 写膜电位
// 接着 x_32/h_35/s_36... 即下一个时间步（共 4 段）
```

### L6：TTGIR — GPU 线程块布局（`artifacts/neuron_if_kernel.ttgir`）
```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
```
每 CTA 处理 `2×32×4 = 256` 个元素（=BLOCK_NCL）；warp 内 32 线程访问连续地址 → **合并访存**。

### L8：PTX — 落到 A100 指令（`artifacts/neuron_if_kernel.ptx`，`.target sm_80`）
```ptx
fma.rn.bf16x2  %r21, %r1, %r20, %r3;          // 充电 H=V+X，打包 2×bf16 向量 FMA
setp.le.f64    %p4, %rd19, %rd25;             // 发放比较 v_threshold <= H
selp.b16       %rs3, 0x3F80, 0x0000, %p5;     // 选脉冲值 1.0/0.0 (0x3F80=bf16 1.0)
selp.f64       %rd26, 0d3FF0..., 0d0..., %p4; // 重置选择
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];     // 全局读
@%p1 st.global.v2.b32 [...], { ... };         // 向量化写回
```

---

## 卷积/矩阵乘也走 Triton（不是 cudnn/cublas）

- `artifacts/triton_convolution2d.{ttir,ptx}` — ResNet 的 NxN 卷积用 **Triton 卷积模板**（不是 cudnn）。关键前提：**bf16** 把模板的共享内存占用减半，fp32 下大 block 配置需 256–288KB 超 A100 上限 163KB 会 OOM 回退 ATEN。
- `artifacts/triton_mm.{ttir}` — 最后的 `fc` 全连接用 **Triton matmul 模板**（不是 cublas）。

> 实测该模型在 bf16 下整网 100% Triton（neuron + 卷积模板 + matmul 模板 + 融合逐元素），cublas/cudnn 时间占比 0。
