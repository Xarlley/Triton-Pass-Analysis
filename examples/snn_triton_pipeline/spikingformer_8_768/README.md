# Spikingformer-8-768：从 SpikingJelly 到 GPU 的完整下降流程

> 真实运行截获。Spikingformer-8-768（ImageNet-1K，T=4，dim=768，8 个 spiking transformer block）一轮 bf16 + `torch.compile` 推理的逐层 IR，全部来自 a100 `~/charlley/snn_infer_triton/capture/sf/`（`artifacts/` 为精选）。
> 与 SEW 的卷积网络不同，这里是**脉冲 Transformer**：神经元是 **LIF**（带 τ 衰减），主算力是 1×1 卷积（QKV/MLP 投影）与注意力矩阵乘——它们都被下降为 **Triton matmul 模板**（实测 wrapper 中 `extern_kernels`(cublas/cudnn) 调用数 = 0）。

---

## 模型结构与脉冲神经元（L1 用户代码）

源码 `Spikingformer/imagenet/model.py`。一个 block = Spiking Self-Attention (SSA) + Spiking MLP，残差相连：

```
x = x + SSA(x)        # SSA：LIF → 1×1Conv(q/k/v) → BN → LIF → 注意力 (k^T@v 再 q@·) → LIF → 1×1Conv 投影
x = x + MLP(x)        # MLP：LIF → 1×1Conv → BN → LIF → 1×1Conv → BN
```
patch embed (`SpikingTokenizer`) 是 4 级 `Conv3×3→BN→LIF→MaxPool` 下采样到 14×14。所有 LIF 节点 `tau=2.0`，注意力里的 `attn_lif` 用 `v_threshold=0.5`。

本轮经 `sj_compat.py` 把原仓库 `MultiStepLIFNode(tau=2.0, backend='cupy')` 映射为：
```python
class MultiStepLIFNode(activation_based.neuron.LIFNode):
    def __init__(self, tau=2.0, ...):
        super().__init__(tau=tau, decay_input=True, ..., step_mode='m', backend='triton')
```

LIF 充电/发放/重置（与 IF 的差别在充电带 τ 衰减，`decay_input=True`）：
```
H[t] = V[t-1] + (1/τ)·(v_reset − V[t-1] + X[t])   # 充电（漏积分）
S[t] = (H[t] >= v_threshold)                        # 发放
V[t] = S[t]·v_reset + (1−S[t])·H[t]                 # 硬重置
```

---

## L4 SpikingJelly 的 LIF Triton kernel（手写，时间维内融合）

`spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py`：
```python
@triton.jit
def _multistep_lif_forward_kernel(x_seq_ptr, v_init_ptr, s_seq_ptr, h_seq_ptr, v_seq_ptr,
                                  tau, v_threshold, v_reset, T: tl.constexpr, ...):
    r_tau = 1.0 / tau
    v = tl.load(v_init_ptrs, ...)
    for t in tl.static_range(0, T, 1):
        x = tl.load(x_ptrs, ...)
        h = v + r_tau * (v_reset - v + x)   # decay_input=True 充电
        s = (h >= v_threshold).to(dtype)    # 发放
        v = s * v_reset + (1.0 - s) * h     # 硬重置
        convert_and_store(s_ptrs, s, ...)
```

---

## 完整下降链（`artifacts/` 真实文件）

```
L1 model.py + sj_compat（LIF→triton 后端）
L2 ATen FX 图 ............ artifacts/fx_graph.excerpt.py    （LIF 作为 custom op：multistep_lif_inference）
L3 Inductor 输出 ......... artifacts/output_code.excerpt.py（LIF kernel 嵌入+autotune；call() 启动序列）
L5 TTIR .................. artifacts/neuron_lif_kernel.ttir
L6 TTGIR ................. artifacts/neuron_lif_kernel.ttgir
L7 LLVM IR ............... artifacts/neuron_lif_kernel.llir
L8 PTX (sm_80=A100) ...... artifacts/neuron_lif_kernel.ptx
L9 CUBIN → GPU
```

### L5：TTIR — LIF 充电带 τ 衰减（`artifacts/neuron_lif_kernel.ttir`，真实片段）
```mlir
%r_tau_5 = arith.divf %r_tau, %tau : f64                 // 1/τ
%h_21 = arith.subf %h_20, %h_19                          // v_reset - v
%h_24 = arith.addf %h_21, %h_23                          // (v_reset - v) + x
%h_28 = arith.mulf %h_27, %h_24                          // (1/τ)·(...)
%h_29 = arith.addf %h_19, %h_28                          // H = v + (1/τ)(v_reset - v + x)
%s_30 = arith.cmpf oge, %h_29, %s                        // S = (H >= v_threshold)
%v_34 = arith.mulf %v_33, %h_20                          // S·v_reset
%v_35 = arith.subf %cst_3, %s_31                         // 1 - S
%v_39 = arith.addf %v_34, (1-S)·H                        // V = S·v_reset + (1-S)·H
```
（对比 SEW 的 IF：那里充电是单条 `addf`，这里多了 `divf/subf/mulf` 的 τ 衰减项 —— 这正是 LIF vs IF 在 IR 上的唯一区别。）

### L8：PTX（`artifacts/neuron_lif_kernel.ptx`，`.target sm_80`）
```ptx
div.rn.f64  %rd19, %rd18, %rd17;        // 1/τ
setp.gt.f64 %p2, %rd20, %rd22;          // 发放比较
selp.b32    %r4, 1, -1, %p2;            // Heaviside 选择
```

---

## 注意力 / 投影：Triton matmul 模板（不是 cublas）

- `artifacts/triton_mm.{ttir,ptx}` —— SSA 里的 1×1 卷积（经 `conv_1x1_as_mm` 视作矩阵乘）与注意力 `k^T@v`、`q@(kv)`，全部走 **Triton matmul 模板**。
- `artifacts/patch_embed_conv.ttir` —— patch embed 的 3×3 卷积走 **Triton 卷积模板**（`triton_tem_fused__to_copy_convolution`）。

`call()` 真实启动序列（patch-embed 开头，全是 triton kernel）：
```python
triton_poi_fused__to_copy_repeat_unsqueeze_view_0.run(arg0_1, buf0, ...)      # 输入→bf16 + 沿 T 复制
triton_tem_fused__to_copy_convolution_repeat_unsqueeze_view_2.run(buf0,buf1,buf2) # patch-embed 卷积(Triton)
...                                                                            # LIF / 1×1conv(mm) / 注意力 mm / ...
```
> 实测：wrapper 中 `extern_kernels.{mm,bmm,addmm,convolution}` 出现 **0 次** —— 即整网 0 cublas / 0 cudnn（bf16 下注意力矩阵乘与卷积模板都放得进 A100 共享内存）。
