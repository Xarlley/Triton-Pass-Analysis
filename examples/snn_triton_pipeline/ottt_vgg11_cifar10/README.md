# OTTT VGG-11-WS (CIFAR-10)：从 SpikingJelly 到 GPU 的完整下降流程

> 真实运行截获。OTTT-A VGG-11-WS（CIFAR-10，T=6）一轮 bf16 + `torch.compile` 推理的逐层 IR，全部来自 a100 `~/charlley/snn_infer_triton/capture/ottt/`（`artifacts/` 为精选）。
> 这是本系列第 4 个模型、第 1 个 **CNN-SNN（权重标准化 VGG）**。原仓库 [pkuxmq/OTTT-SNN](https://github.com/pkuxmq/OTTT-SNN)（NeurIPS'22）。

## 这一轮做了什么（与前三个模型的不同）

前三个模型（SEW/Spikingformer/SDT-V2）原本就用 spikingjelly 的多步神经元，所以只需把 cupy/cext 后端切到 triton。
**OTTT 不一样**：它的神经元是自定义的 **OnlineLIFNode**（在线训练，单步、外层循环 T 次），不是 spikingjelly 多步节点。所以本轮是真正的**重写**——把它改写成 spikingjelly `activation_based` 的**多步 LIF + Triton 后端**，并验证与原模型逐样本一致。

- 重写代码：`scripts/ottt_vgg_triton.py`（清晰的多步 WS-VGG-11 + 检查点按位置重映射）。
- 等价性：`scripts/verify_equiv.py` 用同一权重、同一输入对比"原 OnlineLIFNode 循环 T 次" vs "我的多步模型"，**argmax 全部一致**（logits 仅 FP 排序级差异）。
- 精度复现（CIFAR-10 测试集 10000 张，无损，从 HF parquet 解码）：

| 配置 | top-1 |
|------|-------|
| 原论文 (OTTTA, VGG, T=6) | 93.52% |
| 重写, 神经元 backend=torch | **93.60%** |
| 重写, 神经元 backend=**triton** | **93.60%**（与 torch 逐位一致）|
| 整网 triton（bf16 + compile, conv/GEMM 全 triton）| **93.70%** |

> 单 batch kernel 归类：**100% Triton，0 cublas，0 cudnn**（`output_code` 里 `extern_kernels` 调用 = 0）。

## 模型结构与神经元映射（L1）

VGG-11 配置 `A=[64,128,'M',256,256,'M',512,512,'M',512,512]`，每个卷积层 = `ScaledWSConv2d → LIF → Scale(2.74)`，'M'=AvgPool2d，头部 `AdaptiveAvgPool2d(1)→Linear(512,10)`。
- **ScaledWSConv2d**（权重标准化）：`w' = (w-mean)/sqrt(var·fanin+eps)·gain`（逐输出通道，var 无偏）。这是 OTTT 的特色算子，检查点里每个卷积带 `weight/bias/gain` 三件套。
- **神经元等价**：OnlineLIFNode `v=v·(1-1/τ)+x`、`v_threshold=1`、`v_reset=None`（**软重置**）、τ=2，逐时间步循环 == spikingjelly `LIFNode(tau=2, decay_input=False, v_threshold=1.0, v_reset=None, step_mode='m', backend='triton')`（前向无跨层时间反馈，时间维展开等价）。

## 完整下降链（`artifacts/` 真实文件）

```
L1 scripts/ottt_vgg_triton.py（多步 WS-VGG + LIF triton）
L2 ATen FX 图 ......... artifacts/fx_graph.excerpt.py   （WS 权重标准化 + 不透明 LIF op）
L3 Inductor 输出 ...... artifacts/output_code.excerpt.py（LIF kernel + WS-conv 融合 + call() 启动序列）
L5 TTIR .............. artifacts/neuron_lif_kernel.ttir
L6 TTGIR ............. artifacts/neuron_lif_kernel.ttgir
L7 LLVM IR ........... artifacts/neuron_lif_kernel.llir
L8 PTX (sm_80) ....... artifacts/neuron_lif_kernel.ptx
ANN 算子：artifacts/triton_convolution2d.{ttir,ptx}（WS 3×3 卷积，Triton 模板）、triton_mm.ttir（头部）
```

### L2 亮点：权重标准化被编译进图（`artifacts/fx_graph.excerpt.py`）
OTTT 的 WS 卷积不是黑盒——`get_weight()` 的 mean/var/归一化/gain 被 TorchDynamo 追踪成 ATen 算子：
```python
mean = torch.ops.aten.mean.dim(arg1_1, [1, 2, 3], True)
var  = torch.ops.aten.var.correction(arg1_1, [1, 2, 3], correction = 1, keepdim = True)  # 无偏方差
# ... (w-mean)/sqrt(var*fanin+eps) * gain ...  (File: ottt_vgg_triton.py:31)
```
LIF 神经元则是不透明 custom op：`multistep_lif_inference (lif.py:497)`。

### L3：真实 GPU 启动序列（`artifacts/output_code.excerpt.py`）
`call()` 把权重标准化 + 卷积 + 神经元全部交给 Triton（节选）：
```python
triton_per_fused_mean_var_0.run(arg1_1, buf0, buf2, ...)                 # WS: 求 weight 的 mean/var
triton_poi_fused__to_copy_repeat_unsqueeze_view_2.run(arg0_1, buf5, ...) # 输入→bf16 + 沿 T=6 复制
triton_poi_fused__to_copy_add_div_mean_mul_pow_sub_var_3.run(...)        # WS: (w-mean)/sqrt(var*fanin+eps)*gain
triton_tem_fused__to_copy_add_convolution_..._var_4.run(buf5,buf6,buf7)  # WS 3×3 卷积（Triton 模板，与权重标准化融合）
_multistep_lif_forward_kernel_0.run(buf10, buf11, ..., 2.0, 1.0, 0.0)    # LIF 神经元（tau=2, vth=1, vreset=0）
...  # AvgPool / 下一层 / ... / 头部 triton_mm
```
全部是 `triton_*` / `_multistep_lif_forward_kernel`——`extern_kernels`（cublas/cudnn）出现 **0** 次。

### L5 TTIR：LIF 充电（带 τ 衰减）+ **软重置**（`artifacts/neuron_lif_kernel.ttir`，bf16，T=6 展开）
```mlir
%r_tau_6 = arith.divf %r_tau, %tau                  // 1/τ  (τ=2)
%h_22 = arith.subf %h_21, %h_20                      // v_reset - v
%h_26 = arith.mulf %h_25, %h_22                      // (1/τ)·(v_reset - v)
%h_27 = arith.addf %h_20, %h_26                      // v + (1/τ)(v_reset - v)
%h_30 = arith.addf %h_27, %x                         // H = ... + x      （decay_input=False）
%s_31 = arith.cmpf oge, %h_30, %v_threshold          // S = (H >= 1.0)
%v_35 = arith.mulf %s, %v_threshold                  // S·v_threshold
%v_36 = arith.subf %h_30, %v_35                      // V = H - S·v_threshold   ← 软重置！
```
> 对比：SEW/MS-ResNet 是**硬重置**（`V=(1-S)·H`，用 select/乘减实现）；OTTT 是**软重置**（`V=H-S·v_th`），这正是两类 LIF 在 IR 上的可见区别。

### L8 PTX：`.target sm_80`（A100），`div.rn.f64`（1/τ）、`setp`/`selp`（发放）、`fma.rn.bf16x2`（打包充电）。

## 结论
OTTT VGG-11-WS 完整跑通 spikingjelly-Triton 重写：神经元 triton 后端与原实现逐样本一致，整网（含权重标准化、卷积、AvgPool、头部）100% Triton、0 cublas/cudnn，CIFAR-10 精度 93.6–93.7%（论文 93.52%）。
