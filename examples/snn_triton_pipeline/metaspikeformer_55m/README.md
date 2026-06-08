# Meta-SpikeFormer-55M (Spike-Driven-Transformer-V2)：从 SpikingJelly 到 GPU

> 真实运行截获。Meta-SpikeFormer-55M（ImageNet-1K，T=4，KD，`metaspikformer_8_512`）一轮 bf16 + `torch.compile` 推理的逐层 IR，来自 a100 `~/charlley/snn_infer_triton/capture/sdtv2/`。
> 本模型是三者中**唯一无法 100% Triton** 的：它的 `SepConv` 含 **depthwise（分组）卷积**，inductor 没有 depthwise 卷积的 Triton 模板，只能回退 ATEN 原生 kernel——但仍**不是 cudnn / cublas**。实测 profile：Triton 64% / ATEN-depthwise 36% / **cublas 0 / cudnn 0**。

---

## 模型结构与脉冲神经元（L1 用户代码）

源码 `Spike-Driven-Transformer-V2/classification/models.py`（`Spiking_vit_MetaFormer`）。前段是 MS_ConvBlock（含 `SepConv`：逐点 1×1 + **7×7 depthwise** + 1×1），后段是 MS_Block（脉冲注意力 + MLP）。神经元是多步 **LIF**（`tau=2.0`），经 `sj_compat.py` 强制 `step_mode='m', backend='triton'`。
推理时 `model.T=4`，前向沿时间维复制 T 份，最后对 T 求平均（KD 头与主头平均）。

LIF 充电/发放/重置同 Spikingformer（见 [`../spikingformer_8_768/`](../spikingformer_8_768/)）：`H=V+(1/τ)(v_reset−V+X)` → `S=(H≥v_th)` → `V=S·v_reset+(1−S)·H`。

---

## 完整下降链（`artifacts/` 真实文件）

```
L1 models.py + sj_compat（LIF→triton）            L2 ATen FX 图 ........ artifacts/fx_graph.excerpt.py
L3 Inductor output_code ....... artifacts/output_code.excerpt.py
L5/6/7/8 神经元 LIF kernel ..... artifacts/neuron_lif_kernel.{ttir,ttgir,llir,ptx}  (.target sm_80)
ANN 算子：artifacts/triton_mm.{ttir,ptx}（注意力/线性，Triton）、artifacts/sepconv_pointwise_conv.ttir（1×1，Triton 模板）
```

神经元/矩阵乘/逐点卷积的下降与前两个模型一致（LIF kernel TTIR 见 `artifacts/neuron_lif_kernel.ttir`，与 Spikingformer 逐位同构）。本模型的**特别之处**在下面这条"无法 Triton 化"的路径。

---

## 关键点：depthwise 卷积 = 唯一回退 ATEN 的算子

`SepConv` 的 7×7 **depthwise**（分组）卷积，inductor 无 Triton 模板，落到 ATEN。真实 wrapper（`artifacts/output_code.excerpt.py`）：
```python
# 全部 72 个 extern_kernels 调用都是 groups>1 的 depthwise 卷积，没有一个是 mm/bmm：
buf20 = extern_kernels.convolution(buf19, buf18, stride=(1,1), padding=(3,3),
                                   dilation=(1,1), transposed=False, output_padding=(0,0),
                                   groups=128, bias=None)        # ← depthwise (groups=通道数)
```
统计：`extern_kernels.convolution` × 72，`extern_kernels.mm/bmm/addmm` × **0**。
关闭 cudnn 后，这个 `aten.convolution(groups>1)` 在运行时 dispatch 到 **`at::native::conv_depthwise2d_forward_kernel`**（CUDA 原生 kernel），即 profile 里那 36% 的 "other"——**它既不是 cudnn 也不是 cublas**。

`call()` 真实序列因此是「Triton kernel 之间插入若干 `extern_kernels.convolution(groups=…)`」：
```python
triton_tem_fused__to_copy_convolution_*.run(...)     # 1×1 / 普通卷积：Triton 模板
_multistep_lif_forward_kernel_*.run(..., 2.0, 1.0, 0.0)   # LIF 神经元：Triton（手写）
extern_kernels.convolution(..., groups=128, ...)      # 7×7 depthwise：ATEN 原生（无 Triton 模板）
triton_mm_*.run(...)                                  # 注意力/线性：Triton matmul 模板
```

---

## 结论

| 算子类别 | 后端 | 占比(单 batch) |
|---------|------|------|
| 脉冲 LIF 神经元 | Triton（SpikingJelly 手写 kernel） | 计入 64% |
| 1×1 卷积 / 注意力 / 线性 | Triton（inductor matmul/conv 模板） | 计入 64% |
| 逐元素 / BN / T 复制平均 | Triton（inductor 融合） | 计入 64% |
| **7×7 depthwise 卷积** | **ATEN 原生**（无 Triton depthwise 模板） | **36%** |
| cublas / cudnn | — | **0 / 0** |

要让这 36% 也走 Triton，需要为 depthwise/grouped 卷积**手写 Triton kernel**（inductor 现成模板覆盖不到）——这是把脉冲 Transformer 完全 Triton 化的唯一缺口。其余路径（神经元 + 矩阵乘 + 普通卷积 + 逐元素）均已是 Triton。
