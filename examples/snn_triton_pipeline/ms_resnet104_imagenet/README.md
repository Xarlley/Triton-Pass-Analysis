# MS-ResNet-104 (ImageNet)：从 SpikingJelly 到 GPU 的完整下降流程

> 真实运行截获。MS-ResNet-104（ImageNet-1K，T=6，membrane-shortcut）一轮 bf16 + `torch.compile` 推理的逐层 IR，来自 a100 `~/charlley/snn_infer_triton/capture/msresnet/`。
> 本系列第 5 个模型、第 2 个 **CNN-SNN**，特点是 **膜电位残差（membrane shortcut）** + **TDBN**。原仓库 [Ariande1/MS-ResNet](https://github.com/Ariande1/MS-ResNet)（TNNLS'24）。

## 这一轮做了什么

MS-ResNet 的神经元是自定义的 `mem_update`（纯 torch 的时间循环），不是 spikingjelly 节点。本轮**重写**为 spikingjelly 多步 LIF + Triton 后端，方式是**最小侵入的 monkeypatch**（保留原模型结构与检查点键，只换神经元实现 + 把逐时间步卷积循环改成一次批量卷积）：
- `scripts/ms_resnet_triton.py`：把 `mem_update` 换成 spikingjelly `LIFNode(...,backend='triton')`，`Snn_Conv2d` 改批量卷积；`batch_norm_2d`(TDBN) 原样保留（编译友好）。检查点 **0 missing / 0 unexpected**。
- 神经元等价（`scripts/verify_equiv.py`）：原 `mem_update` vs `LIFNode(tau=4/3, decay_input=False, v_threshold=0.5, v_reset=0.0)` 在随机 [T,B,C,H,W] 上 **max|diff|=0、脉冲逐位相同（bit-exact）**。

为什么 τ=4/3：`mem_update` 充电 `H=decay·H_prev·(1-S_prev)+X`，decay=0.25。spikingjelly `LIFNode(decay_input=False)` 充电 `H=V·(1-1/τ)+X`，令 `1-1/τ=0.25 ⇒ τ=4/3`；硬重置到 0、阈值 0.5，与原式完全一致。

### 精度复现（ImageNet val 前 2000 张）
| 配置 | top-1 | top-5 |
|------|-------|-------|
| 原论文（全 5 万张）| 76.02% | — |
| 重写, 神经元 backend=torch | 74.20% | 92.20% |
| 重写, 神经元 backend=**triton** | **74.20%** | **92.20%**（与 torch 逐位一致）|
| 整网 triton（bf16+compile, N=500）| 74.00% | 92.60% |

> 子集 2000 张，74.2% 与论文 76.02%（全集）在抽样范围内。单 batch kernel 归类：**Triton 98.7% / cublas 0 / cudnn 0 / other 1.3%**。唯一非 Triton 的 1.3% 是 ATEN 的 `adaptive_avg_pool3d`（inductor 无自适应 3D 池化的 Triton 模板）——既非 cudnn 也非 cublas。

## 模型结构（L1）
- 3 个 3×3 卷积的 stem + TDBN；4 个 stage（block 数 3/8/32/8，BasicBlock）；头部 `fc(512,1000)`。T=6。
- **BasicBlock**：`mem_update→conv→TDBN→mem_update→conv→TDBN_zero`，残差 `out = residual_function(x) + shortcut(x)`——**相加发生在膜电位/模拟域**（两端都以 BN 结尾、不是脉冲），由下一个 block 的 `mem_update` 再转脉冲。这就是 membrane shortcut。
- **TDBN**（`batch_norm_2d`）= 对 (T·B,H,W) 逐通道归一化，等价于把 T 并入 batch 的 BatchNorm；推理期是逐元素仿射，inductor 直接融成 Triton。

## 完整下降链（`artifacts/` 真实文件）
```
L1 scripts/ms_resnet_triton.py（monkeypatch: mem_update→LIF triton, Snn_Conv2d→批量卷积, TDBN 保留）
L2 ATen FX 图 ......... artifacts/fx_graph.excerpt.py
L3 Inductor 输出 ...... artifacts/output_code.excerpt.py
L5 TTIR .............. artifacts/neuron_lif_kernel.ttir
L6 TTGIR ............. artifacts/neuron_lif_kernel.ttgir
L7 LLVM IR ........... artifacts/neuron_lif_kernel.llir
L8 PTX (sm_80) ....... artifacts/neuron_lif_kernel.ptx
ANN：artifacts/triton_convolution2d.{ttir,ptx}（3×3 卷积，Triton 模板）、tdbn_fused.ttir（TDBN 融合）、triton_mm.ttir（头部）
```

### L2：膜电位残差 + TDBN + 时间复制（`artifacts/fx_graph.excerpt.py`）
```python
add   = torch.ops.aten.add.Tensor(arg8_1, 1e-05)                       # TDBN: var+eps
add_1 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_11)                 # TDBN 仿射 (γ·x̂+β)
add_5 = torch.ops.aten.add.Tensor(mul_8, unsqueeze_35)                 # ... 另一支 BN
avg_pool3d = torch.ops.aten.avg_pool3d.default(permute_3, [1,2,2], [1,2,2])  # shortcut 下采样（逐 t 空间池化）
add_8 = torch.ops.aten.add.Tensor(permute_13, permute_17)                    # 膜电位残差 = residual_path + shortcut_path
# LIF 为不透明 custom op（triton_kernel_wrapper_functional，lif.py:497）
```
（输入沿 T=6 复制：FX 图里是 `select_scatter ×6` → `view` 到 [T·B,3,224,224]；在 `output_code` 里这些 select_scatter 记为 `_generalized_scatter`，由一个 `triton_poi_fused__to_copy_view_zeros_0` kernel 完成。）

### L5 TTIR：LIF 充电（decay 0.25）+ **硬重置**（`artifacts/neuron_lif_kernel.ttir`，T=6 展开）
```mlir
%r_tau_7 = arith.divf %r_tau, %tau          // 1/τ = 0.75  (τ=4/3) → 衰减 1-0.75=0.25
%h_23 = arith.subf ... ; %h_27 = arith.mulf ... ; %h_28 = arith.addf ...   // H = V·0.25 + X
%s_32 = arith.cmpf oge, %h_31, %v_threshold // S = (H >= 0.5)
%v_37 = arith.subf %cst_5, %s_33            // 1 - S
%v_40 = arith.mulf %v_39, %h_31             // (1-S)·H   ← 硬重置到 0
```
> 与 OTTT 的软重置（`V=H-S·v_th`）形成对照：MS-ResNet 是硬重置 `V=(1-S)·H`。

### L3：`call()` 真实启动序列（`artifacts/output_code.excerpt.py`）
真实的 LIF 启动行（注意 τ=4/3 是运行期 fp64 实参，不是字面量；profiler 归类时它是 triton kernel）：
```python
_multistep_lif_forward_kernel_0.run(buf12, buf13, buf10, buf11, buf11,
                                    1.3333333333333333, 0.5, 0.0, stream=raw_stream0)  # tau=4/3, vth=0.5, vreset=0
```
整条序列：`T=6 复制输入(_generalized_scatter) → triton 卷积 → TDBN(融合) → LIF(上行) → ... → aten._adaptive_avg_pool3d → 头部 triton_mm`。在完整 `output_code.py`（11289 行）里 `extern_kernels.{mm,bmm,convolution}` 出现 **0** 次。

> Triton 占比 98.7% / cublas 0 / cudnn 0 / other 1.3% 来自单 batch 的 **kernel 分类 profiler**（`run_msresnet.py --profile`），不是从上面这段节选直接得出；节选只用于展示算子来源。唯一非-Triton 计算 kernel 是 ATEN 的 `_adaptive_avg_pool3d`（inductor 无自适应 3D 池化 Triton 模板）。

## 结论
MS-ResNet-104 的 membrane-shortcut + TDBN 结构完整跑通 spikingjelly-Triton 重写：神经元与原 `mem_update` **逐位一致**，整网 **98.7% Triton、0 cublas/0 cudnn**，仅 `_adaptive_avg_pool3d`（1.3%）回退 ATEN 原生 kernel。精度 74.2%（论文 76.02% 全集）。
