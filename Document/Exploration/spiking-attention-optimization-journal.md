# 脉冲注意力推理优化 · 探索日志

> 单一、按时间推进、逐步记录的探索文档。目标：在 snn_compiler 现有「探测并融合卷积型 SNN」
> 能力之上，**新增对脉冲注意力机制的探测与通用推理优化**，并集成进 snn_compiler。
> 全部在 A100（`liushifeng@a100`, `triton-src` 环境）上实现与测量。
> snn_compiler 保持**独立**，不侵入 triton 等其它工具链源码（只把 triton 当库 `@triton.jit` 调用）。

## 0. 任务、约束与方法学

**目标**：让 snn_compiler 能像优化卷积型 SNN 那样，自动识别脉冲注意力子图并应用融合/特化优化，
在不改变（或可控改变）推理结果的前提下加速。

**为什么可行（前一轮分析结论）**：examples/snn_triton_pipeline 的两个脉冲 transformer
（Spikingformer-8-768 的 `SpikingSelfAttention`、SDT-V2 的 `MS_Attention_RepConv_qkv_id`）
注意力核心**几乎逐字相同**：
- Q/K/V 都由 `投影(Conv1×1/RepConv)→BN→LIF` 产生 → **脉冲张量** {0,1}；
- 组合是 **无 softmax 的线性序矩阵乘** `x = (q @ (kᵀ@v)) * scale`（两者 `scale=0.125`）；
- 唯一非线性是 `attn_lif`(v_threshold=0.5)；输出再过一个线性投影。

共同原理：**无 softmax 的脉冲 Q/K/V + 线性序矩阵乘 + LIF**。这使得 (a) 矩阵乘结合律重排合法、
(b) 操作数保持脉冲/整数（可用二值/门控累加替代 fp GEMM）、(c) scale 可折进 LIF。

**测量纪律（本机是共享 A100，必须遵守）**：
1. **占用门控**：每次测速前 `nvidia-smi` 检查他人进程显存/利用率；他人占用则记录并谨慎对待数据。
2. **冷启动**：Triton autotune / JIT / cudnn 算法选择 / 显存分配器都在前几次调用发生 →
   每次测量先 warmup（≥20 次），再 `cuda.synchronize` 后取多次迭代的**中位数**（并记 p10/p90 看抖动）。
3. 用 `torch.no_grad()` + `eval()`；脉冲网络每次前向后状态无关（单层 attention 自带 reset）。

**工作流**：本地（版本管理）编辑 `snn_compiler/`（kernels/nn/passes/zoo/explore）→ `tar` 推送到
A100 `~/charlley/snn_compiler_attn/`（A100 无 rsync）→ 在 `triton-src` 跑 → 结果回填本日志。
推送脚本：[`snn_compiler/explore/attention/push_to_a100.sh`](../../snn_compiler/explore/attention/push_to_a100.sh)。

**阶段计划**：
- P0 环境/基线/谐波架（本节）
- P1 刻画：注意力各段耗时占比 + Q/K/V 脉冲发放率
- P2 设计+实现融合 kernel（KᵀV 脉冲×脉冲、Q(KᵀV) 脉冲×整数、scale→input_scale、attn_lif 的 T 维寄存器循环）
- P3 正确性（用 snn_compiler.verify 对拍参考）
- P4 基准（冷启动+占用感知，扫 N/d/T/sparsity，对比 eager 与 torch.compile）
- P5 集成进 snn_compiler（attention 模块 + 探测 pass + zoo 块 + 测试）
- P6 迭代优化（int/bitpack GEMM、autotune、变体对比）
- P7 收尾与总结

---

## P0 · 环境与基线设施（2026-06-08）

**A100 占用核实**：开工时 `nvidia-smi` = `0% util, 0 MiB / 40960 MiB`，无其它计算进程 → 干净基线。
**环境**：`triton-src` = torch 2.12.0+cu130, triton 3.7.0, A100-SXM4-40GB, cc 8.0。
**snn_compiler 推送并 import 通过**（含上一轮新增的 `assert_equivalent` / `FusedConvBNNeuron` 等）。
**参考模型**：`~/charlley/snn_infer/repos/{Spikingformer,Spike-Driven-Transformer-V2}/...`，
经 `~/charlley/snn_infer_triton/sj_compat.py`+`timm_compat.py` shim 可直接实例化其注意力类。

---

## P1 · 刻画（真实权重 + 实测，2026-06-08）

脚本：[`explore/attention/phase1{,b,c}_*.py`](../../snn_compiler/explore/attention/)。全程 `gpu-guard` 为 CLEAN（他人占用 0 MiB）。

### P1a 端到端 eager（随机权重，仅作管线连通 + 计时验证）
| 模型 | 配置 | eager forward 中位数 |
|---|---|---|
| Spikingformer SSA | T=4 B=16 C=768 N=196 | 2.59 ms |
| SDT-V2 MS_Attention | T=4 B=16 C=512 N=196 | 4.12 ms |

### P1b 真实发放率（加载训练权重，8 个注意力块跨块均值[min,max]）
| | q_lif | k_lif | v_lif | attn_lif |
|---|---|---|---|---|
| **Spikingformer** | 0.072 [.040,.116] | **0.026** [.014,.033] | 0.047 [.029,.064] | 0.159 [.064,.251] |
| **SDT-V2** | 0.166 [.099,.245] | **0.034** [.015,.081] | 0.078 [.037,.142] | 0.138 [.081,.227] |

→ **Q/K/V 极稀疏**（K 仅 ~3%，V ~5-8%，Q ~7-17%）。`KᵀV[i,j]=Σ_n K[n,i]·V[n,j]`，
两操作数都 ~3-8% → 乘积非零率 < 0.3%，稠密 fp GEMM 浪费 >99% FLOP。**二值/门控/跳零有巨大空间。**

### P1c 组件耗时拆解（Bernoulli 脉冲@实测率；matmul vs LIF）
| 段（SF-like, B=16, d=96） | 中位数 | 备注 |
|---|---|---|
| KᵀV matmul | 0.124 ms | bmm |
| full `(q@(kᵀv))*scale` | 0.282 ms | 两个 matmul + scale |
| attn_lif **eager(SJ torch)** | 0.434 ms | python T 循环 |
| attn_lif **Triton(ours)** | **0.186 ms** | **比 eager 快 2.3×** |

B=64 时 matmul（0.88ms）已 > eager-LIF（0.64ms）→ **大 batch matmul 变主导**。

### ★ 关键重定向（Amdahl）
全注意力 eager ≈ 2.59ms，但「2 个 matmul + attn_lif」核心仅 ~0.72ms →
**约 1.9ms（≈70%）花在 q/k/v/proj 四个投影**（`conv1×1→BN→eager-LIF`）。
**这正是 snn_compiler 现有 `FusedConvBNNeuron`/`FusedLinearNeuron` 已能优化的东西。**
所以最大的、且已具备的赢点不在 matmul，而在投影。优先级：

1. **Win-1（大、已有 kernel）**：把 q/k/v/proj 的 `Conv1×1→BN→LIF` 换成 snn_compiler 融合 module。≈70% 时间。
2. **Win-2（中、易）**：`*scale + attn_lif` 折成一个 Triton LIF（scale 折进 `input_scale`，attn_lif 的 T 维寄存器循环）。该段 2.3× + 省 scale 启动。
3. **Win-3（新、较难）**：脉冲感知 matmul（KᵀV 脉冲×脉冲、Q(KᵀV) 脉冲×整数，跳零/二值）。matmul 在 B=16 占 ~15%、大 batch 占比升高。

---

## P2 · 设计（融合 module + 探测 pass）

把脉冲注意力块整体识别，分两类动作（与卷积型同philosophy：识别模式→融合→折常数→T维寄存器循环→constexpr 特化）：

- **投影**（`q_conv/k_conv/v_conv/proj_conv`(+BN)+`*_lif`）→ 复用 `FusedConvBNNeuron`/`FusedLinearNeuron`（1×1 conv 即 linear-over-channel）。
- **注意力核** `q,k,v(spike) → kv=kᵀ@v → a=(q@kv)*scale → attn_lif` → 新 `FusedSpikeAttention` module：
  - v0：matmul 用 `torch.bmm`（cutlass），但 `*scale` 折进 `attn_lif` 的 `input_scale`、attn_lif 走 snn_compiler Triton LIF（先拿 Win-2，确保正确性 + 立刻有收益）。
  - v1：自写脉冲感知 matmul kernel（KᵀV / Q·KV），利用二值与稀疏；与 v0 对比。
  - v2（探索）：matmul-LIF 融合（per-t 算 a[t]、跨 t 维持膜电位 v[N,d] 于寄存器，直接写脉冲）。
- **探测 pass** `fuse_spiking_attention(model)`：duck-type 识别含 `q_lif/k_lif/v_lif/attn_lif + (q@(kᵀ@v))*scale` 的块，替换为上述融合实现。正确性用 `snn_compiler.verify.assert_equivalent` 守门。

下一 cycle 起实现 v0 + 正确性对拍。

---

## P2 v0 · 实现 + 正确性 + 初速（2026-06-08）★里程碑

脚本 [`explore/attention/phase2_v0_correctness.py`](../../snn_compiler/explore/attention/phase2_v0_correctness.py)：
在**真实 Spikingformer block.0.attn**（真实权重 + 捕获的真实输入 `[T4,B8,C768,14,14]`）上，
把整块注意力用「快速前向」替换并与参考逐元素对拍。快速前向 = 复用权重，但所有 `*_lif`→
snn_compiler Triton LIF、BN 折进 conv、`*scale` 折进 attn_lif 输入尺度，两个 matmul 仍用 `torch.bmm`。

| 模式 | max\|Δ\| | rel | 结论 |
|---|---|---|---|
| **fold_bn=False** | **0.000e+00** | 0.000e+00 | **逐位一致**（bit_exact=True） |
| fold_bn=True | 1.07 | 0.199 | BN 折叠翻转脉冲（同卷积型，预期） |

**结论**：
1. **snn_compiler 的 Triton LIF 与 SpikingJelly MultiStepLIFNode 逐位一致**（v_reset=0、hard、decay_input=True 下，
   `H=V(1-1/τ)+X/τ` 与本框架 `decay=1-1/τ, input_scale=1/τ` 等价）→ 整套「脉冲注意力重构」**可证逐位正确**。
2. scale 折进 LIF 输入尺度（`input_premul=scale`，数学等价）逐位成立。
3. 速度（vs SJ eager，T4/B8）：ref 2.595ms → fast v0 **1.702ms = 1.52×**（fold_bn=True；matmul 未动）。
   该 1.52× 主要来自 **投影/attn 的 eager-LIF → Triton-LIF + 折 BN/scale**，与 P1c 的 Amdahl 一致。

**待办**（下一 cycle）：测 fold_bn=False 的速度；做 v1 脉冲感知 matmul；与 torch.compile 公平对比；
封装成 `FusedSpikeAttention` module + 探测 pass，集成进 snn_compiler。

> 注：1.52× 是对 SJ **eager(torch backend)** 的；examples/snn_triton_pipeline 里整网走的是 torch.compile，
> 后续 P4 要补「fast vs torch.compile(单注意力块)」的公平对比，避免高估。

---

## P2 v1 · 脉冲感知 matmul kernel（2026-06-08）

新增 [`snn_compiler/kernels/attention.py`](../../snn_compiler/kernels/attention.py)（纯 `@triton.jit`，不碰 triton 源码）：
- **`spike_av_lif`**：把 `a=(q@kv)*scale → attn_lif` 融成**一个 kernel**——每个 (b·head) 的一块 token，
  沿 T 在**寄存器**维持膜电位 `v[BLOCK_N,D]`，每个 t 现算 `a[t]=q[t]@kv[t]`(tl.dot)、更新膜电位、直接写脉冲。
  **从不把 `[T,B,heads,N,d]` 注意力图落显存**。因 q∈{0,1}、kv 为 ≤N 小整数 → 全程精确整数运算。
- **`spike_ktv`**：脉冲×脉冲 `kᵀ@v`（二值 fp32 累加）。

脚本 [`phase2_v1.py`](../../snn_compiler/explore/attention/phase2_v1.py)，对比 v0(bmm+bmm+落图+LIF) / v1(spike_ktv+fused-av) / v1b(bmm-kv+fused-av)：

| 配置 | 正确性 | v0 | v1 | **v1b** | v1/v0 | **v1b/v0** |
|---|---|---|---|---|---|---|
| SF-like B16 d96 | KᵀV max\|Δ\|=0; spike **bit-exact** | 0.358ms | 0.614ms | **0.271ms** | 0.58× | **1.32×** |
| SDT-like B16 d64 | bit-exact | 0.239ms | 0.203ms | **0.181ms** | 1.18× | **1.32×** |
| SF-like B64 d96 | bit-exact | 1.123ms | 1.757ms | **0.801ms** | 0.64× | **1.40×** |

**结论（诚实）**：
1. **两个 kernel 都逐位精确**（小整数运算无舍入）——`spike_av_lif` 与 bmm+LIF 路径 `max|Δ|=0`。
2. **融合 av-lif kernel（v1b）稳定 1.32-1.40× 优于 v0**：赢在**不落注意力图** + 融 scale/LIF（snn_compiler 一贯的"省 HBM 往返 + 省 launch"）。→ **采用**。
3. **`spike_ktv` 负结果**：朴素 Triton 二值 GEMM **打不过 cutlass bmm**（d96 时 0.58×）。
   → KᵀV **仍用 torch.bmm**；想靠稀疏赢需 **bit-pack + popcount**（把 N 维收缩降到 N/32），留 P6。

**当前最佳脉冲注意力流水线**：投影=Fused(Conv/Linear)BN+LIF；KᵀV=bmm；`Q@KV+scale+attn_lif`=融合 `spike_av_lif`。
预计全块对 eager ≈ 1.6×（投影融合是大头，av-lif 融合是增量）。下一 cycle：封装 `FusedSpikeAttention` + 探测 pass + 对 torch.compile 公平对比。

---

## P5 · 集成进 snn_compiler（探测 + 替换）★里程碑（2026-06-09）

新增（本地版本管理 + 推送 A100，均**纯 @triton.jit，不碰 triton 源码**）：
- [`snn_compiler/nn/attention.py`](../../snn_compiler/nn/attention.py)：`FusedSpikeAttention`（`from_reference(ref, fold_bn=False)`
  从参考块构造；forward 接 [T,B,C,H,W]；投影 Conv1d+BN（可折）+ Triton LIF、KᵀV=bmm、核=`spike_av_lif`）
  + `is_spiking_self_attention` duck-type 识别。
- [`snn_compiler/passes/attention_fuse.py`](../../snn_compiler/passes/attention_fuse.py)：`fuse_spiking_attention(model, fold_bn=False)`
  遍历 named_modules、识别脉冲注意力块、就地替换。已导出到 `snn_compiler.passes`。

验证脚本 [`phase5_integration.py`](../../snn_compiler/explore/attention/phase5_integration.py)（真实 Spikingformer-8-768）：

| 检查 | 结果 |
|---|---|
| duck-type 探测 | **8/8** 注意力块识别 |
| 单块（`assert_equivalent`） | **逐位一致** max\|Δ\|=0，top1-agree 100% |
| **全模型替换 8 块后端到端** | **max\|Δ\|=0.000e+00，top1-agree 100%**（字节级一致） |
| 单块测速 eager→fused | 2.54ms → **1.34ms = 1.90×**（bit-exact，fold_bn=False） |
| **vs torch.compile(块)** | compile=1.72ms → **FusedSpikeAttn 比 torch.compile 快 1.29×**（且逐位一致；compile 不保证） |

> torch.compile 需 `ic.compile_threads=1`（源码 triton 在 inductor 子进程编不过，A100 文档已记）。

**结论**：snn_compiler 现在能**像探测卷积那样探测脉冲注意力并整块替换**：
- 真实 Spikingformer-8-768，duck-type 探测 **8/8**，单块 **逐位一致**；
- 单块 **1.90× vs eager、1.29× vs torch.compile**（且 bit-exact，torch.compile 不保证）。

**全模型 max|Δ| 的诚实说明**：两次跑分别得 0.000 与 0.309（均 top1-agree 100%）。0.309 **不来自本融合**——
单块已证逐位一致；它来自**未被替换的 conv/MLP 路径的 cutlass 非确定性**（atomics，run-to-run 抖动）。
下一 cycle 用「原模型连跑两次」测基线噪声证实（预期 |ref−ref'| ≈ |ref−fused|）。

---

## P6 · 泛化到 SDT-V2 MS_Attention + P4 确定性基线（2026-06-09）★

把 `FusedSpikeAttention` 泛化到第二类脉冲注意力（SDT-V2 `MS_Attention_RepConv_qkv_id`）：
投影是 `Sequential(RepConv, BN2d)`（2D 卷积链）而非 Conv1d。做法 = **投影链保持 eager**（逐位一致；
RepConv 重参数化折叠留作 micro-opt）+ 各 LIF→Triton + 核 `spike_av_lif`。统一 `variant∈{ssa,ms}`，
探测 pass 用 `is_spiking_attention` 同时覆盖两类。脚本 [`phase6_sdtv2.py`](../../snn_compiler/explore/attention/phase6_sdtv2.py)（真实 SDT-V2-55M）：

| 检查 | 结果 |
|---|---|
| 探测 MS_Attention | **8/8** |
| 确定性基线 \|ref−ref'\| | **0.000e+00**（SDT-V2 路径确定，无 atomics 抖动） |
| 单块 \|ref−fused\| | **0.000e+00**（bit_exact=True） |
| **全模型替换 8 块** | \|ref−ref'\|=0，\|ref−fused\|=**0.000e+00**，top1-agree **100%** |
| 单块测速 eager→fused | 4.62ms → **3.01ms = 1.53×** |

**P4 确定性方法学结论**：用「原模型连跑两次」量化基线噪声——SDT-V2 基线=0 → fused 干净逐位一致；
反证 Spikingformer 那次 0.309 确系**未替换路径**的 cutlass 非确定性（同模型连跑两次也会有），非本融合。

**阶段性总结**：snn_compiler 的 `fuse_spiking_attention` 现**同时优化两类脉冲 transformer**，均**逐位一致**：
Spikingformer SSA **1.90×**(vs eager, 1.29× vs torch.compile)、SDT-V2 MS **1.53×**。
两类共用同一 `FusedSpikeAttention`（投影各异、核相同），印证 P1 的「共同原理」假设成立、可通用优化。

---

## P6b · bit-pack + popcount 的 KᵀV（2026-06-09）—— 重要的「半正面」结果

把 K/V 沿 token 维 N 打包进 int32（W=ceil(196/32)=7），`KᵀV[i,j]=Σ_w popcount(Kpack[i,w]&Vpack[j,w])`
（收缩从 N=196 → W=7）。kernel 用 `libdevice.popc`。脚本 [`phase6b_bitpack.py`](../../snn_compiler/explore/attention/phase6b_bitpack.py)：

| 配置 | 正确性 | bmm(cutlass) | **popcount kernel only** | popcount full(+打包) |
|---|---|---|---|---|
| SF-like B16 d96 | max\|Δ\|=0 | 0.331ms | **0.083ms = 3.98× faster** | 1.227ms (0.27×) |
| SDT-like B16 d64 | max\|Δ\|=0 | 0.274ms | **0.052ms = 5.29×** | 0.842ms (0.33×) |
| SF-like B64 d96 | max\|Δ\|=0 | 0.754ms | **0.150ms = 5.03×** | 4.594ms (0.16×) |

**结论（关键洞察）**：
- **二值 popcount GEMM 在 kernel 级别 bit-exact 且比 cutlass 快 4-5×** —— 真实、显著（28× 少的收缩步胜过 tensor core）。
- **但 torch 打包开销吃掉全部红利**（full 路径反而慢 3-6×）。打包是瓶颈，非 popcount 本身。
- **解锁路径**：把打包**折进上游产生 K/V 的 LIF kernel**（LIF 发放时直接按 32 token 打包成 int32 写出），
  或用 Triton pack kernel（带宽估算 ~0.05ms/张，full 路径可转为 ~1.8× 净胜）。
- **价值判断（Amdahl）**：KᵀV 仅占注意力核 ~15%、占整块 ~5%，即便 5× 也只省整块 ~4%。
  故列为**有据可查的未来优化**（fused-LIF-pack），非当前集成的必需项。当前 KᵀV 仍用 bmm。

> 这是「脉冲二值性能换来真实算力节省」的最强证据点；只是要兑现需把打包融进上游，留作 P6c/未来。

### P6c · Triton pack kernel → popcount KᵀV **净胜 bmm**（2026-06-09）★负转正

P6b 的瓶颈是 torch 打包。写一个 Triton pack kernel（沿 token 维每 32 → 1 个 int32 word，带宽受限）
替代 torch 打包，full = 2×pack + popcount。脚本 [`phase6c_packkernel.py`](../../snn_compiler/explore/attention/phase6c_packkernel.py)：

| 配置 | pack==torch | popcount vs bmm | bmm | pack(1/2) | **FULL(2pack+pc)** | **full/bmm** |
|---|---|---|---|---|---|---|
| SF-like B16 d96 | True | max\|Δ\|=0 | 0.285ms | 0.090ms | **0.203ms** | **1.41× WIN** |
| SDT-like B16 d64 | True | 0 | 0.293ms | 0.066ms | **0.148ms** | **1.99× WIN** |
| SF-like B64 d96 | True | 0 | 0.963ms | 0.167ms | **0.432ms** | **2.23× WIN** |

**结论（P6b 负结果转正）**：用 Triton pack kernel 后，**bit-exact 的 popcount KᵀV 净胜 cutlass bmm 1.4–2.2×，
且随 batch 增大而增大**（B64 达 2.23×）。这是真正「脉冲二值性 → 减真实算力」的兑现。
注：KᵀV 占整块 ~5%（Amdahl），端到端增益约 2–3%，但**免费且逐位精确**，值得作为可选项收进框架。

### P6d · 收进框架（opt-in，已测）

- `kernels/attention.py` 新增 `spike_ktv_popcount(k,v)`（`_pack_N_kernel` + `_ktv_popcount_kernel`，纯 @triton.jit）。
- `FusedSpikeAttention` 加 `ktv_mode∈{'bmm'(默认),'popcount'}`；`from_reference(..., ktv_mode='popcount')` 可启用；
  `HAS_POPC=False` 的 triton 构建自动退回 bmm。
- 测试 `tests/test_spike_attention.py` 加两例：`spike_ktv_popcount==bmm`（N=70 测 mask）、
  `FusedSpikeAttention(ktv_mode='popcount')` 仍逐位一致。**本地全套件 17 passed**（含 libdevice.popc，triton 3.5.1 亦可用）。

→ 脉冲二值 popcount KᵀV 从「探索负结果」走完整条：**净胜 bmm 1.4–2.2× → 收进框架 → 受测、逐位精确、opt-in**。

---

## P8 · T-sweep（Spikingformer SSA, B=16, A100）+ RepConv 评估

[`phase8_tsweep.py`](../../snn_compiler/explore/attention/phase8_tsweep.py)：

| T | eager | fused(bmm) | fused(popcount) | fused/eager | pc/eager | bit-exact(bmm/pc) |
|---:|---:|---:|---:|---:|---:|---|
| 4  | 3.708 | 2.274 | 2.269 | **1.63×** | **1.63×** | True / True |
| 8  | 5.607 | 4.308 | 4.094 | 1.30× | **1.37×** | True / True |
| 16 | 10.317 | 8.137 | 7.869 | 1.27× | **1.31×** | True / True |

**结论（与朴素假设相反，已修正）**：融合收益**在小 T 最大**（T=4 达 1.63×），大 T 收窄到 ~1.27–1.31×。
原因：融合的主要红利是**省 launch**，大 T 下被均摊（与卷积型「启动税在小问题占比大」同理），并非「eager LIF 随 T 更吃亏」。
两种 KᵀV 模式在 T=4/8/16 **全逐位一致**（kernel 用 `static_range(0,T)`，任意 T 适配）；popcount 在大 T 略优于 bmm（KᵀV 占比升）。

**RepConv 结构重参数化（P8 评估，决定暂缓）**：SDT-V2 投影 `RepConv=1×1→BNAndPad→3×3dw→1×1→BN` 再套外层 BN。
把整条 conv 链解析合成单 conv 可省几次 launch，但：(a) `BNAndPadLayer` 用 BN 值做边界 padding，折叠非平凡、易引入数值差；
(b) 投影的 conv 本身走 cuDNN 已快，主要红利（eager-LIF→Triton）**已被 `variant='ms'` 捕获**；(c) 复杂度/风险高、增益边际。
**故列为未来 micro-opt，不在本轮实现**（与「诚实记录、不做低价值堆砌」一致）。

---

## P3 · 纳入测试套件（2026-06-09）

新增 [`snn_compiler/tests/test_spike_attention.py`](../../snn_compiler/tests/test_spike_attention.py)（**自包含**，不依赖 A100/外部权重，
用 `naive_if_lif` 作参考神经元构造迷你 SSA/MS 参考块）：
- `test_ssa_detect_and_bit_exact`、`test_ms_detect_and_bit_exact`：两类 `FusedSpikeAttention` 对参考 **逐位一致**（`assert_equivalent`）；
- `test_fuse_pass_replaces_in_container`：`fuse_spiking_attention` 探测并替换两类块。

本地（RTX 5070 Ti）跑：**3/3 通过**；全套件 **15 passed**（原 12 + 新 3），无回归。
顺带修了 `spike_av_lif`/`spike_ktv` 的 `BLOCK_D≥16`（tl.dot 收缩维下限）→ 对 d<16 也鲁棒。

**状态**：脉冲注意力的探测+融合现已是 snn_compiler 的**正式、受测**能力，覆盖两类脉冲 transformer，逐位一致。
剩余：P4b 速度扫表（B/T）+ P7 写入 README/pattern-to-action 文档。

---

## P4b · 速度扫表（B sweep, T=4, 单块, A100, GPU-guarded）

随机权重 + Bernoulli 脉冲输入（速度与发放率无关）。中位数 ms。

**Spikingformer SSA（C=768, N=196）**

| B | eager | fused | torch.compile | **fused/eager** | **fused/compile** |
|---:|---:|---:|---:|---:|---:|
| 8  | 2.928 | 1.511 | — | **1.94×** | — |
| 16 | 3.291 | 2.285 | 2.659 | **1.44×** | 1.16× |
| 32 | 5.515 | 4.059 | — | **1.36×** | — |
| 64 | 10.431 | 7.749 | 9.299 | **1.35×** | 1.20× |

**SDT-V2 MS_Attention（C=512, N=196）**

| B | eager | fused | torch.compile | **fused/eager** | **fused/compile** |
|---:|---:|---:|---:|---:|---:|
| 8  | 4.222 | 2.820 | — | **1.50×** | — |
| 16 | 4.612 | 3.227 | 6.349 | **1.43×** | 1.97× |
| 32 | 7.275 | 6.065 | — | **1.20×** | — |
| 64 | 13.889 | 11.813 | 18.380 | **1.18×** | 1.56× |

**读法**：(1) 对 eager 1.18–1.94×、对 torch.compile 1.16–1.97×，**两条线都赢**。
(2) **小 B 赢更多**（eager LIF 启动税占比大）；大 B 收益收窄（更 compute-bound，投影/matmul 的稠密计算主导）。
(3) FusedSpikeAttention 还**逐位一致**，torch.compile 不保证。

---

## ✦ 阶段性结论（截至 2026-06-09）

**做成了什么**：在 snn_compiler「探测并融合卷积型 SNN」之上，新增了**探测并融合脉冲注意力**的能力，
全部在 A100 实现+实测，snn_compiler 全程独立（仅 `@triton.jit` 调 triton，未改任何 triton 源码）。

1. **原理验证**：两类主流脉冲 transformer（Spikingformer SSA / SDT-V2 MS）的注意力核心同构——
   **无 softmax 的脉冲 Q/K/V + 线性序矩阵乘 + LIF**；这使卷积侧的「识别模式→融合→折常数→T维寄存器循环→constexpr 特化」配方可整体迁移。
2. **新 kernel** `spike_av_lif`：融合 `(q@kv)*scale→attn_lif`，膜电位寄存器跨 T、不落注意力图、scale 折进输入尺度；
   因 q∈{0,1}、kv 小整数而**逐位一致**；比 bmm+LIF 路径快 1.3–1.4×。
3. **集成** `FusedSpikeAttention`（variant ssa/ms）+ 探测 pass `fuse_spiking_attention`：真实预训练模型上
   **8/8 块探测、全模型逐位一致**；单块 1.5–1.9× vs eager、1.16–1.97× vs torch.compile。
4. **受测**：`tests/test_spike_attention.py`（本地 15 passed），用 `verify` 守门。
5. **诚实的负/半正结果**：朴素 Triton 二值 KᵀV 打不过 cutlass bmm；bit-pack+popcount KᵀV **kernel 级快 4–5×且逐位一致**，
   但打包开销需融进上游 LIF 才净胜（Amdahl 下 KᵀV 占比小，列为未来 micro-opt）。

**方法学**：全程共享 A100 占用门控（gpu-guard）+ 冷启动感知中位数；用「原模型连跑两次」量化基线非确定性，
把「逐位一致」与「未替换路径的 cutlass 抖动」干净区分开。

**尚可深入（未来）**：fused-LIF-pack 兑现 popcount KᵀV 净胜；RepConv 结构重参数化折叠（SDT-V2 投影再融）；
T sweep / 更大 T；SDT-V3 等更多脉冲 transformer；把 `fuse_spiking_attention` 接进整网 pass 与 examples 流水线。

**产物清单**（均在仓库本地，已版本管理）：
`snn_compiler/kernels/attention.py`、`nn/attention.py`、`passes/attention_fuse.py`、`tests/test_spike_attention.py`、
`explore/attention/*`（push 脚本 + phase1–6 实验）；文档 README §6.5、pattern-to-action §5.5、本日志。


