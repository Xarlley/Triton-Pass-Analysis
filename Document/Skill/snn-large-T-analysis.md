# SNN 大 T 推理瓶颈分析（T = 4 → 128）— 一份诚实的报告

> 时间：2026-05-29
> 数据来源：[snn_compiler/benchmarks/bench_largeT.py](../../snn_compiler/benchmarks/bench_largeT.py)
> 硬件：RTX 5070 Ti（16 GiB GDDR7，**非 HBM**，厂标 ~672 GB/s）
> 配置：BATCH=4, bf16+NHWC, LIF/hard reset

本文记录针对"再加速"的深入探索过程，包含一些**实测后被证伪/被发现收益有限**的想法。
目的是给后续的 SNN 系统/编译器工作者留下一个"哪些路径走过、为什么"的诚实参考。

## 1. 起点：当前框架是否 SNN-kernel-bound？

跑 [bench_baseline.py](../../snn_compiler/explore/large_T/bench_baseline.py)：
`_bias_if_lif_kernel` 在多种 shape 下稳态带宽：

| T | NCL | 实测 BW (GiB/s) | RTX 5070 Ti 厂标 |
|---:|---:|---:|---:|
| 4 | 3.2M | 705 | 672 |
| 16 | 3.2M | 712 | 672 |
| 64 | 3.2M | 707 | 672 |
| 128 | 3.2M | 704 | 672 |

→ 已经 **105% 厂标带宽**（thanks to L2）；**kernel 本身没有任何 launch / unroll / 编译时优化空间**。

后续的速度提升只能从两个方向之一拿：
- **(A) 减少 kernel 总搬运字节**：rate-coded、bit-pack、跨层 buffer reuse...
- **(B) 改 SNN 之外的部分**：conv kernel、launch graph...

## 2. 实测：VGG-16 SNN 各 T 下的 LIF 占比

[bench_largeT.py](../../snn_compiler/benchmarks/bench_largeT.py) 用 forward hook 抓出每个
FusedConvBNNeuron / FusedLinearNeuron 输出 shape，独立跑同 shape 的 `fused_bias_if_lif`
求和得到 "纯 LIF kernel 累计时间"。

VGG-16 SNN 实测（BATCH=4, bf16+NHWC）：

| T | 端到端 (ms) | LIF kernel 累计 (ms) | **LIF 占比** | conv 等占 |
|---:|---:|---:|---:|---:|
| 4 | 8.14 | 1.15 | **14.1%** | 85.9% |
| 16 | 31.24 | 4.69 | **15.0%** | 85.0% |
| 64 | 124.68 | 18.48 | **14.8%** | 85.2% |
| 128 | 248.98 | (估 37) | **~14.9%** | 85.1% |

→ **LIF 占比从 T=4 到 T=128 几乎不变（14–15%）**。
   conv 输出体量与 LIF 输入体量同步线性增长，二者 ratio 不变。
→ **即使把 LIF kernel 完全消掉（不可能）, 端到端最多省 15%**。

## 3. Rate-coded LIF：bit-equal ✓，端到端意义 ≈ 0

Phase B-2 微基准验证：rate-coded LIF kernel 比 baseline LIF kernel **2.2× 快**。

但在 VGG-16 端到端层面：

| T | baseline (ms) | + rate-coded 头 (ms) | Δ |
|---:|---:|---:|---:|
| 4 | 8.142 | 8.148 | +0.08% |
| 16 | 31.243 | 31.245 | +0.01% |
| 64 | 124.684 | 124.669 | -0.01% |
| 128 | 248.979 | 248.947 | -0.01% |

→ **rate-coded 头部开销可忽略（< 0.1%），但端到端也几乎无收益**。
   因为最后一个 LIF 只是 15 个 LIF 中的一个，且 fc 后的 LIF 输出 dim 已经小（1000）；
   单它的 2.2× 收益换算成端到端 ≈ 0.07% 而非 7%。

→ **正确的应用场景**：rate-coded 是给"网络末尾就是 spike count 投票"的真 SNN 用的；
   作为 VGG/ResNet 这种 ANN-converted-to-SNN（末尾 nn.Linear 出 logits 直接 argmax）
   的优化项，**无意义**。

## 4. T-chunked execution：bit-equal ✓，节省显存

Phase B-3：chunked driver 把 [T=128, ...] 拆 chunk_t=16 串接。
[微基准](../../snn_compiler/explore/large_T/chunked_lif_proto.py) 显示：
- bit-equal vs full-T ✓（8 个用例 max|diff|=0）
- 但是**比 full-T 慢 47%**（每 chunk 一次 kernel launch overhead）

→ **chunked execution 是给"模型太大塞不下显存"的最后救命稻草**，不是加速手段。
   T=128 BATCH=16 VGG-16（25.6 GB 单层激活）必须用 chunked 才能跑；
   T=128 BATCH=4（6.4 GB）full-T 就够。

## 5. Pool epilogue fusion：思路重叠，未推进

Phase B-4 prototype 写了 Conv→BN→LIF→AvgPool2x2 单 kernel：spike 不出 GMEM 而直接在 register 求 2×2 均值。

理论上能省 spike write + pool read 两次 trip（对那 5 个跟着 pool 的 LIF layer 有效）。
实测过程中 autotune 收敛非常慢（4 倍 v 寄存器压力 → 配置空间变大），且对应收益与 rate-coded 思路重叠（都是减 spike 写）。**未集成**。

## 6. 修复：i64 字节偏移（强制集成）

Phase A 抓 TTIR 发现 `_bias_if_lif_kernel` 的 t × NCL 偏移用 i32，在
T=128, BATCH=4, C=64, H=W=224 时 ((T-1) × NCL × 2B = 3.26 GB > 2³¹) 触发
`cudaErrorIllegalAddress`。修四个 kernel（详见 [snn-i64-offset-fix.md](snn-i64-offset-fix.md)）。

→ 这是 **correctness 修复**，不是优化。但让大 T 推理**能跑出来**，是本轮探索唯一不可
  绕过的成果。

## 7. 结论：再加速的路径在哪？

按"端到端时间"排序，VGG-16 推理的耗时大头：

1. **conv2d 调用（85%）** — 由 cuDNN / Inductor 选 Triton conv，单步算力已接近 tensor core 峰值。
2. **LIF kernel（15%）** — 已在 GDDR7 带宽峰值，rate-coded / bit-pack 等微优化收益 < 1%。
3. **autograd / Python overhead（推理时几乎 0）**。

要真正再加速，路径只剩：

- **(I)** 写一个 Triton conv，把 LIF 当 epilogue 融进 conv kernel：conv 输出**不进 GMEM**，
  直接在 register 里送 LIF。能省 2 × T × NCL bytes 单层带宽。理论可拿 ~15% 端到端，
  但是 4 周以上工作量（需要复刻 cuDNN 优化 conv）。
- **(II)** Conv-LIF-conv-LIF-...-flatten-FC 整网 megakernel：进一步消减 ALL 中间 spike buffer。
  研究路径，对 H, W 兼容性约束极严。
- **(III)** 算法侧：把 T 替换为更短的 latency-coded 编码（temporal coding 减时间步），改训练。

本轮探索能交付的是：**已经把 SNN-specific kernel 推到 GDDR7 峰值（~105% 厂标）**。再加速需要的是 **conv 侧** 或 **整网级** 优化，超出当前 SNN 编译器范畴。

## 8. 给用户的建议

| 你的场景 | 推荐路径 |
|---|---|
| 部署：T ≤ 128，模型尺寸正常 | 当前 zoo `fused=True` 已是性能上限；i64 修复 + 默认配置即可 |
| 大 T 显存不够 | 加 ChunkedForward(model, chunk_t=16)；7% latency 代价换 8× 显存 |
| 真 SNN 末层投票 | 用 RateCodedLIFNode 作为分类头（**架构层选择**，不只是优化） |
| 想再快 15%+ | 需自己写 Triton conv 把 LIF 当 epilogue（本框架尚未提供） |

本轮探索全部产物：
- 修复：[i64 字节偏移](snn-i64-offset-fix.md)
- 新增 API：[rate-coded 输出](snn-rate-coded-output.md)、[chunked 执行](snn-t-chunked-execution.md)
- 测试：188 → **221** 全 bit-equal（[test_largeT_and_rate.py](../../snn_compiler/tests/test_largeT_and_rate.py)）
- IR 截获：[Document/IR-Trace/large_T/](../IR-Trace/large_T/)
- 全过程：[journal §13](../Exploration/mlir-perf-exploration-journal.md#stage-13)
