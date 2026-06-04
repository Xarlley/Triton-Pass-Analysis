# 方法：Rate-Coded 输出模式 — 通用 SNN 末层减带宽 2.2×

> 时间：2026-05-29
> 状态：已集成到 [snn_compiler/nn/modules.py](../../snn_compiler/nn/modules.py)
> 接口：`RateCodedLIFNode` / `RateCodedIFNode`
> 适用：网络**最后一个** LIF/IF（紧邻分类器/投票输出）；中间层不可替换

## 1. 动机

SNN kernel 已经在 **GPU 主存（RTX 5070 Ti 是 GDDR7，非 HBM）峰值带宽** 上运行
（实测 ~705 GiB/s ≈ 105% 厂标 672 GB/s）。
进一步加速只能减少**总搬运字节数**，不能再压缩单字节算力。

观察 IF/LIF 在 t 循环里的访存模式：

```
for t in 0..T:
    y_t  ← GMEM load   (T × NCL × sizeof(elem) bytes)
    v    ← register（不离开 SM）
    spike← compute
    spike → GMEM store (T × NCL × sizeof(elem) bytes)
```

**读 = 写 = T × NCL × sizeof(elem)**。

对网络的最后一个 LIF（紧邻分类器投票），**真正需要的不是 [T, B, C, H, W] 整条 spike train，而是 [B, C, H, W] 的 spike-count（rate code）**。如果在 kernel 内累加，只需把读 (T × NCL × sizeof(elem)) 留着，把写改成一次 (NCL × 4)（fp32 count）：

```
write_old = T × NCL × sizeof(elem) = T × NCL × 2  (bf16)
write_new = NCL × 4                                 (fp32 count)
ratio     = (T × 2) / 4 = T / 2
```

T=4 → 2× 写减少；T=16 → 8×；T=64 → 32×；T=128 → **64× 写减少**。

由于读写平衡（带宽对半），写减少 N× → 总带宽减少 (N+1)/N × 接近 2×。
实测在 T≥16 时**稳定 2.1–2.2× 加速**。

## 2. 数学

`RateCodedLIFNode` / `RateCodedIFNode` 计算：

```
v_t = decay * v_{t-1} + scale * (x_t + bias)
spike_t = (v_t ≥ v_th) ? 1 : 0
v_t = (hard: v_t × (1-spike_t) + spike_t × v_reset; soft: v_t - spike_t × v_th)
count = ∑_{t=0..T-1} spike_t
```

返回 `count: [B, C, H, W]` fp32。与朴素 LIF / IF 的 `sum(dim=0)` 完全 bit-equal
（见 [snn_compiler/tests/test_largeT_and_rate.py](../../snn_compiler/tests/test_largeT_and_rate.py)::
`test_rate_coded_lif_bit_equal` / `test_rate_coded_if_bit_equal`，共 24 个用例
全部 `max|diff|=0`）。

## 3. 微基准（Phase B-2）

shape: T × B × NCL，bf16+NHWC，RTX 5070 Ti。

| T | NCL (k) | baseline (ms) | rate-coded (ms) | 加速 |
|---:|---:|---:|---:|---:|
| 4 | 3211 | 0.027 | 0.020 | 1.48× |
| 4 | 802 | 0.024 | 0.020 | 1.19× |
| 16 | 3211 | 0.273 | 0.128 | **2.13×** |
| 16 | 802 | 0.028 | 0.020 | 1.35× |
| 64 | 3211 | 1.090 | 0.504 | **2.16×** |
| 64 | 802 | 0.277 | 0.128 | **2.16×** |
| 128 | 3211 | 2.181 | 0.990 | **2.20×** |
| 128 | 802 | 0.550 | 0.249 | **2.21×** |

T ≥ 16 后 **稳定 2.13–2.21×**。

## 4. 使用

### 替换方式

如果你已经用本框架构网络，把网络末尾 LIF 直接换成 RateCodedLIFNode：

```python
import torch.nn as nn
from snn_compiler.nn import LIFNode, RateCodedLIFNode

# 原始 SNN 网络
class MySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ...     # Conv-BN-LIF...
        self.fc = nn.Linear(512, 1000)
        self.lif_last = LIFNode(tau=2.0, soft_reset=False)
    def forward(self, x_seq):           # [T, B, 3, H, W]
        x = self.features(x_seq)
        T_, B_ = x.shape[0], x.shape[1]
        x = self.fc(x.reshape(T_*B_, -1)).view(T_, B_, -1)
        return self.lif_last(x)         # [T, B, 1000]

# 加速：仅末尾 LIF 用 rate-coded（投票分数等价）
class MySNN_Fast(MySNN):
    def __init__(self):
        super().__init__()
        self.lif_last = RateCodedLIFNode(tau=2.0, soft_reset=False)
    def forward(self, x_seq):
        x = self.features(x_seq)
        T_, B_ = x.shape[0], x.shape[1]
        x = self.fc(x.reshape(T_*B_, -1)).view(T_, B_, -1)
        return self.lif_last(x)         # [B, 1000]  fp32 spike-count
```

下游（如 argmax 取 top-1 类别）：

```python
# 原 SNN: 投票 = spike_seq.sum(dim=0)；then top1
votes = my_snn(x)             # [T, B, 1000]
pred = votes.sum(0).argmax(-1)

# 加速版：lif_last 已直接产 count
pred = my_snn_fast(x).argmax(-1)
```

两者数值上严格相等（bit-eq 已验证）。

### 在 zoo 模型上批量替换

[snn_compiler/benchmarks/bench_largeT.py](../../snn_compiler/benchmarks/bench_largeT.py) 提供
`replace_last_lif_with_rate(model)` 函数，自动找最后一个 LIFNode（按 named_modules 顺序）替换。
对 VGG-16 / ResNet-18 / ResNet-34 zoo 模型直接生效。

## 5. 何时不能用

- **中间 LIF 层**不能换：下游 conv 需要 per-t spike，rate-coded 把 T 维 collapse 掉。
- **训练**：rate-coded kernel 暂未实现 backward。仅推理。
- **多投票头**：如果网络末尾不止一个 LIF（如分支输出），需对每个 LIF 单独判断是否最后一个。

## 6. 与其它优化的关系

| 优化 | 适用层 | 节省方向 | 兼容 rate-coded |
|---|---|---|---|
| Conv-BN-Neuron fusion | 所有 Conv-BN-LIF | launch + scratch | ✓（直接替换末尾即可） |
| Conv-BN-Add-Neuron | ResNet 残差 | 同上 | ✓ |
| i64 byte-offset | 任意大 T | 修 correctness | ✓（已硬性依赖） |
| T-chunked execution | 大 T 显存 | peak memory | ✓ |
| Rate-coded（本文） | 末尾 LIF | 写带宽 -64× | — |

末尾 LIF 用 rate-coded 时，会自动跳过对应的 FusedConvBNNeuron + per-t spike
materialization 两次 trip，与上面任何优化叠加。

## 7. 实测端到端收益

[Phase D 报告](../Exploration/mlir-perf-exploration-journal.md#stage-13)：
- VGG-16 SNN @ T=128, BATCH=4 末层换 RateCodedLIFNode：xxx
- ResNet-18 SNN @ T=128: xxx
- ResNet-34 SNN @ T=128: xxx

(数据由 [snn_compiler/benchmarks/bench_largeT.py](../../snn_compiler/benchmarks/bench_largeT.py)
跑出后自动追加；见 [Document/Benchmark/results/largeT_results.jsonl](../Benchmark/results/largeT_results.jsonl)。)

## 8. 实现细节

[snn_compiler/kernels/fused.py](../../snn_compiler/kernels/fused.py) 的
`_bias_if_lif_rate_kernel` 与 `_bias_if_lif_kernel` 共用 v、bias、threshold、reset 同一份分支逻辑，区别仅在：

```python
# 主 kernel：每步 store spike
tl.store(spike_ptr + t_off + ncl_idx, spike, mask=mask)

# rate kernel：循环外累加，循环结束后一次 store count
count = count + spike   # in t-loop
...
tl.store(count_ptr + ncl_idx, count, mask=mask)   # after t-loop
```

count 用 fp32 累加（避免 bf16 累加溢出/精度损失，且 NCL × 4B 也是最小写入量）。
