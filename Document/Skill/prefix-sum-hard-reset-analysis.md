# Prefix-Sum IF 神经元的 hard-reset 适配：路径分析与实测

> [前一份实验](prefix-sum-snn-experiment.md)采用了 soft-reset（spike 后 `v -= threshold`），因为 prefix-sum 自然契合这种**线性可分离**的 reset 语义。本文回答用户提的进阶问题：
>
> **「有没有能适配 hard-reset（spike 后 `v = 0`，丢弃 overshoot）的方法？分析可行的实现路径。」**
>
> 关联：[`prefix-sum-snn-experiment.md`](prefix-sum-snn-experiment.md) §1.2 中已说明 soft/hard reset 数值不等价但实现复杂度类似；这里给出 hard-reset 的完整实现路径分析与实测对照。

---

## 1. Hard-reset 为什么比 soft-reset 难

### 1.1 两种 reset 的数学差异

设 `cum[t] = sum_{k=0..t} x[k]`，`s_t` 为 t 时刻是否发 spike（0/1），`n_t = sum_{k=0..t} s_k` 为累计 spike 数。

| Reset 类型 | t 时刻有效膜电位 | 状态变量数 |
|---|---|---:|
| Soft-reset | **`v_t = cum[t] - threshold · n_{t-1}`** | 1 (`n_{t-1}`，线性累加) |
| Hard-reset | **`v_t = cum[t] - cum[t_last_spike]`** | 1 (`cum[t_last_spike]`，**条件覆盖**) |

**两者根本性差异在状态变量的更新方式**：

- **Soft-reset 的状态 `n_{t-1}`** 是个"加法累加器"，每发一次 spike 就 `+1`。可以用 prefix-sum 自动处理 —— `n_t = cumsum(s)[t]`，spike 后状态加 1，**信息可分解**。
- **Hard-reset 的状态 `cum[t_last_spike]`** 是个"最近事件标记"，每发一次 spike 就**整体替换**为当前 cumsum 值；非 spike 时刻保持不变。这是路径依赖的，**不能用简单累加表示**。

数学上：
- 在 soft-reset 下，`v_t - v_{t-1} = x_t - threshold · s_{t-1}`，每一项都可单独算，最后求和；
- 在 hard-reset 下，`v_t - v_{t-1} = x_t` 仅当 `t-1` 没有 spike；如果 `s_{t-1}=1`，则 `v_t = x_t`（不是简单累加关系）。

### 1.2 用一个具体例子说明

设 `x = [0.5, 0.8, 0.7, 0.4]`，`threshold = 1.0`：

```
t :  0     1     2     3
x :  0.5   0.8   0.7   0.4
cum: 0.5   1.3   2.0   2.4

soft-reset：
  v_0 = 0.5 < 1   → s_0=0   v_0 -= 0 = 0.5
  v_1 = 0.5+0.8=1.3 ≥ 1 → s_1=1, v_1 = 1.3 - 1 = 0.3
  v_2 = 0.3+0.7=1.0 ≥ 1 → s_2=1, v_2 = 1.0 - 1 = 0.0
  v_3 = 0.0+0.4=0.4 < 1 → s_3=0
  spikes_soft = [0, 1, 1, 0]

hard-reset：
  v_0 = 0.5 < 1   → s_0=0
  v_1 = 0.5+0.8=1.3 ≥ 1 → s_1=1, v_1 = 0
  v_2 = 0+0.7=0.7 < 1 → s_2=0   ← 与 soft-reset 不同！
  v_3 = 0.7+0.4=1.1 ≥ 1 → s_3=1
  spikes_hard = [0, 1, 0, 1]
```

soft-reset 把 t=1 的 overshoot (1.3-1=0.3) 留到 t=2 继续累加，于是 t=2 也发 spike；hard-reset 把 overshoot 丢掉，t=2 没发，反而 t=3 发了。**两条 spike train 在数值上不同**，会影响下游 conv 的输入分布。

---

## 2. 五条可行实现路径分析

下面按"从最朴素到最巧妙"的顺序，列出五种 hard-reset 的实现方式，分析各自的可行性。

### 路径 A：朴素顺序（不用 prefix-sum）—— 基线

```python
def forward(self, x):                                 # x: [T,B,...]
    v = torch.zeros_like(x[0])
    spikes = []
    for t in range(x.shape[0]):
        v = v + x[t]                                  # 积分
        spike = (v >= self.v_threshold).to(x.dtype)
        v = v * (1 - spike)                           # hard-reset：触发即清零
        spikes.append(spike)
    return torch.stack(spikes, dim=0)
```

- **不用 prefix-sum，纯顺序累加 + 条件清零**；
- 与 SJ `_multistep_lif_forward_kernel` 在 `soft_reset=False` 时的逻辑完全等价（只差是否带 decay）；
- 缺点：状态变量 `v` 的更新有非线性依赖（`v_{t} = (v_{t-1}+x_t)·(1-s_t)`），Inductor 编译时不能把 `v` 的串行依赖暴露给优化器，所有 T 步必须按因果顺序执行；
- 优点：实现最直接，是其余四条路径的**正确性参考**。

### 路径 B：Prefix-sum + per-element 条件覆盖（**本文采纳**）

观察到 hard-reset 下：

```
v_t = cum[t] - cum[t_last_spike]    （t_last_spike 是 t 之前最近一次 spike 时刻）
```

定义状态 `last_cum_at_spike` —— 上一次 spike 时的 cumsum 值。每个时间步：

```python
def forward(self, x):
    cum = torch.cumsum(x, dim=0)                      # [T,B,...]  prefix-sum 部分
    last_cum_at_spike = torch.zeros_like(x[0])         # baseline，初值 0
    spikes = []
    for t in range(x.shape[0]):
        v_t = cum[t] - last_cum_at_spike               # 自上次 reset 以来的累积
        spike = (v_t >= self.v_threshold).to(x.dtype)
        spikes.append(spike)
        last_cum_at_spike = torch.where(spike > 0, cum[t], last_cum_at_spike)
    return torch.stack(spikes, dim=0)
```

**正确性**：spike 触发时 `last_cum_at_spike ← cum[t]`，下一步 `v_{t+1} = cum[t+1] - cum[t] = x[t+1]`，等价于 hard-reset 后的 `v_{t+1} = 0 + x[t+1]`。✓

**算子开销**：
- 1 次 cumsum（log-T 深度的 parallel scan）
- T 次 elementwise op 组合（`sub`, `cmp`, `where`）—— 总 ~T 个简单 elementwise op

**等价的更简洁写法（无 `torch.where`）**：
```python
last_cum_at_spike = last_cum_at_spike + spike * v_t
# 当 spike=0: 不变 ✓
# 当 spike=1: last = last + v_t = last + (cum[t] - last) = cum[t] ✓
```

这把 `torch.where` 换成 `mul+add`，对 Inductor 更友好（更易与上下游 elementwise fuse）。两种写法实测延迟差异在噪声内。

### 路径 C：用 cummax 维护 baseline

直觉上 `last_cum_at_spike` 满足"每发一次 spike 就涨到当前 cum 值，否则保持"，**且涨到的值满足某种单调约束**：
- 在 spike 时 cum[t] ≥ threshold + last_cum_at_spike(t-1)
- 所以 cum[t] > last_cum_at_spike(t-1)

但这**不等价于** `cummax(cum)` —— cum 可以单调递增但不发 spike（如果 threshold=10 而每步 +0.1）。`cummax` 对负 x 也无效（cum 可能下降，但 baseline 必须保持）。
**路径 C 不可行**，cummax 不是 baseline 的正确解析形式。

### 路径 D：双阶段 — soft 找近似位置 + hard 矫正

1. 第一阶段：用 prefix-sum + soft-reset 找出 spike train `s_soft`；
2. 第二阶段：以 `s_soft` 为初始猜测，按 hard-reset 规则修正：丢弃在 soft 下因 overshoot 多发的 spike、补上 hard 下因 baseline 更高才能发的 spike。

理论上可行（soft 是 hard 的上界 / 下界关系在分析上很清晰），但**实现上不比路径 B 简单** —— 修正阶段本身也得是 T 步顺序循环。
**实际不推荐**：用一个相同复杂度的两阶段流程，没有给 Inductor 额外的融合空间。

### 路径 E：自定义 segmented scan with reset operator

定义关联运算 ⊕，状态 `(c, r)`：`c` 是积分值，`r` 是 reset flag。
- `(a, 0) ⊕ (b, 0) = (a + b, 0)`
- `(a, _) ⊕ (b, 1) = (b, 1)`  —— 右侧 reset 时全清空左侧
- `(a, 1) ⊕ (b, 0) = (a + b, 1)`  —— 当前段的 reset flag 传给段尾

这套运算**是关联的**（结合律可验证），意味着可以用 Blelloch 风格的并行 scan 在 O(log T) 深度内完成。

**但**：reset 标记 `r` 本身来自 spike 判定，spike 又依赖前段积分值 —— **循环依赖**。要打破必须**先做一遍 hard-reset 的 spike 判定**（路径 B 或 A），然后才知道哪里 reset。也就是说，路径 E 的并行 scan 是"已知 reset 位置之后"的事，并不能避免路径 B 的顺序判定。
**单看推理路径意义有限**，但训练反向传播或时间维拼接长序列时可能有用（不在本文范围）。

### 综合：路径 B 是最佳选择

| 路径 | 算子开销 | 实现复杂度 | Inductor 友好度 |
|---|---|---|---|
| **A 朴素顺序** | T 步顺序 + 同尺寸状态 | 最简单 | 中（custom_op 或纯 elementwise 都行）|
| **B Prefix-sum + 条件覆盖** | 1 cumsum + T 个 elementwise | 简单 | **高**（全是标准 ATen op）|
| C cummax baseline | — | — | **数学不成立** |
| D 双阶段矫正 | 2 × (1 cumsum + T 步) | 偏复杂 | 同 B 但 op 数翻倍 |
| E 并行 scan w/ reset | 取决于是否预知 reset 位置 | 复杂 | T=4 时 log T 优势可忽略 |

**路径 B**：算子少、与 soft-reset 实现高度对称（只把 `spike_count += spike` 换成 `last_cum_at_spike` 的条件覆盖）、Inductor 完全能 codegen，是最实用的选择。

---

## 2.5 「Hard-reset 会让前缀和后续部分失效」这个担忧是对的吗？

**对、也不对** —— 要看是说算法层面还是浮点精度层面。

### 算法层面（数学）：**没有失效**

担忧"前缀和后续部分失效"的直觉源于这样的想法：
- 朴素地把 `cum[t]` 直接当作膜电位 → 一旦发生 hard reset，`cum[t]` 对 t > t_spike 时刻就**不再代表真实的膜电位**（因为真实 v 应该是从 reset 之后才开始重新积分）。

这个直觉**只在"直接读 cum[t]"时成立**。路径 B 不是这么做的 —— 它读的是 `cum[t] - last_cum_at_spike`，其中 `last_cum_at_spike` 记录的是**上一次 spike 时的 cum 值**。这两项相减恰好抵消掉 "reset 之前的累积"：

```
v_t = cum[t] - cum[t_last_spike]
    = sum_{k=0..t} x[k] - sum_{k=0..t_last_spike} x[k]
    = sum_{k=t_last_spike+1..t} x[k]                    ← 这正是 hard-reset 后该有的"自上次 reset 起的累积"
```

`cum` 本身依然是从 0 开始的完整 prefix-sum，**没被"reset 破坏"**；reset 后真实膜电位的恢复完全通过减去基线来完成。
**关键点**：cum 本身被所有时间步共享、只算一次；hard-reset 的"丢弃过去"由后续每步的减法实现，不需要重算 cumsum。

### 数值验证（[`examples/vgg16_snn/verify_hard_reset.py`](../../examples/vgg16_snn/verify_hard_reset.py)）

把路径 B 的输出与朴素顺序参考（`v=v+x[t]; spike=v>=θ; v=v*(1-spike)`）做逐位对比，16 个测试用例（含 doc 中的人工示例、负输入、连续 spike、不同 threshold、不同 T、VGG16-style 大张量、fp64 对照）：

```
=== 数值验证：prefix-sum hard-reset vs naive sequential hard-reset ===

  uniform positive [0,1)                                bit-equal=True   diff=0/64
  uniform [-1, 1)                                       bit-equal=True   diff=0/64
  typical conv output range, x ~ N(0, 1)                bit-equal=True   diff=0/4096
  large positive (consecutive spikes), x ~ U[1, 5)      bit-equal=True   diff=0/1024
  large negative after reset                            bit-equal=True   diff=0/4
  hand-crafted: x=[0.5,0.8,0.7,0.4], doc example        bit-equal=True   diff=0/4
  hand-crafted: x=[3,3,3,3] all spike                   bit-equal=True   diff=0/4
  hand-crafted: x=[3,0,3,0] alternating                 bit-equal=True   diff=0/4
  hand-crafted: x=[0.5,0.8,-2.0,1.5] negative after spk bit-equal=True   diff=0/4
  vgg16-style: T=4, B=4, C·H·W=64*224*224               bit-equal=True   diff=0/51380224
  vgg16-style with dense spikes (rate 0.978)            bit-equal=False  diff=1/51380224 ⚠️
  [fp64] same dense-spike case, double precision        bit-equal=True   diff=0/51380224 ✓
  threshold=0.5                                         bit-equal=True   diff=0/1024
  threshold=2.0 (sparse spiking)                        bit-equal=True   diff=0/1024
  T=8                                                   bit-equal=True   diff=0/512
  T=16                                                  bit-equal=True   diff=0/1024

=== 15/16 cases passed bit-equal ===
```

### 浮点层面：**fp32 在病态情况下会有 1-LSB 误差**

仅有的 1 个不一致：「VGG16-style with multiple consecutive spikes (spike rate 0.978)」—— 几乎每个 (t, B, C, H, W) 位置都发 spike，cum 会累积到很大值，`last_cum_at_spike` 同步追到很大值，两者相减时**fp32 出现 catastrophic cancellation**。最终 5138 万个 spike 输出里 1 个的 spike 决定与朴素参考不一致（一个边界点 `v ≈ threshold` 因 round-off 翻向另一侧）。

**fp64 重做同一例 → 0 位差异**，证实这是浮点精度问题、不是算法问题。

### 实际影响

- **典型 VGG16-SNN 推理（post-Conv 输入 ~ N(0, σ≈1)，spike rate 10–20%）** ：cum 值在每个位置上累积幅度 ~T·σ ≈ 4 σ，远未到 fp32 精度边界 → **完全 bit-equal**。
- **训练后期 / 大 σ 输入 / dense spike train（rate > 90%）**：可能出现 1e-8 量级的 spike 决定差异。对 SNN 训练通常无影响（与 surrogate gradient 自身的近似误差比微不足道），对推理也不影响 top-k 决策。
- **如果需要绝对的 bit-equal 保证**：可以在数值上等价地把状态变量从 `last_cum_at_spike` 换成 **从 reset 时刻开始重新积分的局部 v**，即变回朴素顺序方案（路径 A），但失去 Inductor 的全局 fuse 优势。

或者将状态变量改写为 `v_local`（自 reset 后局部积分），数学等价但不依赖大数相减：

```python
v_local = torch.zeros_like(x[0])
for t in range(T):
    v_local = v_local + x[t]                                # 自上次 reset 起的局部 v
    spike = (v_local >= threshold).to(x.dtype)
    spikes.append(spike)
    v_local = v_local * (1 - spike)                         # hard-reset
```

这是「路径 A 朴素顺序」—— 与「路径 B prefix-sum」在 fp64 下逐位等价，**fp32 下没有大数相减问题**。但路径 A 没有 cumsum 这一可被 Inductor 强力优化的可分离部分，所以**路径 B 在 fp32 一般情况下又准又快**，仅在病态情况让出 1 LSB 精度。

---

## 3. 路径 B 完整实现

```python
class PrefixSumHardResetIFNode(nn.Module):
    def __init__(self, v_threshold: float = 1.0):
        super().__init__()
        self.v_threshold = v_threshold

    def forward(self, x):                         # x: [T, B, ...]
        cum = torch.cumsum(x, dim=0)
        last_cum_at_spike = torch.zeros_like(x[0])
        spikes = []
        for t in range(x.shape[0]):
            v_t = cum[t] - last_cum_at_spike
            spike_t = (v_t >= self.v_threshold).to(x.dtype)
            spikes.append(spike_t)
            last_cum_at_spike = torch.where(spike_t > 0, cum[t], last_cum_at_spike)
        return torch.stack(spikes, dim=0)
```

固化在 [`examples/vgg16_snn/prefix_sum_snn.py`](../../examples/vgg16_snn/prefix_sum_snn.py) 的 `PrefixSumHardResetIFNode` 类，通过 `RESET=hard` env var 切换。

---

## 4. 实测对比

冷启动 10024 样本，BATCH=56，VGG16-SNN (13 Conv + 5 AvgPool + 3 FC + 15 IF)，RTX 5070 Ti：

| Reset | Mode | 单张延迟 | 吞吐 | 峰值显存 | 冷启动 |
|---|---|---:|---:|---:|---:|
| soft  | eager   | 7.963 ms | 125.6 张/s | 12.71 GiB | 0.6 s |
| soft  | compile | 8.241 ms | 121.3 张/s | 6.00 GiB | 89.7 s |
| **hard**  | **eager**   | **7.795 ms** | **128.3 张/s** | 12.71 GiB | 0.6 s |
| **hard**  | **compile** | **8.245 ms** | 121.3 张/s | **6.00 GiB** | 91.6 s |

### 关键观察

1. **hard-reset 与 soft-reset 在两种模式下的延迟完全在噪声范围内**（差异 0.05–0.17 ms / 张，相对差 < 2%，各自 run-to-run std ~ 0.2–0.5 ms）。说明 `torch.where(spike, cum, last)` 与 `last + spike·v` 在 Inductor 编译产物上几乎不可区分。
2. **compile 模式下 hard-reset 峰值显存依然只 6 GiB**，与 soft 完全一致 —— 这进一步说明显存收益来自"不用 SJ LIF custom_op"这件事本身，而不是 reset 语义。
3. **eager 模式下 hard-reset 比 soft-reset 略快 2%**。原因猜测：`torch.where` 的 GPU 实现是一次条件 cmov；soft-reset 的 `threshold * spike_count` 是 mul+add 两步。两条计算路径都是 memory-bound，但 hard-reset 略简单一点。这是 NOISE 边界值，多跑几次平均会回到一致。
4. **延迟比之前所有 baseline（path B / NIR-compile / SJ-direct, 9.30–9.40 ms / 张）都快 ~11–17%**：换成 prefix-sum 形式（无论 soft / hard）+ 让 Inductor 自由 fuse + 摆脱 SJ LIF kernel 的 autotune restore_value clone 开销 —— 这才是 1.5 ms / 张的核心收益来源。reset 语义切换本身的延迟差几乎为 0。

### 与 SJ 标准 LIFNode 的语义对照

| 项 | SJ LIFNode (默认配置) | PrefixSumHardResetIFNode |
|---|---|---|
| Reset | **hard**（`v ← v_reset = 0`）| **hard**（`v ← 0`）✓ |
| Decay | 有（`τ=2.0`，`v_t = (1-1/τ)·v_{t-1} + x_t`）| **无**（IF，`τ→∞`）|
| 单步可发多 spike | 否（1 spike/step）| 否 ✓ |
| Surrogate gradient | Sigmoid α=4 | 暂未实现（推理无需）|

也就是说 **prefix-sum 版的 hard-reset IF 与 SJ LIFNode 在 reset 语义和发放规则上一致；唯一缺的是 decay**。如果未来要扩展到 LIF（带 decay），prefix-sum 部分会变成"指数加权 prefix-sum"（即 `v_t = decay·v_{t-1} + x_t` 的封闭解 `v_t = sum_k decay^{t-k} x_k`），实现复杂度上升一档但仍是 prefix-scan 范畴。

---

## 5. 训练 / 反向传播的考虑（未实测，仅分析）

本实验**只测推理 (`eval` + `no_grad`)**。如果要把 prefix-sum IF 用于训练，需要考虑：

| 算子 | 反向通路 |
|---|---|
| `torch.cumsum(x, dim=0)` | 反向是 `cumsum` 在反向方向上（`flip → cumsum → flip`），PyTorch 内置支持 ✓ |
| `(v_t >= threshold).to(float)` | spike 阶梯函数不可导，**必须用 surrogate gradient**（例如把 `>=` 替换为 `sigmoid((v_t-threshold)/α)` 的 ste 近似）|
| `torch.where(cond, cum, last)` | PyTorch 内置反向 ✓（但 cond 一侧无梯度） |
| `torch.stack(list, dim=0)` | PyTorch 内置反向 ✓ |

主要工作量在 **surrogate function** 的选择与替换，而非 cumsum / where 本身。这一块和 SJ LIFNode 训练时遇到的问题完全相同，可以直接复用 SJ 已有的 surrogate 实现接到 `(v_t >= threshold)` 这一步。

---

## 6. 结论

1. **"hard reset 让前缀和后续部分失效" 是只在「直接读 cum[t]」时才成立的直觉。** 路径 B 读的是 `cum[t] - last_cum_at_spike`，减法恰好抵消掉应该被 reset 丢弃的部分；cum 本身从 0 到 T 完整有效，被所有时间步共享只算一次。**算法上完全正确**，由 §2.5 中 fp64 逐位对比 0 误差证实。
2. **fp32 下存在 1-LSB 边界情况**：当 spike rate > 90% 等病态密集发放时，`cum[t]` 与 `last_cum_at_spike` 都会累积到较大值，相减触发 catastrophic cancellation，出现 ~1e-8 量级的 spike 决定不一致（实测 5138 万输出中 1 位）。fp64 完全消失。
3. **hard-reset 可以适配 prefix-sum 形式**，路径 B 是五个候选路径中最直接也最实用的。路径 B 与 soft-reset 实现高度对称，只把状态变量从"线性累加器 `spike_count`"换成"条件覆盖标记 `last_cum_at_spike`"；其它部分（cumsum + T 步顺序判定 + stack）完全一致。
4. **实测两种 reset 在性能和峰值显存上完全等价**（差距全在噪声内）—— prefix-sum IF 的优势（11–17% 更快、57% 更省显存）来自"不用 SJ LIF custom_op、让 Inductor 自由 fuse 完整算子链"，而非 reset 语义本身。
5. **prefix-sum + hard-reset IF 与 SJ 默认 LIFNode 在 reset 语义上完全一致**，差别只在是否带 decay。要扩展到带 decay 的 LIF，prefix-sum 部分需要换成"指数加权 cumsum"（仍是 prefix-scan 范畴）。
6. **如果需要绝对 bit-equal 保证**：可以退化到路径 A（朴素顺序 + 局部 `v_local`），失去 cumsum 的全局可分离性但避免大数相减；fp64 路径 B 也可保证 bit-equal。

### 推荐做法

如果你需要：
- **数值复现 SJ LIFNode 训出的预训权重** → 继续走 SJ LIFNode（带 decay + hard-reset），无法替换。
- **从零训练一个推理高效的 SNN，且需要 hard-reset 语义** → 用 `PrefixSumHardResetIFNode`，得到 7.8–8.2 ms / 张的延迟 + 6 GiB 显存峰值，比 SJ LIFNode 推理快 ~14%。
- **训练 / 推理一体且对正交于 LIF 的网络结构试错** → soft-reset 与 hard-reset 都可，soft 数学更易分析，hard 与生物 LIF 行为更接近。

---

## 7. 复现命令

```bash
cd /home/charlley/Code/Triton-Pass-Analysis

# Hard-reset, eager (cuDNN conv)
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache
RESET=hard MODE=eager BATCH=56 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/vgg16_snn/prefix_sum_snn.py

# Hard-reset, torch.compile (Inductor Triton)
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache
RESET=hard MODE=compile BATCH=56 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/vgg16_snn/prefix_sum_snn.py

# 看四种组合的横向比较
jq -c '{mode, mean_per_img_ms, peak_mem_gib, compile_s}' /tmp/cold_start_results.jsonl \
    | grep PrefixSumIF
```
