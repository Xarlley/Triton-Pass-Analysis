# AutoChunkInference —— SpikingJelly-Triton 路线的全自动分块推理

把"整段 T 物化"的脉冲网络（典型：**SpikingJelly 多步 + triton 后端**，神经元 `step_mode='m', backend='triton'`，
一次前向处理整段 `[T,B,...]`）包成**自动分块推理**：在**不触发 OOM、不触发编译悬崖**的前提下，**最大化利用显存**地
把任意大 T 的推理跑通，而无需回落到逐步 python 循环。

成果已增量集成进 snn_compiler：[`snn_compiler/nn/autochunk.py`](../../snn_compiler/nn/autochunk.py)
（`from snn_compiler.nn import AutoChunkInference`）。本目录是其**探索与验证脚本**（非侵入，独立于 snn_compiler 包）。

---

## 1. 它解决什么问题

在 `experiments/large-T-oom-fallback/` 我们已实测：同一个 SNN，在 SpikingJelly 多步 triton 后端下，
**激活显存 ∝ T**，大 T 必 OOM；而本轮进一步发现大 T 还会撞上**编译悬崖**。两类失败模式：

| 网络类型 | 大 T 整段多步的失败模式 | 机理 |
|---|---|---|
| compute-bound（大 B·H，卷积重） | **OOM** | 激活 `[T,B,C,H,W]` 显存 ∝ T，超显存 |
| launch-bound（小 B·H，算子碎小） | **编译悬挂（>100s 不返回）** | 神经元 kernel `static_range(T)` 编译期展开，耗时随 T 超线性 |

整段多步一旦失败，只能回落到逐步 python 循环（只持 `[B,...]`），**丢掉 triton 的高性能**。
AutoChunkInference 在两类失败上都让推理**用 triton 跑通**。

---

## 2. 顺序决策：优化在内、分块在外（optimize-then-chunk）

> 任务要求自行判断"先优化再分块"还是"先分块再优化"。**结论：优化在内、分块在外。**

- **非侵入**：分块是一层**外部驱动**，对已优化的模型（多步 triton 后端 / snn_compiler 融合）原封不动，只把时间维
  切片后喂进去。既有推理加速功能完全不受影响（满足"不要破坏原有优化"）。
- **天然封住编译悬崖**：神经元 triton kernel 用 `static_range(chunk)` 在编译期展开，编译耗时随展开长度超线性。
  分块把每次喂进的时间长度限制为 `chunk`（≤64）→ kernel 只展开 `chunk` 步而非 `T` 步 → **编译被钉在安全区**。
  若反过来"先分块再优化"，对 SpikingJelly 路线无意义（优化本身就是多步 triton 后端，分块是其外层驱动）。
- **每块满速**：每个 chunk 仍走完整的优化 kernel（triton 编译一次、跨块复用），单块吞吐 = 整段 triton 吞吐。
- **正确性靠状态串接**：`reset` 一次 → 按块喂 `x[i:i+chunk]` → **块间不 reset**，神经元自带的膜电位 `v` 跨块串接
  （SpikingJelly 多步神经元满足，已实测，见 §4.1）。

---

## 3. 自动选块策略（基于实测定律）

### 3.1 显存：实测倍增搜索，绝不主动 OOM
早期版本用"两点仿射外推一步到位"（`mem≈a·chunk+b`，该律对 snn_compiler 自带融合 triton 卷积成立，R²≈1）。
但 SpikingJelly 走 `torch.nn.Conv2d`→**cuDNN**，大 batch 下 cuDNN 切换算法/申请工作区，**显存对 chunk 非仿射**，
从 chunk=2,4 外推到 64 严重低估而 OOM —— 且**一次主动 OOM 会污染分配器**，使后续更小的块也放不下（实测：失败的
chunk=64 后仅余 811 MiB，本可放下的 chunk=32 也 OOM）。

改为**倍增搜索**：从小块起步，逐步把 chunk 翻倍并在**真实尺寸**上实测峰值，仅当"再翻倍预计 ≤ 预算"时才真正探针下一档，
**绝不探针一个预计会 OOM 的尺寸**。对 cuDNN 也稳健。`forward` 仍保留"真跑时 OOM 就减半重试"的兜底（理论上选块已留余量，极少触发）。

### 3.2 速度/编译：compute-bound 与 launch-bound 的不同策略，但共享编译天花板
- 编译耗时随 `static_range(chunk)` 展开长度**超线性**，**与 regime 无关**（实测 chunk=64 可编译；chunk≥128 / full@T=256
  编译悬挂 >100s）。故两类都封顶在**编译安全上限 64**。
- **差别在于谁先成为瓶颈**：
  - **launch-bound**（小 B·H）：小块极慢（`large-T-oom-fallback` 实测 chunk=1 最高 **54×** 慢），显存极宽裕
    → 选块由**编译上限**封顶（≈64）。策略 = "为速度尽量取大块"。
  - **compute-bound**（大 B·H）：速度对 chunk 不敏感，但激活 ∝ chunk → 选块由**显存预算**先封顶（常 <上限）。
    策略 = "为显存利用率取最大可放块、速度无损"。
- 用一次（已热身、排除编译）的轻量速度探针判别 regime（单帧耗时随块增大是否显著下降）。

> **对早期设想的修正**：曾设想 launch-bound 应取**远大于** compute-bound 的块。实测表明 chunk>64 触发编译悬崖、
> 而 launch 开销在 ~64 帧后已摊薄、速度收益饱和 —— 故两者收敛到同一编译安全上限，真正的差异体现在
> "由**显存**还是由**编译上限**封顶"，而非上限值不同。

---

## 4. 验证结果（本机 RTX 5070 Ti 16 GiB, sj_triton 环境：spikingjelly 0.0.0.0.15 + triton 3.7 + torch 2.12）

### 4.1 前提：SpikingJelly 多步神经元跨块串接 v（`verify_premise.py`）
`reset` 一次后按块喂、块间不 reset：`max|Δ(整段 vs 分块)|` 在 chunk∈{1,2,3,6,8,12} 上**恒定** ≈3.8e-3 → 串接成立
（残差是 cuDNN 卷积按 `[chunk·B]` vs `[T·B]` 的批次数值差，非 bug）。

### 4.2 正确性：驱动逐位精确，conv 残差是 cuDNN 固有差（`test_generalization.py` §1）

| 模型 | 配置 | nchunks | dchunk(分块vs整段) | dsingle(单步vs整段) |
|---|---|---|---|---|
| fc（无卷积） | T24 B4 H16 | 3 | **5.96e-8** | 5.96e-8 |
| conv_wide | T12 B4 H48 | 3 | **0.0** | 0.0 |
| conv_small | T24 B8 H64 | 3 | 2.69e-2 | 4.47e-8 |
| conv_small | T24 B8 H112 | 3 | 1.62e-2 | 5.96e-8 |

- **fc 逐位一致(~1e-7)** → 证明分块驱动（reset 一次→按块喂→串接→cat）本身**数值精确**。
- **conv 残差**来自 cuDNN：单步(batch=B) 与整段(batch=T·B) 一致到 1e-7，而分块(batch=chunk·B) 在**特定 batch**
  （如 64）下 cuDNN **确定性地**选了不同卷积算法 → ~1e-2 差，与脉冲阈值耦合后偶发翻转一个脉冲。
  - 关键判别：`cudnn.deterministic=True` **不改变**该差（排除"非确定性 autotune"）；且分块差**非单调**于 chunk
    （只在某些 batch 出现，chunk=1 逐位一致），是 cuDNN 按 batch 选算法的签名，**不是分块/串接 bug**。
  - 与 snn_compiler 既有 [[BN-fold 非逐位一致]] / 多步vs单步 **同源**，确定性、训练网上预测稳定。

### 4.3 不 OOM / 不编译悬挂 + regime（`test_generalization.py` §2）

| 模型 | T | B | H | 整段多步 | autochunk | chunk_t | regime | 净峰值/预算(GiB) |
|---|---|---|---|---|---|---|---|---|
| conv_small | 256 | 16 | 112 | **OOM** | OK | 32 | compute-bound | 6.86 / 11.9 |
| conv_small | 512 | 16 | 112 | **OOM** | OK | 32 | compute-bound | 7.43 / 11.4 |
| conv_wide | 128 | 8 | 112 | **OOM** | OK | 32 | compute-bound | 4.97 / 12.3 |
| conv_wide | 256 | 8 | 64 | **OOM** | OK | 64 | compute-bound | 3.20 / 12.3 |
| conv_small | 256 | 1 | 8 | **编译悬挂>100s** | OK | 64 | launch-bound | 0.04 / 12.4 |
| fc | 256 | 1 | 16 | **编译悬挂>100s** | OK | 64 | launch-bound | 0.04 / 12.4 |
| fc | 512 | 4 | 32 | **编译悬挂>100s** | OK | 64 | launch-bound | 0.06 / 12.4 |

- 整段多步在 compute-bound 上 **OOM**、在 launch-bound 上**编译悬挂**；autochunk 两类都**跑通**。
- **regime 判别正确**：大 B·H → compute-bound，选块由**显存**封顶（32/64）；小 B·H → launch-bound，选块由**编译上限**封顶（64）。
- compute-bound 净峰值占预算 ~55–74%（块尺寸按 2 的倍数离散，再翻倍即超预算 → 取最大不超预算的安全块）。

---

## 5. 用法

```python
from snn_compiler.nn import AutoChunkInference
from spikingjelly.activation_based import functional

# net：已设好 step_mode='m', backend='triton' 的 SpikingJelly 模型（优化在内）
auto = AutoChunkInference(net, reset_fn=functional.reset_net)   # 分块在外，非侵入
y = auto(x_seq)            # x_seq=[T,B,...]；首次自动选块并缓存(按输入形状)，之后直接复用
print(auto.last_plan)      # 选块诊断：regime / chunk_t / 实测峰值 / 预算
```

- `memory_fraction=0.85`：用空闲显存的多大比例做激活预算（留余量防碎片/他人占用）。
- `compile_cap=64 / launchbound_cap=64`：编译安全上限（见 §3.2；kernel 在大展开下编译仍快时可上调）。
- `fixed_chunk=k`：跳过自动选块，直接用 k（复现/调试）。
- 默认 `reset_fn` 为 duck-type 调子模块 `.reset()`，兼容 SpikingJelly，**不 import 它**。

---

## 6. 脚本清单

| 脚本 | 作用 |
|---|---|
| `verify_premise.py` | 验证 SJ 多步神经元跨块串接 v（分块多步 == 整段多步） |
| `test_autochunk_sj.py` | 单模型端到端：正确性 + 大 T 不 OOM + 选块/峰值 |
| `test_generalization.py` | 多模型(conv_small/conv_wide/fc) × 多参数：正确性 / 不OOM不悬挂 / regime（子进程隔离） |
| `generalization.log` | `test_generalization.py` 的实测输出存档 |

> 单元测试（随 snn_compiler 测试套件）：[`snn_compiler/tests/test_autochunk.py`](../../snn_compiler/tests/test_autochunk.py)
> —— CPU 驱动逐位正确性 + CUDA 自动选块。
