# 大 T 显存 OOM → 回退 python 循环：验证全过程记录

> 本文是**过程档案**：完整记录为验证「较大网络 + 较多时间步 T → triton 高性能路径因激活显存 ∝ T 而 OOM、
> 只能回退到 python 循环才能推理」这一现象所跑过的**全部实验**，包括每个实验的**网络结构与参数**、**详细结果**，
> 以及过程中遇到的波折、错误判断与修正（含被对抗性核查纠正的过度声称）。
> 简明结论见同目录 [`README.md`](./README.md)；本文偏重「跑了什么、怎么跑的、得到了什么、错在哪、怎么改对」。

---

## 0. 目标与判据

要验证的现象（用户定义）：**同一个网络，用 triton 会因 OOM 无法运行推理，而回退到 python 循环却可以运行推理。**

拆成三条可证伪的子命题：
1. **存在 T 阈值**：T 小时 triton 路径跑得通，T 大时 OOM。
2. **回退路径存活**：同一网络换成逐步 / 分块的 python 循环，在 triton 已 OOM 的 T 下仍跑得通。
3. **是真·激活显存 ∝ T 驱动的 OOM**，不是权重、不是碎片、不是别的崩溃误记。

---

## 1. 方法学与环境

### 1.1 硬件
- **NVIDIA GeForce RTX 5070 Ti**，显存 16 GiB（`torch` 报告总量 15.4 GiB，可用更少）。选它是因为 16 GiB 比可用的 A100（40 GiB）**更容易复现 OOM**。

### 1.2 软件环境（两套）
| 用途 | conda 环境 | 关键版本 |
|---|---|---|
| 实验 A、B（snn_compiler） | base | torch 2.9.1+cu128，triton 3.5.1 |
| 实验 C（最新版 spikingjelly + triton） | `sj_triton` | spikingjelly 0.0.0.0.15，triton 3.7.0，torch 2.12.0+cu130 |

### 1.3 不侵入原则
- **不修改、不写入 `snn_compiler` 开源项目目录**；实验代码全部放在仓库根的新目录 `experiments/large-T-oom-fallback/`，
  只把 snn_compiler 当库调用其**公开 API**（`fused_bias_if_lif`、`fused_bias_if_lif_stateful`、`zoo.vgg16_snn`）。
- **不修改任何第三方包文件**：spikingjelly 在 triton 3.7 下有一处 kernel bug（`triton_kernel/triton_utils.py` 的
  `convert_and_store` 多写一层 `.element_ty`），用**运行时 monkey-patch** 绕过（把 `integrate_and_fire`/`lif`/`plif`
  三个模块里 `from ..triton_utils import convert_and_store` 引入的名字替换成修好的 `@triton.jit` 版本），不动磁盘上的包。

### 1.4 测量纪律
- **显存**：`torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_allocated()`。
- **OOM 捕获**：`torch.cuda.OutOfMemoryError`（及消息含 `out of memory` 的 `RuntimeError`）。
- **子进程隔离**（实验 C 最终版必须用）：每个「(模式, T)」测量在**全新子进程 / 全新 CUDA 上下文**里跑，进程退出由
  操作系统回收全部显存——避免有状态神经元 / 失败前向的残留污染同进程内后续测量（§3.1 详述为何必须如此）。
- **真·OOM 判定**（实验 C 最终版）：子进程若没打印 `PEAK=` 行，**只在其 stderr 确含 `out of memory` 时才记 OOM**，
  否则记 `crash` 暴露出来（防「非 OOM 崩溃被误记为 OOM」，§3.3）。
- **冷启动**：涉及计时的地方先预热再取多次中位数；本文重点是显存/存活性，计时为辅。

---

## 2. 实验清单（结构 · 参数 · 测什么 · 详细结果）

脚本：[`oom_fallback_demo.py`](./oom_fallback_demo.py)（A+B）、[`sj_triton_oom.py`](./sj_triton_oom.py)（C）。原始日志在 [`results/`](./results/)。

### 2.1 实验 A — snn_compiler 整网 VGG-16 SNN（融合 triton）

**网络结构**：snn_compiler `zoo.vgg16_snn` —— 标准 VGG-16 主干（13 个 3×3 卷积，通道
`64,64,M,128,128,M,256,256,256,M,512,512,512,M,512,512,512,M`，M=2×2 maxpool）+ 分类头，每个 ReLU 换成 LIF 神经元，
conv-BN 折叠进卷积，最后 `Linear→1000` 类。

**参数**：
- 构造：`vgg16_snn(num_classes=1000, neuron="lif", tau=2.0, soft_reset=False, layout="NHWC", fused=True)`，
  随后 `.cuda().eval().to(torch.bfloat16)`。
- 神经元：LIF，`tau=2.0`，硬复位，`v_threshold=1.0`，`v_reset=0.0`。
- 数据布局 NHWC（channels_last），精度 **bf16**。
- 输入 `[T, B=8, 3, 224, 224]`，扫 `T ∈ {4,16,32,64,128,192,256,384}`（OOM 即停）。
- 执行方式：**全 T 融合**（一次 `model(x)`，每层物化 `[T,B,C,H,W]`）。

**测什么**：每个 T 的峰值显存、每图耗时、是否 OOM；并算 `Δpeak/ΔT` 看是否线性。

**详细结果**（`results/snn_compiler_oom.log`）：权重常驻显存 **0.258 GiB（与 T 无关）**。

| T | 状态 | 峰值显存 | 每图 ms | Δpeak/ΔT |
|---:|:--:|---:|---:|---:|
| 4 | OK | 0.90 GiB | 1.99 | — |
| 16 | OK | 2.60 GiB | 7.77 | 144 MiB/步 |
| 32 | OK | 4.93 GiB | 15.56 | 149 MiB/步 |
| 64 | OK | 9.60 GiB | 31.06 | 149 MiB/步 |
| **128** | **OOM** | **> 15 GiB** | — | 激活 ∝ T 超上限 |

**读出**：峰值显存随 T **线性**增长（≈149 MiB/步），权重恒定 → OOM 由**激活 ∝ T** 驱动而非权重。整网 **T=128 OOM**。

### 2.2 实验 B — snn_compiler 多层 conv-bn-LIF 栈：全 T 融合 vs 分块

这是为了把「全 T 物化」与「分块 python 循环」放在**同一网络、同一权重**上严格对比而自建的栈（结构可控、便于精确归因显存）。

**网络结构**（`ConvSNNStack`，5 层 conv-bn-LIF + 全局池化 + 分类头）：

| 层 | 算子 | 输出通道 | 池化 |
|---|---|---|---|
| 1 | Conv2d(3→64, 3×3, pad1, bias=False) + BatchNorm2d + LIF | 64 | — |
| 2 | Conv2d(64→64, 3×3, pad1) + BN + LIF | 64 | MaxPool 2×2 |
| 3 | Conv2d(64→128) + BN + LIF | 128 | — |
| 4 | Conv2d(128→128) + BN + LIF | 128 | MaxPool 2×2 |
| 5 | Conv2d(128→256) + BN + LIF | 256 | MaxPool 2×2 |
| 头 | AdaptiveAvgPool2d(1) → flatten → Linear(256→100) | 100 类 | — |

**参数**：
- BN running stats 随机初始化（`running_mean~N(0,0.1)`、`running_var~U(0.5,1)`、`weight~U(0.5,1)`、`bias~N(0.2,0.1)`），eval 模式。
- LIF：`tau=2.0`，`decay_input=True`，硬复位，`v_threshold=0.5`，`v_reset=0.0`。
- `B=16`，`H=W=112`，**bf16**；分块大小 `chunk=16`。
- 两条执行路径**共享同一权重**：
  - **全 T 融合**：每层 `conv-bn`（eager）后 `fused_bias_if_lif(y[T,B,C,H,W])` —— 每层物化整段 `[T,B,...]`，峰值 ∝ T。
  - **分块（逐 chunk python 循环 + 膜电位状态）**：外层按 `chunk` 切 T；每层 `fused_bias_if_lif_stateful` 用 fp32 的
    `v_init`/`v_out` 跨 chunk 串接膜电位；每层只持 `[chunk,B,C,H,W]`，外加每层一个 fp32 `[B,C,H,W]` 膜电位 + `[T,B,100]` 极小输出 → 峰值 ∝ chunk（与 T 无关）。

**测什么**：(a) 两路在**同一输入**下是否逐位一致；(b) 各 T 的峰值显存与存活性；(c) 速度。

**详细结果**（`results/snn_compiler_oom.log`）：

正确性（同一输入）：
- `T=24, chunk=8`：full-T 与 chunked **逐位一致**，`max|Δ|=0.000e+00`。
- `T=64, chunk=16`（在 full-T 仍跑得通的较大 T 上复测）：**逐位一致**，`max|Δ|=0.000e+00`。

显存 / 存活（B=16, H=112, chunk=16）：

| T | 全 T 融合 峰值 | 全 T | 分块 峰值 | 分块 | 速度比(chunk/full) |
|---:|---:|:--:|---:|:--:|---:|
| 16 | 1.94 GiB | OK | 1.97 GiB | OK | 1.01× |
| 32 | 3.87 GiB | OK | 2.02 GiB | OK | 1.01× |
| 64 | 7.74 GiB | OK | 2.02 GiB | OK | 1.00× |
| **128** | **> 16 GiB** | **OOM** | **2.16 GiB** | **OK ★ 仅分块存活** |  — |

**读出**：全 T 融合峰值 ∝ T（1.94→3.87→7.74，约翻倍），**T=128 OOM**；分块峰值 ≈ 常数（~2 GiB，∝ chunk）、**T=128 存活**；
两路**逐位一致**。（注：此处 chunk=16 的分块本身每块仍用 triton kernel，故 T≤64 速度与 full-T 基本相同；真正失去 triton 高性能的是 chunk=1 / 单步——见实验 C。）

### 2.3 实验 C — 最新版 spikingjelly + triton：多步 triton vs 单步 python

**这是「仅用 spikingjelly + triton」复现该现象的核心实验。**

**网络结构**（spikingjelly `layer.*` + `neuron.IFNode` 的多层脉冲 CNN）：
```
Sequential(
  layer.Conv2d(3,64,3,pad1,bias=False),  layer.BatchNorm2d(64),  neuron.IFNode(),
  layer.Conv2d(64,64,3,pad1,bias=False), layer.BatchNorm2d(64),  neuron.IFNode(),
  layer.MaxPool2d(2),
  layer.Conv2d(64,128,3,pad1,bias=False),layer.BatchNorm2d(128), neuron.IFNode(),
  layer.MaxPool2d(2),
  layer.Conv2d(128,128,3,pad1,bias=False),layer.BatchNorm2d(128),neuron.IFNode(),
  layer.AdaptiveAvgPool2d((1,1)), layer.Flatten(), layer.Linear(128,100),
)
```

**参数**：
- 神经元 `neuron.IFNode()` 默认：`v_threshold=1.0`，`v_reset=0.0`，硬复位（IF：`H=V+X`，无 τ 衰减）。
- `B=16`，`H=W=112`，精度 **fp32**（spikingjelly 默认）。
- 两条执行方式（**同一网络、同一权重**，只切 step-mode + backend）：
  - **多步 triton**：`functional.set_step_mode(net,'m')` + 神经元 `backend='triton'`。输入 `[T,B,3,H,W]`，`net(x)` 一次跑完整段 T，**每层物化 `[T,B,...]`**，峰值 ∝ T。这是「triton 高性能路径」（神经元 triton kernel 只在多步形态存在）。
  - **单步 python 循环**：`set_step_mode(net,'s')` + `backend='torch'`。`for t in range(T): net(x_t)`，每步只持 `[B,...]`，膜电位由神经元内部跨步维持，峰值 ≈ 常数。这是「回退的 python 循环」。

**测什么**：(a) 两路在同一输入下的差异（探针 T=8 + 大 T 等价 T=32）；(b) 各 T 峰值显存与存活；(c) OOM 是否真·显存 OOM。

**详细结果**（`results/sj_triton_oom.log`，**子进程隔离**测显存）：

后端探针：`spikingjelly IFNode 支持后端 = ('torch','cupy','triton','inductor')`；monkey-patch 生效（4 个模块）。

显存 / 存活（B=16, H=112）：

| T | 多步 triton 峰值 | 多步 triton | 单步 python 峰值 | 单步 python |
|---:|---:|:--:|---:|:--:|
| 4 | 1.46 GiB | OK | 0.45 GiB | OK |
| 16 | 4.93 GiB | OK | 0.45 GiB | OK |
| 32 | 9.56 GiB | OK | 0.45 GiB | OK |
| **64** | **OOM** | **OOM** | **0.45 GiB** | **OK ★** |
| 128 / 256 / 512 / 1024 | OOM | OOM | ~0.45–0.46 GiB | OK ★ |

**读出**：多步 triton 峰值 ∝ T（1.46→4.93→9.56），**T=64 起 OOM**；单步 python 循环峰值**恒定 ~0.45 GiB**，**一路存活到 T=1024**。
现象在 spikingjelly+triton 上成立。

正确性（同一输入）：
- 探针 `T=8`（小网络）：`max|Δ|=0.000e+00`。
- **大 T 等价 `T=32`：`max|Δ|=1.351e-3`** —— **近似一致、非逐位**（见 §3.2，这是被对抗性核查纠正的一点）。

### 2.4 expandable_segments 对照（排除「显存碎片」）

OOM 报错会提示 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（缓解碎片）。开启后单独跑实验 C 的多步 triton worker：

| T（多步 triton, expandable_segments:True） | 峰值 / 状态 |
|---:|---|
| 32 | 9.558 GiB / OK |
| 48 | 14.188 GiB / OK |
| **64** | **OOM** |
| 64（单步 python，对照） | 0.448 GiB / OK |

**读出**：换分配器后峰值仍 ∝ T、仍在 T=64 OOM → **是根本的激活显存需求超上限，不是碎片**。

### 2.5 正确性探针 + 大 T 等价性（方法）
- 探针在同进程小网络小 T 下跑，仅验「triton 后端能跑通 + 与单步同一输入大致一致」。
- 大 T 等价性用**子进程**在「多步仍跑得通的较大 T」（C 用 T=32、B 用 T=64）上、**用同一输入**再测一次差异——
  这是对抗性核查要求补的，正是它在 C 上暴露了 1.35e-3（探针的小 T 看不出）。

---

## 3. 过程中的波折、错误判断与修正（详细结果演变）

这一节是「过程中的详细结果」的核心——记录哪些中间结果是错的、为什么、怎么改对。

### 3.1 实验 C 第一版（同进程测量）→ 单步显存被污染
- **做法**：同一进程里，对每个 T 先测多步、再测单步，中间 `empty_cache`。
- **错误结果**：单步峰值在 T≤32 时干净（0.46/0.48/0.52 GiB），但 **T≥64 时跳到 9.83–11.56 GiB**。
- **原因**：多步在该 T 先 OOM，留下的显存（有状态神经元的膜电位 + 失败前向被 traceback 引用的张量）`empty_cache` **回收不掉**，污染了随后单步测量的基线。
- **第二版尝试**（每个 T 重建网络 + 单步逐步生成输入）：单步反而更糟，1.67→5.55→10.73→OOM，**说明跨迭代的残留 empty_cache 也清不掉**（spikingjelly 的 triton custom-op 持有显存）。
- **修正**：改为**子进程隔离**——每个 (模式,T) 在全新进程跑、退出即由 OS 回收。第三版结果干净：单步**恒定 ~0.45 GiB**（即 §2.3 的最终表）。

### 3.2 「多步 triton 与单步 python 逐位一致」被纠正为「近似一致 ~1e-3」
- **初始声称**：基于探针（T=8）的 `max|Δ|=0`，我曾声称 spikingjelly 两路逐位一致。
- **对抗性核查要求**：在**大 T、同一输入**下复测，不要只信小 T 探针。
- **复测结果**：`T=32` 时 `max|Δ|=1.351e-3` —— **并非逐位一致**。
- **原因**：多步走 triton 后端、单步走 torch 后端，且卷积按 `[T·B]` vs `[B]` 不同批次计算，fp32 浮点累加差异随 T 累积；功能等价（分类不受影响）但非逐位。
- **对照**：snn_compiler 的 full-T 与 chunked 用的是**字节相同**的两个 kernel（`_bias_if_lif_kernel` 与 `_bias_if_lif_stateful_kernel`），膜电位以 fp32 无损跨块串接 → **真·逐位一致**（已在 T=24 **和 T=64** 同一输入上确认 `max|Δ|=0`）。
- **结论修正**：snn_compiler **逐位一致**；spikingjelly **近似一致 ~1e-3**。这是本轮最重要的诚实修正。

### 3.3 「假 OOM」隐患 → stderr 真·OOM 判定
- **隐患**（核查指出）：实验 C 的子进程分类器最初把「任何没打印 `PEAK=` 的子进程」一律记为 OOM——但 timeout / triton 编译错 / 非法地址 / import 失败也会没有 `PEAK=`，会被**误记为 OOM**，恰好落在被声称「会 OOM」的多步路径上。
- **修正**：(a) worker 显式捕获 `RuntimeError` 中含 `out of memory` 的情形并打印 `STATUS=oom`，其它异常照常抛出；
  (b) 父进程**检查子进程 stderr**，只有确含 `out of memory` 才记 `oom`，否则记 `crash(rc=...)` 暴露出来。
- **重跑结果**：多步在 T≥64 全部判为 **真·显存 OOM**（日志显示 `OOM` 而非 `crash`）→ 排除假 OOM。

### 3.4 实验 B 高 T（T=256）同进程卡死 → 封顶重跑
- **现象**：实验 B 跑到 T=256 时，full-T 先 OOM 留下 ~14 GiB 回收不掉，随后 chunked 在近满显存上**卡死约 10 分钟无进展**（同 §3.1 的残留问题，但出现在 snn_compiler 侧的同进程扫描中）。
- **修正**：杀掉卡死进程，把实验 B 的 T 扫描**封顶到 T=128**（恰好是 full-T OOM、chunked 存活的临界点，且不触发卡死的高 T 行），重跑得到干净完整的日志（即 §2.2 的最终表 + 完成标记）。
- **教训**：对有状态/会 OOM 的路径，**同进程多 T 扫描不可靠**，应像实验 C 那样子进程隔离（B 因 full-T 路径无状态、numbers 在 T≤128 一直干净，故封顶即可）。

---

## 4. 对抗性核查（独立怀疑论者 + 本机复现）

用一个多智能体核查流程（4 个不同审查角度 + 综合）对结论做对抗性核查。两个先返回的怀疑论者（"experiment-bugs"、"fallback-validity"）：
- 结论均为 **supported_with_caveats**；
- **在本机独立复现**：跑了 snn_compiler 的正确性检查，得 `max|Δ|=0`（chunk=8/7/1）；验证了 monkey-patch 真生效（patched n=4、kernel 取到修复版）、spikingjelly 包确实有该 bug；核对了「峰值 ∝ T 的多步 vs ≈常数的单步」机理（估算实验 B T=256 时输入 0.29 GiB vs 单层激活 6.12 GiB → OOM 由激活而非输入驱动）；
- **提出的有效问题**（均已在 §2–§3 落实）：假 OOM 分类、仅小 T 验逐位、step-mode 与 backend 混淆、缺原始日志产物、autotuner bench-cache 也可能触发 OOM。

> 过程提示：核查中有智能体**自行在本机重跑实验**占用了 GPU，多个并行智能体在单卡上**争用显存**导致后续核查不可靠；
> 我据此停止该流程、改为按其已给出的有效结论逐条加固（§3.2/§3.3 + 大 T 等价 + 子进程隔离 + 提交原始日志），并补了 expandable_segments 对照。

**关于 step-mode vs backend 的澄清**（核查重点）：OOM 的根因是**「多步 / 全 T 物化 `[T,B,...]`」这一执行方式**，不是 triton 本身——
即便多步用 torch 后端也会 OOM。triton 是让**多步**路径**变快**的后端；要享受 triton 高性能就得走多步、就得物化 `[T,B,...]`。
回退路径是**单步 python 循环**（spikingjelly 中即 torch 后端，神经元 triton kernel 没有单步形态）。故「triton 路径 OOM、python 循环存活」成立且表述精确。

---

## 5. 结论

1. **现象成立、且双框架复现**：snn_compiler（实验 A 整网 VGG-16 在 T=128 OOM；实验 B full-T 在 T=128 OOM 而分块存活）与
   最新版 spikingjelly + triton（实验 C 多步 triton 在 T=64 起 OOM 而单步 python 一路存活到 T=1024）都证实：
   **triton 的全 T / 多步路径在大 T OOM、跑不动；逐步 / 分块 python 循环低显存、跑得通。**
2. **机理坐实**：峰值显存随 T **线性**（实验 A 149 MiB/步、权重恒定）、分块/单步峰值**与 T 无关**；OOM 是**激活 ∝ T** 驱动。
3. **排除替代解释**：不是碎片（expandable_segments 仍 OOM）、不是假 OOM（stderr 真·OOM 判定）、不是「网络太大与 T 无关」（小 T 跑得动、单步全 T 跑得动）。
4. **正确性（诚实结论）**：snn_compiler 两路**逐位一致**（T=24 与 T=64 同一输入 `max|Δ|=0`）；spikingjelly 两路**近似一致 ~1.35e-3（T=32）**，功能等价但非逐位。

---

## 6. 复现与产物

```bash
# A + B（snn_compiler，base 环境）
python experiments/large-T-oom-fallback/oom_fallback_demo.py
# C（最新版 spikingjelly + triton，sj_triton 环境）
~/miniconda3/envs/sj_triton/bin/python experiments/large-T-oom-fallback/sj_triton_oom.py
#   单配置子进程：... sj_triton_oom.py --worker {multistep|singlestep|equiv} <T> <B> <H>
#   排除碎片对照：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ... --worker multistep 64 16 112
```

产物（均在 `experiments/large-T-oom-fallback/`，不触碰 `snn_compiler`）：
- `oom_fallback_demo.py`（A+B）、`sj_triton_oom.py`（C：子进程隔离 + monkey-patch + stderr 真 OOM 判定 + 等价性复测）；
- `results/snn_compiler_oom.log`、`results/sj_triton_oom.log`（本机真实运行日志）；
- `README.md`（简明结论）、本文（全过程档案）。
