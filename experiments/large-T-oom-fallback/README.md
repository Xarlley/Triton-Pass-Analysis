# 大 T 显存 OOM → 回退 python 循环：现象验证

> **独立实验**（不修改、不写入 `snn_compiler` 开源项目，也不修改任何包文件）。本目录把 snn_compiler 与
> spikingjelly 当库调用，验证一个现象：**较大网络 + 较多时间步 T 时，「全 T / 多步」的 triton 高性能路径
> 会因激活显存 ∝ T 而 OOM、无法完成推理；而「逐时间步 / 分块」的 python 循环路径（低显存）却能跑通**。
>
> 硬件：本机 **RTX 5070 Ti（16 GiB，实际可用 ~15.4 GiB）**——比可用的 A100（40 GiB）更易复现该现象。
> 实测日志见 [`results/`](./results/)（`snn_compiler_oom.log`、`sj_triton_oom.log`，均为本机真实运行输出）。

## 结论（一句话）

**同一个网络**：用 triton 的全 T / 多步路径在大 T 时 **OOM 跑不动**，回退到 python 逐步 / 分块循环则 **跑得动**。
在 snn_compiler 与「最新版 spikingjelly + triton」上**都复现**。两条路径的数值关系：snn_compiler **逐位一致**；
spikingjelly **近似一致（~1e-3）**（见 §正确性）。

---

## 机理：峰值显存 ∝ T vs ≈常数

脉冲网络推理要在时间维 T 上展开。两类执行方式的峰值显存截然不同：

| 执行方式 | 中间激活 | 峰值显存 | 是不是 triton 高性能 |
|---|---|---|---|
| **全 T / 多步（materialize-all-T）** | 每层都物化整段 `[T, B, C, H, W]` | **∝ T** | triton 是其高性能后端（一个 kernel 跑完整段 T） |
| **逐步 / 分块（python 循环）** | 每层只持 `[B,…]` 或 `[chunk, B,…]` + 膜电位状态 | **≈ 常数（与 T 无关）** | 否（逐步/逐块，退回 torch 或失去跨 T 融合） |

> **澄清（step-mode vs backend）**：OOM 的根因是**「多步 / 全 T 物化 `[T,B,...]`」这个执行方式**，不是 triton 本身——
> 即便多步用 torch 后端也会 OOM。triton 是让**多步**路径**变快**的后端；要享受 triton 的高性能就得走多步、就得物化 `[T,B,...]`。
> 回退路径是**单步 python 循环**（spikingjelly 中即 torch 后端，triton 神经元 kernel 没有单步形态）。所以「triton 路径 OOM、python 循环存活」成立。

存在一个 T 阈值：超过它，多步 / 全 T 路径的激活超出显存上限 → OOM；逐步 / 分块路径不随 T 增长，始终跑得动。
这正是 snn_compiler 提供 `ChunkedForward`/`fused_bias_if_lif_stateful`、spikingjelly 提供单步模式的原因。

---

## 实测数据

显存为 `torch.cuda.max_memory_allocated`。spikingjelly 实验用**子进程隔离**（每个测量在全新 CUDA 上下文里跑，
进程退出由操作系统回收，避免有状态神经元 / 失败前向的残留污染同进程后续测量——**实测同进程会严重污染**，故必须隔离）。

### A — snn_compiler 整网 VGG-16 SNN（融合 triton, bf16, NHWC, B=8）

权重常驻显存 = **0.258 GiB（与 T 无关）**。

| T | 状态 | 峰值显存 | 每图 ms | Δpeak/ΔT |
|---:|:--:|---:|---:|---:|
| 4 | OK | 0.90 GiB | 1.99 | — |
| 16 | OK | 2.60 GiB | 7.77 | 145 MiB/步 |
| 32 | OK | 4.93 GiB | 15.56 | 149 MiB/步 |
| 64 | OK | 9.60 GiB | 31.06 | 149 MiB/步 |
| **128** | **OOM** | **> 15 GiB** | — | （激活 ∝ T 超上限） |

→ 峰值显存**随 T 线性增长（149 MiB/步）**、权重恒定 → OOM 由**激活 ∝ T** 驱动而非权重。整网在 T=128 OOM。

### B — snn_compiler 多层 conv-bn-LIF 栈：全 T 融合 vs 分块（B=16, H=112, chunk=16）

| T | 全 T 融合(triton) 峰值 | 全 T | 分块(python循环+膜电位) 峰值 | 分块 |
|---:|---:|:--:|---:|:--:|
| 16 | 1.94 GiB | OK | 1.97 GiB | OK |
| 32 | 3.87 GiB | OK | 2.02 GiB | OK |
| 64 | 7.74 GiB | OK | 2.02 GiB | OK |
| **128** | **> 16 GiB** | **OOM** | **2.16 GiB** | **OK ★ 仅分块存活** |

→ 全 T 融合峰值 ∝ T、T=128 OOM；分块峰值 ≈ 常数（∝ chunk）、存活。

### C — 最新版 spikingjelly + triton：多步 triton vs 单步 python（子进程隔离, B=16, H=112）

环境 `sj_triton`：**spikingjelly 0.0.0.0.15 + triton 3.7.0 + torch 2.12**。模型为 spikingjelly IFNode 多层脉冲 CNN。

| T | 多步 triton 峰值 | 多步 triton | 单步 python 循环 峰值 | 单步 python |
|---:|---:|:--:|---:|:--:|
| 4 | 1.46 GiB | OK | 0.45 GiB | OK |
| 16 | 4.93 GiB | OK | 0.45 GiB | OK |
| 32 | 9.56 GiB | OK | 0.45 GiB | OK |
| **64** | **OOM** | **OOM** | **0.45 GiB** | **OK ★** |
| 128 / 256 / 512 / 1024 | OOM | OOM | ~0.45 GiB | OK ★ |

→ 多步 triton 峰值 ∝ T、T=64 起 OOM；单步 python 循环峰值**恒定 ~0.45 GiB**、一路存活到 T=1024。

> 不改 spikingjelly 包文件：仅在运行时 monkey-patch 其 triton-3.7 下的一处 bug
> （`convert_and_store` 多写一层 `.element_ty`），见 `sj_triton_oom.py`。

---

## 正确性（回退路径算的是同一个东西吗？）

| 框架 | 对比 | 结果 |
|---|---|---|
| **snn_compiler** | 全 T 融合 vs 分块（**同一输入**, T=24 chunk=8 与 **T=64 chunk=16**） | **逐位一致 `max\|Δ\|=0`** |
| **spikingjelly** | 多步 triton vs 单步 torch（同一输入, 小 T=8 / T=32） | T=8: `max\|Δ\|=0`；**T=32: `max\|Δ\|=1.35e-3`（近似一致，非逐位）** |

- snn_compiler 的全 T kernel(`_bias_if_lif_kernel`)与有状态 kernel(`_bias_if_lif_stateful_kernel`)是**字节相同**的 fp32 递推，
  分块只把膜电位 `v` 以 fp32 无损跨块串接 → **逐位一致**（在 full-T 仍跑得通的 T=64 上也已确认）。
- spikingjelly 多步(triton)与单步(torch)走**不同后端**、且卷积按 `[T·B]` vs `[B]` 批次计算，浮点累加差异随 T 累积
  → 大 T 下为**近似一致 ~1e-3**（功能等价）而非逐位。**这一点由对抗性核查在大 T 复测时发现**（小 T 探针看不出）。

---

## 排除替代解释

- **不是显存碎片**：开启 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后，多步 triton 仍 OOM——
  T=32 峰值 9.56 GiB / OK、T=48 14.19 GiB / OK、**T=64 仍 OOM**，单步 python T=64 仅 0.45 GiB / OK。峰值仍 ∝ T。
- **不是「假 OOM」（崩溃误记）**：`sj_triton_oom.py` 的子进程分类器**检查 stderr**，只有确含 `out of memory` 才记 OOM，
  其它失败（timeout / 编译错 / 非法地址）记为 `crash` 暴露出来。实测多步失败全部判定为**真·显存 OOM**（日志中显示 `OOM` 而非 `crash`）。
- **不是「网络太大、与 T 无关」**：同一网络在小 T（4/16/32）下跑得动、大 T OOM；且单步 python 在**全部** T 下都跑得动。

> 上述结论经一轮**对抗性核查**（多个独立怀疑论者审代码 + 在本机复现）后修正定稿：核查确认了「峰值 ∝ T 的多步
> vs ≈常数的单步」、monkey-patch 真生效、snn_compiler 逐位一致；并纠正了「spikingjelly 也逐位一致」的过度声称（实为 ~1e-3 近似）。

---

## 复现

```bash
# A + B（snn_compiler；本机 base 环境 torch 2.9 + triton 3.5）
python experiments/large-T-oom-fallback/oom_fallback_demo.py

# C（最新版 spikingjelly + triton；sj_triton 环境 torch 2.12 + triton 3.7）
~/miniconda3/envs/sj_triton/bin/python experiments/large-T-oom-fallback/sj_triton_oom.py
#   单配置子进程：... sj_triton_oom.py --worker {multistep|singlestep|equiv} <T> <B> <H>
#   排除碎片   ：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ... --worker multistep 64 16 112
```

- 不修改 `snn_compiler` 开源项目，也不修改任何包文件。A/B 仅 import snn_compiler 公开 API
  （`fused_bias_if_lif` 全 T、`fused_bias_if_lif_stateful` 分块、`zoo.vgg16_snn`）；C 仅 import spikingjelly
  公开 API + 运行时 monkey-patch 一处 triton-3.7 bug。
- 原始运行日志见 [`results/`](./results/)。
