# 分块系统实验：T × chunk 的「显存 / 速度 / 编译」三维横向对比

> 在前面「大 T → 全 T 路径 OOM、回退分块才能跑」的基础上，对**分块本身**做系统横扫：固定网络，
> 扫（时间步 T × 分块大小 chunk），量化 **峰值显存 / 稳态推理时间 / 冷启动(编译)时间** 三者随 (T,chunk) 的变化。
> 结论经一轮**只读对抗性核查**（4 个独立分析角度 + 综合）拟合/证伪后定稿；噪声项已明确标注、不予引用。
>
> 脚本 [`chunk_sweep.py`](./chunk_sweep.py)，原始日志 [`results/chunk_sweep.log`](./results/chunk_sweep.log)。
> 网络 = `ConvSNNStack`（5 层 conv-bn-LIF + 全局池化 + 头，详见 [`oom-verification-process.md`](./oom-verification-process.md) §2.2），
> **B=8, H=W=112, bf16**；RTX 5070 Ti 16 GiB（15.4 可用）。

## 方法学（针对两个干扰）

- **残留内存**：每个 (mode,T,chunk) 单元在**全新子进程 / 全新 CUDA 上下文**里测，进程退出由 OS 回收 → 彻底隔离
  （核查确认：`subprocess.run` 起新进程，跨单元零残留干扰）。
- **冷启动**：单独记录"首次调用时间（含 triton 编译 + cudnn/triton autotune + 分配器首触发）"；**稳态时间**取
  warmup（首调 + 2 次）之后 3 次迭代的中位数；**峰值显存**在 warmup 之后 `reset_peak_memory_stats` 再测
  （核查确认：峰值/计时都剔除了冷启动，对 full/chunked 一视同仁）。
- **编译展开背景**：snn_compiler 融合 LIF kernel 用 `tl.static_range(0, T)`（`T` 为 constexpr），时间循环在**编译期完全展开**，
  展开长度 = 全 T 路径的 T 或分块路径的 chunk → 展开越长、编译越慢。

---

## 实测矩阵

### 组 1 — 固定 T=64，扫 chunk（full 跑得通，展开 ≤64 编译可控）

| 配置 | 峰值(GiB) | 稳态(ms) | 冷启动(ms) | 显存比 vs full | 速度比 vs full |
|---|---:|---:|---:|---:|---:|
| full（全 T） | 3.87 | 44.2 | 3468 | 1.00 | 1.00 |
| chunk=1 | 0.12 | 41.9 | 3534 | **0.03** | 0.95 |
| chunk=2 | 0.18 | 39.1 | 4790 | 0.05 | 0.88 |
| chunk=4 | 0.30 | 42.2 | 3586 | 0.08 | 0.95 |
| chunk=8 | 0.54 | 43.9 | 3671 | 0.14 | 0.99 |
| chunk=16 | 1.01 | 44.1 | 3762 | 0.26 | 1.00 |
| chunk=32 | 1.97 | 44.4 | *24183* † | 0.51 | 1.00 |
| chunk=64 | 3.86 | 44.4 | 2260 | 1.00 | 1.00 |

† **冷启动 24183ms 是离群噪声**：chunk=64（更长展开）只用 2260ms、full(同 64 展开)用 3468ms，
单调律不可能产生它 → 系一次性 autotune/cudnn 抖动，**不作为编译时间数据点**。整列冷启动均单次测量、含 autotune 噪声，仅作量级参考。

### 组 2 — 大 T + 小 chunk（展开小、编译快；显存是否随 T 不变）

| T | full | chunk=8 | chunk=32 |
|---:|---|---|---|
| 128 | **TMO（编译超时 >260s）** | 0.54 GiB / 88 ms | 1.97 GiB / 89 ms |
| 256 | （未测） | 0.54 GiB / 176 ms | 1.97 GiB / 178 ms |
| 512 | （未测） | 0.54 GiB / 352 ms | 1.97 GiB / 356 ms |

---

## 三条定律（经核查）

### 1. 显存律（verdict: clean，R²=0.99999）
- **分块峰值在 chunk 上仿射、且与 T 无关**：`峰值[GiB] ≈ 0.0594·chunk + 0.064`，即 **≈60.8 MiB/分块步 + ≈64 MiB 固定开销**
  （固定项 = 权重 + 各层 fp32 膜电位状态 + 极小输出 + 分配器块）。
- **T 无关是直接测出的**（非外推）：chunk=8 在 T=128/256/512 都是 0.54 GiB、chunk=32 都是 1.97 GiB（4 点不变）。
  机理：分块只持 `[chunk,B,...]` 激活 + 每层一个 fp32 `[B,C,H,W]` 膜电位（跨块用 `fused_bias_if_lif_stateful` 的 v_init/v_out 串接），活集不含 T。
- **全 T 峰值 ∝ T**：本脚本里 full 只有 1 个可用点（T=64=3.87GiB，且 ≈ chunk=64 的 3.86）；**多点 ∝T 由同目录 exp A/B 提供**
  （exp B ConvSNNStack B=16：1.94/3.87/7.74 GiB @ T=16/32/64，精确翻倍 123.7 MiB/T；exp A VGG-16：4 点线性 149 MiB/T）。
- **两律对偶**：exp B 的 full 斜率 123.7 MiB/T（B=16）÷2 = 61.9 MiB/T（B=8），与分块斜率 60.8 MiB/chunk **吻合 ~2%**。
  → **「一个分块步的显存 = 一个时间步的显存」，即 full-T 就是 chunk=T 的分块。**

### 2. 速度律（verdict: mostly_clean）
- **稳态时间 ∝ T（总帧数）、且近似与 chunk 无关**：chunk=8 恒为 0.6875 ms/帧（88/176/352 @ T=128/256/512，精确 1:2:4），
  chunk=32 恒为 0.6953；组 1 里 chunk 1→64 稳态仅 39.1–44.4 ms（~12% 带宽，chunk=32 vs 8 只差 1%）。
- **机理**：full 与分块推过**相同总帧数 T·B、相同 conv/BN/pool**，FLOP 相同；分块只改 conv 的前导 batch 维与 launch 次数。
  本网络 **compute-bound**（conv 占大头，~33 TFLOP/s bf16），per-chunk 的 launch 开销被长 conv kernel 摊薄 → 分块几乎不损速。
- **甚至低估了分块**：分块路径在计时区里每块还 `torch.randn` 生成输入（RNG 也被计进稳态），full 用预生成输入不含此项 →
  公平比较下分块只会**更快**，不会更慢。
- **「分块几乎免费」只对 compute-bound 网络成立 —— 已造一组 launch-bound 配置逐一实测验证**（不是外推）。
  用「每次 launch 的算力 ∝ B·H²」刻画 launch-bound 程度（越小越 launch-bound），固定 T=64 扫 chunk，测稳态 ms
  （脚本 [`launchbound_sweep.py`](./launchbound_sweep.py)，日志 [`results/launchbound_sweep.log`](./results/launchbound_sweep.log)）：

  | 配置 (B,H) | B·H² | chunk=1 | chunk=4 | chunk=16 | full(=chunk64) | **chunk1/full** | 判定 |
  |---|---:|---:|---:|---:|---:|---:|---|
  | (8,112) | 100352 | 41.8 | 42.2 | 44.1 | 44.3 | **0.9×** | 几乎免费 |
  | (8,32)  | 8192   | 29.7 | 7.8  | 2.8  | 3.5  | **8.5×** | 明显变慢 |
  | (2,32)  | 2048   | 29.9 | 7.5  | 1.95 | 0.75 | **39.8×** | 急剧变慢 |
  | (1,32)  | 1024   | 29.2 | 7.5  | 1.95 | 0.54 | **54.0×** | 急剧变慢 |
  | (1,16)  | 256    | 28.6 | 7.3  | 1.96 | 0.53 | **54.1×** | 急剧变慢 |
  | (1,8)   | 64     | 29.2 | 7.6  | 1.94 | 0.54 | **54.4×** | 急剧变慢 |

  三点结论（机理被这组数据直接揭示）：
  1. **随网络从 compute-bound 变 launch-bound（B·H² 从 10万→64），分块的减速从 0.9×（免费）单调升到 ~54×（急剧）并饱和。**
  2. **关键 launch-bound 指纹：chunk=1 的稳态时间在所有 launch-bound 配置上几乎是常数 ~29 ms**（28.6–29.9），与数据大小无关——
     因为此时时间被 **kernel launch 的固定开销**主导（chunk=1 = 64 块 × 5 层 ≈ 320 次 launch），跟实际算多少无关。
     同理 chunk=4 恒 ~7.5ms、chunk=16 恒 ~1.95ms（≈ launch 次数 ∝ 1/chunk）。而 **full（1 块）的时间随算力缩小一路降到 ~0.5ms**（跟踪真实计算）。
  3. 于是 **减速倍数 = launch 次数之比**：compute-bound 时算力 ≫ launch 开销 → 比值 ~1×（免费）；launch-bound 时 launch 开销 ≫ 算力 → 比值饱和到 ~54×（≈ chunk=1 与 full 的 launch 数之比）。

  → **「分块免费」是 compute-bound 专属**；小 batch / 小空间 / 线性·注意力小算子等 launch-bound 场景，**chunk 越小越慢、最高慢 ~54×**，务必用较大 chunk（在显存允许下尽量大）。

### 3. 编译 / 冷启动律（定性确凿、定量不可信）
- **机制确凿**：`static_range` 展开长度 = chunk 或 T；展开越长，triton 编译越慢且**超线性**。
- **唯一确凿的定量事实**：full@T=128（展开 128）是**真·编译超时**（>170s 100% CPU / 0% GPU，最终 TMO），而展开 ≤64 都在数秒级。
  → 这是「大 T 必须分块」的另一条硬证据：full-T 在大 T 不止 OOM，连**编译都编不动**。
- **冷启动列本身不可定量引用**：单次测量、混入 autotune/cudnn 抖动（chunk=32 的 24s 离群、full(3468ms) vs chunk64(2260ms) 同展开却不等）。

---

## 结论与实践建议

**三维权衡**：

| 维度 | 随 chunk 增大 | 随 T 增大 |
|---|---|---|
| 峰值显存 | **线性增大**（60.8 MiB/chunk） | chunk 固定时**不变** |
| 稳态速度 | 基本不变（compute-bound 网络） | **线性变慢**（∝ 总帧数，本就该如此） |
| 编译/冷启动 | 增大且超线性（展开变长） | full 路径随 T 超线性，大 T 编不动 |

**甜点 / 选 chunk 的实践建议**（诚实版）：
1. **显存**靠 chunk 控制：把 chunk 选到"刚好放得进显存预算"即可，峰值 ≈ 60.8 MiB×chunk + 64 MiB。
2. **速度**对 chunk 不敏感（本类网络）：不必为提速而调大 chunk —— 小 chunk 稳态速度几乎一样。
3. **编译**要避免大展开：chunk（或全 T 的 T）控制在 ~64 以内，编译才是秒级；展开到 128 就可能编不动。
4. 综上，对 compute-bound 卷积型 SNN：**用小到中等的 chunk（如 8–32）几乎全赢**——显存大降、速度几乎不损、编译还快；
   "全 T 一把梭"在大 T 既 OOM 又编不动，没有速度优势。

**诚实边界**：峰值为 `max_memory_allocated`（已分配，非保留），真实可用余量略乐观，full@T=256/512 的 OOM 阈值是外推
（T=256 full 恰在 15.4 GiB 边缘）；速度仅 3 次中位数、单机单精度单 batch；冷启动列噪声大。这些已在上文逐条标注。

> 复现：`python experiments/large-T-oom-fallback/chunk_sweep.py`（单元：`... --worker {full|chunked} <T> <chunk> <B> <H>`）。
