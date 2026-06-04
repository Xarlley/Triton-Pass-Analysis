# SJ multistep LIF kernel 的 TTIR 分析：T 循环是否展开、4 个时间步是否同步处理

> 本文基于一份**真实运行**捕获的 TTIR ——
> [`Document/IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.ttir`](../IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.ttir)
> （来自 VGG16-SNN 实际推理时 SJ `multistep_lif` kernel 的一次 JIT 编译产物，
> num_warps=4, num_stages=3, sm_120, BLOCK_NCL=128, T=4），回答两个问题：
>
> 1. **TTIR 层面，`for t in range(T)` 这个时间步循环是否还存在？**
> 2. **4 个时间步的输入是否在 kernel 内"同步"处理（即并行）？**
>
> 抓取方法见 [`spikingjelly-nir-implementation.md` §6](spikingjelly-nir-implementation.md)；
> 与 LIF kernel 调用链相关的源码走读见 [`nir-call-stack-trace.md` §2.2](nir-call-stack-trace.md)。

---

## 1. 源码层面的设计选择已经决定了答案

SJ 的 `_multistep_lif_forward_kernel` 定义（[`triton_kernel/neuron_kernel/lif.py:34-66`](../../spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py#L34-L66)）：

```python
@triton.jit
def _multistep_lif_forward_kernel(
    x_seq_ptr, v_init_ptr, s_seq_ptr, h_seq_ptr, v_seq_ptr,
    tau, v_threshold, v_reset,
    T: tl.constexpr,                # ★ 编译期常量
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    ...
    for t in tl.static_range(0, T, 1):   # ★ 强制编译期展开
        ...
```

**两个关键标记**：

| 标记 | 含义 |
|---|---|
| `T: tl.constexpr` | T 是**编译期常量**，每个 (T 值) 都对应一份独立编译产物 |
| `tl.static_range(0, T, 1)` | 与 Python `range` 不同 —— `tl.static_range` **强制** Triton 在编译期把循环体复制 T 份 inline 进 SSA，不允许保留运行时 loop op |

也就是说，**还没看 TTIR 就能预判**：T=4 编译产物里不可能有 `scf.for`，只会有 4 段连续的、内联的 per-step 代码。

---

## 2. TTIR 实测：4 段 per-step 代码块清晰可见

### 2.1 顶层结构计数

| 指标 | 值 | 解读 |
|---|---:|---|
| TTIR 总行数 | 144 | |
| `scf.for` / `scf.while` / 其他 loop op | **0** | **确认 T 循环全部 unroll** |
| `tt.get_program_id` | 1 (`pid_ncl = tt.get_program_id x`) | 1D grid，按 NCL 维度切 block |
| `tt.load` | 5 | 1 × `v_init` + 4 × `x_seq[t=0..3]` |
| `tt.store` | 8 | 4 × `s_seq[t]` + 4 × `v_seq[t]` |
| `gpu.barrier` / async 指令 | 0 | 无 cross-warp 同步、无显式 async copy |

### 2.2 unroll 后的 TTIR 体结构（精简）

```mlir
tt.func @_multistep_lif_forward_kernel(%x_seq_ptr, %v_init_ptr, %s_seq_ptr, %v_seq_ptr,
                                       %tau, %v_threshold, %v_reset) {
  // ──── 头部：每个 thread block 算自己的 NCL 索引范围 ────
  %pid_ncl = tt.get_program_id x : i32
  %ncl_offset = arith.muli %pid_ncl, %c128_i32 : i32         // BLOCK_NCL=128
  %r_tau = arith.divf %1.0, %tau : f32                       // 预计算 1/tau
  ...
  %v_15 = tt.load %v_init_ptrs, %mask, %zero                 // load v_init  [+1 load]

  // ──── Step 0 (t=0) —— 用 v_15 作上一步 v ────                   行 38-56
  %x_17  = tt.load %x_seq_t0_ptrs, %mask, %zero              //          [+1 load]
  %h_18  = subf  %v_reset, %v_15
  %h_19  = addf  %h_18, %x_17
  %h_21  = mulf  %r_tau, %h_19
  %h_22  = addf  %v_15, %h_21                                // h_t = v_{t-1} + (x_t + v_reset - v_{t-1}) / tau
  %s_23  = cmpf  oge, %h_22, %v_threshold                    // spike?
  %s_24  = uitofp %s_23                                      // 0.0 / 1.0
  %v_25  = mulf  %s_24, %v_reset
  %v_26  = subf  1.0, %s_24
  %v_27  = mulf  %v_26, %h_22
  %v_28  = addf  %v_25, %v_27                                // ★ v at t=0
  tt.store %s_seq_t0_ptrs, %s_24                              //          [+1 store]
  tt.store %v_seq_t0_ptrs, %v_28                              //          [+1 store]

  // ──── Step 1 (t=1) —— 用 v_28 作上一步 v ────                   行 57-73
  %x_31  = tt.load %x_seq_t1_ptrs, %mask, %zero
  %h_32  = subf  %v_reset, %v_28                             // ★ 用上一步的 v
  %h_33  = addf  %h_32, %x_31
  %h_34  = mulf  %r_tau, %h_33
  %h_35  = addf  %v_28, %h_34
  %s_36  = cmpf  oge, %h_35, %v_threshold
  %s_37  = uitofp %s_36
  ...
  %v_41  = addf  %v_38, %v_40                                // ★ v at t=1
  tt.store %s_seq_t1_ptrs, %s_37
  tt.store %v_seq_t1_ptrs, %v_41

  // ──── Step 2 (t=2) —— 用 v_41 作上一步 v ────                   行 74-90
  %x_44  = tt.load %x_seq_t2_ptrs, %mask, %zero
  %h_45  = subf  %v_reset, %v_41                             // ★ 用上一步的 v
  ...
  %v_54  = addf  %v_51, %v_53                                // ★ v at t=2
  tt.store %s_seq_t2_ptrs, %s_50
  tt.store %v_seq_t2_ptrs, %v_54

  // ──── Step 3 (t=3) —— 用 v_54 作上一步 v ────                   行 91-107
  %x_57  = tt.load %x_seq_t3_ptrs, %mask, %zero
  %h_58  = subf  %v_reset, %v_54                             // ★ 用上一步的 v
  ...
  %v_67  = addf  %v_64, %v_66                                // ★ v at t=3
  tt.store %s_seq_t3_ptrs, %s_63
  tt.store %v_seq_t3_ptrs, %v_67

  tt.return
}
```

每个 step 大约 17 条 IR op，4 步合计 ~68 op；加上头部预计算与各种 ptr 算术构成 144 行的全部内容。

### 2.3 时间步指针偏移

`x_seq` 在 GPU 内存里 layout 为 `[T, NCL]`（行优先），步长 NCL=4096（这个 LIF 实例的特征数）。4 个时间步的 x 指针是基址 + `{0, 4096, 8192, 12288}` 偏移：

```mlir
// 在头部预计算好的 4 个常量偏移
%cst   = arith.constant dense<12288> : tensor<1x128xi64>   // = 3 * NCL  → x_seq[3]
%cst_0 = arith.constant dense<8192>  : tensor<1x128xi64>   // = 2 * NCL  → x_seq[2]
%cst_2 = arith.constant dense<4096>  : tensor<1x128xi64>   // = 1 * NCL  → x_seq[1]
                                                            //   0       → x_seq[0]

// 各 step 的 x 指针 = base + step_offset
%x_29  = arith.addi %v_10, %cst_2 : tensor<1x128xi64>      // step 1
%x_42  = arith.addi %v_10, %cst_0 : tensor<1x128xi64>      // step 2
%x_55  = arith.addi %v_10, %cst   : tensor<1x128xi64>      // step 3
```

四组 stride 常量在 head 已被预计算（line 12-17），unroll 后每步直接复用，不再有 mul 运算。

---

## 3. 答案 1：T 循环是否在 TTIR 还存在？

**否。** 已被完全展开。

证据：
- TTIR 里 `scf.for / scf.while / cf.br` 等控制流 op 一次都没出现（grep 计数 0）；
- 144 行 TTIR 里能直接看到 4 段同构的 per-step body（行 38-56 / 57-73 / 74-90 / 91-107），每段都做「load x → 计算 h → spike → 新 v → store s, v」；
- 4 段之间唯一不同的是用上一段的 `v_N` 作为输入（v_15→v_28→v_41→v_54→v_67）和指针偏移（0→4096→8192→12288）。

源码侧的两个标记 `T: tl.constexpr` + `tl.static_range(0, T, 1)` 直接保证这件事 ——
`tl.static_range` 是 Triton 专门为「**强制**编译期展开」设计的，与 `range`（运行时 loop）相对。

---

## 4. 答案 2：4 个时间步是否同步处理？

要分清两个层面的"并行"：

### 4.1 T 维（时间步之间）—— **不并行**，是顺序 SSA

TTIR 直接显示了**严格的数据依赖链**：

```
v_init (v_15)  ──→ Step 0 计算 ──→ v_28
                                      └──→ Step 1 计算 ──→ v_41
                                                              └──→ Step 2 计算 ──→ v_54
                                                                                      └──→ Step 3 计算 ──→ v_67
```

每个 Step N 的第一条算术是 `subf v_reset, v_{N-1}`，**必须等上一步的 v 计算出来才能开始**。
在 TTIR 的 SSA 图上这是一条**线性的数据流**，Triton / LLVM 后端可以做指令级并行（ILP），可以在一个 step 的尾部和下一个 step 的头部之间穿插指令，但**根本上不可能跨 step 并行**。

这是 LIF 神经元动力学的内在约束：`v_t = f(v_{t-1}, x_t)` 是**严格因果**的差分方程，没有数学等价的并行变形（不像 prefix-sum 这种可结合操作能用 Hillis-Steele 算法并行）。

### 4.2 NCL 维（batch × channel × spatial）—— **大规模并行**

`tt.get_program_id x` + `BLOCK_NCL = 128` 决定了：

- 每个 thread block（program）处理 NCL 维上 128 个 (b, c, h, w) 元素；
- 这个 LIF 实例 NCL = 4096，所以 grid 启动 `cdiv(4096, 128) = 32` 个 blocks 并行；
- block 内 4 warps × 32 threads = 128 lanes 平铺这 128 个 NCL 索引；
- 整个 LIF 在 1 次 kernel launch 内同时算完 (4096) 个 (b, c, h, w) 位置的 4 个时间步。

也就是说，**4 个时间步对所有 NCL 元素是"同时进入 kernel、同步推进"的**：当 SM 在算 step 0 时，所有 128 个 lane 都在算各自 NCL 索引下的 step 0；step 0 全部完成后才一齐进 step 1。
这是 SIMT 模型下的"warp lockstep"自然行为，**不需要显式 barrier**，TTIR 里也没有 `gpu.barrier`。

### 4.3 综合回答

| 维度 | 是否并行 | 形式 |
|---|---|---|
| **跨时间步**（T）| **不并行** | 顺序 SSA 依赖链 v_init→v_28→v_41→v_54→v_67 |
| **NCL 维内同一时间步** | **大规模并行** | grid 内 32 blocks × 128 lanes = 4096 lane 一齐算同一 step |
| **时间步与 NCL 复合视角** | **半并行** | "4 个 step 顺序执行，每 step 内 NCL 全并行" —— 即 4096 个 LIF 神经元的 4 步动力学**同步推进** |

所以用户问的「4 个时间步的输入是否同步处理」这个问法有歧义，**精确说法**是：

- **每个时间步内**：4096 个 LIF 神经元的输入 `x_seq[t]` 由 4096 lane **同时载入、并行计算**（同步处理）。
- **不同时间步之间**：必须按 t=0 → 1 → 2 → 3 顺序依次推进；t=1 的代码必须等 t=0 计算出 v_28 才能开始（顺序处理）。

这两件事各自成立，互不矛盾。Triton 把 T 维放在 SSA 里"顺序内联"是**正确选择** —— 试图把 T 也展到 lane 维度不仅没有数学依据（因果依赖），还会浪费 SM 寄存器（4× v / h 状态同时驻留）。

---

## 5. 这种结构带来的收益

相比"在 Python 端 for t 循环外面调 4 次单步 kernel"的实现方式，SJ 这版"T 内联进单 kernel"有三个具体收益：

1. **省下 3 次 kernel launch 开销**（每次 launch ~20-50 μs CPU 侧 + ~5-10 μs GPU 侧调度）；
2. **v / h 状态全程驻留 SM 寄存器**，不需要 4 次 GMEM 往返（不然每次 single-step 都得把 v 写回全局内存、下一次再读回来）；
3. **共享的预计算**（如 `1/tau`、`pid_ncl * BLOCK_NCL` 这些指针算术）只算一次而非 T 次。

代价是寄存器占用上升（编译期 4 步代码内联会让 register pressure 高于单步版本）—— 实测 sample_kernel 的 PTX 用了 `num_warps=4, num_stages=3`，autotune 选这组而非更大的 BLOCK_NCL，说明 register pressure 是 sm_120 上的实际约束。

---

## 6. 如果将来 T 增大（如 T=32）会怎样？

源码里 `tl.static_range(0, T, 1)` 强制全 unroll，所以：

- **T 小（≤ ~16）**：完全 unroll 可行，register pressure 与 kernel binary 大小都还能承受；
- **T 大（≥ ~32）**：unroll 后 IR 体长 T 倍，PTX 可能爆 register（spill 到本地内存），实际推理变慢；
- **T 极大（≥ 100）**：编译时间会显著增长（autotune × 大 IR），cubin 体积也变大。

SJ 没有做"小 T 全 unroll / 大 T 部分 unroll"的自适应策略 —— 全靠 `tl.static_range` 强 unroll。VGG16-SNN 训练惯例 T=4 在最佳工作点。如果换网络做 T=32 / T=64 实验，需要回头考虑这一假设。

---

## 7. 复现命令

```bash
# 一份代表性 LIF kernel TTIR 已固化在仓库
less Document/IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.ttir

# 验证 scf.for 不存在：
grep -c "scf\." Document/IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.ttir
#   期望: 0

# 验证 4 个时间步的指针偏移在头部预计算好：
grep "arith.constant dense" Document/IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.ttir | head -5
#   应见 0 / 4096 / 8192 / 12288 (= 0/1/2/3 × NCL=4096)

# 重新捕获一份新的 TTIR（清缓存后任何一次 VGG16-SNN 推理都会触发）：
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache
python examples/vgg16_snn/vgg16_via_nir.py
find ~/.triton/cache -name "_multistep_lif_forward_kernel.ttir" | head -1
```

---

## 8. 与其他 IR 阶段的衔接

本文聚焦在 TTIR (Triton IR)。同一份 LIF kernel 在五个 IR 层级上的完整产物（`.source / .ttir / .ttgir / .llir / .ptx / .cubin`）均在
[`Document/IR-Trace/nir_lif_kernel/sample_kernel/`](../IR-Trace/nir_lif_kernel/sample_kernel/) 已备份。

- **TTGIR**（`.ttgir`, 145 行）—— 在 TTIR 基础上加 `#ttg.blocked<{sizePerThread=[1,1], threadsPerWarp=[1,32], warpsPerCTA=[1,4]}>` layout 属性，每个 tensor 显式标出"线程内 / warp 内 / CTA 内"三级分布。**4 段 per-step 代码块的整体结构与 TTIR 完全一致**（unroll 是更早期阶段做的）。
- **LLVM IR**（`.llir`）—— TritonGPU 算子被 lowering 到 LLVM 指令；4 段 per-step 体表现为 4 个连续的 BB（basic block）链，依赖关系保留。
- **PTX / SASS**（`.ptx`, `.cubin`）—— LLVM NVPTX 后端生成的 PTX 指令，4 步的算术指令依然顺序排列，但通过 ILP 调度可能在不同 step 间穿插。
