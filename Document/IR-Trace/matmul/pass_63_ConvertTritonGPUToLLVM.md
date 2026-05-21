# Pass 63：ConvertTritonGPUToLLVM

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`convert-triton-gpu-to-llvm` ｜ 编译流水线第 63 个 Pass

## 这个 Pass 的作用

ConvertTritonGPUToLLVM 是编译流水线中规模最大的 Pass，将整个 Triton GPU IR（Triton + TTG 方言，CFG 形式）全面降低（lower）为 LLVM 方言 IR，为最终生成 PTX 做最后准备。行数从 276 行激增至 7710 行（增加约 27 倍），反映了从高层抽象到完全展开的低层 LLVM 操作的剧烈膨胀。

主要变换包括：

1. **函数降低**：`tt.func` → `llvm.func`，添加 `nvvm.kernel = 1 : ui1`（标记为 CUDA kernel）和 `nvvm.reqntid = array<i32: 64>`（64 = 2 warps × 32 threads）
2. **全局共享内存变量生成**：新增 `llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>`
3. **张量 → LLVM 结构体**：所有 Triton 张量类型展开为 LLVM struct（如累加器 `tensor<16x32xf32>` → `!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>`，每个线程持有 8 个 f32 元素）
4. **共享内存寻址**：`ttg.local_alloc` → `llvm.mlir.addressof @global_smem` + `llvm.getelementptr`（A buffer 偏移 16384，B buffer 偏移 0）
5. **异步拷贝降低**：`ttg.async_copy_global_to_local` → `llvm.inline_asm "cp.async.cg.shared.global ..."`；`ttg.async_commit_group` → `nvvm.cp.async.commit.group`；`ttg.async_wait {num=6}` → `nvvm.cp.async.wait.group 6`
6. **MMA 降低**：`tt.dot` → 大量 `llvm.intr.fmuladd`（逐元素标量 FMA，对应 num_warps=2 的小规模 SIMT 矩阵乘）
7. **全局存储降低**：`tt.store` → `llvm.inline_asm "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };"`（带条件掩码的向量 4-元素写）
8. **`builtin.unrealized_conversion_cast`**：类型桥接操作，连接仍保留的少量 TTG 类型与 LLVM 类型（将在 Pass 64 消除）

## IR 变化

**变换前（TTG 方言，CFG 形式，276 行）：**

```mlir
// 函数签名（tt 方言）：
tt.func public @triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46(
    %arg_A: !tt.ptr<f32> {tt.divisibility = 16 : i32}, ...
) attributes {noinline = false} {
  ...
  cf.br ^bb1(%c0_i32, %cst, %c3_i32, %c-1_i32, %a_38, ..., %b_86 : ...)
^bb1(%acc: i32, %2: tensor<16x32xf32, #blocked>, %acc_87: i32, ...):
  %acc_97 = arith.cmpi slt, %acc, %c128_i32 : i32
  cf.cond_br %acc_97, ^bb2, ^bb3
^bb2:
  ...
  ttg.async_copy_global_to_local ...
  ttg.async_commit_group
  ttg.async_wait ... {num = 6 : i32}
  %a_118 = ttg.local_load ... -> tensor<16x32xf32, #ttg.dot_op<...>>
  %acc_121 = tt.dot %a_118, %b_120, %arg4 : ...
  ...
^bb3:
  ttg.async_wait {num = 0 : i32}
  tt.store ... : !tt.ptr<f32>
  tt.return
```

**变换后（LLVM 方言，7710 行，关键片段）：**

```mlir
// 全局共享内存占位符：
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// 函数（LLVM 方言，CUDA kernel 属性）：
llvm.func @triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46(
    %arg_A: !llvm.ptr<1> {tt.divisibility = 16 : i32, tt.pointee_type = f32}, ...
) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 64>} {

  // 累加器初始化（展开为 8 个 f32 的 struct）：
  %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
  %2 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  ...
  %10 = llvm.insertvalue %0, %9[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>

  // 共享内存基址 + offset（A buffer 偏移 16384，B buffer 偏移 0）：
  %158 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  %a_473 = llvm.mlir.constant(16384 : i32) : i32
  %a_474 = llvm.getelementptr %158[%a_473] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8

  %159 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  %b_481 = llvm.mlir.constant(0 : i32) : i32
  %b_482 = llvm.getelementptr %159[%b_481] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8

  // 异步拷贝（inline PTX）：
  %a_674 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = []
      "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;", "r,l,r"
      %a_668, %a_669, %a_673 : (!llvm.ptr<3>, !llvm.ptr<1>, i32) -> !llvm.void
  nvvm.cp.async.commit.group

  // 主循环（CFG 保持，块参数变为 i32 + struct）：
  llvm.br ^bb1(%15, %10, %141, %79, %a_685, ... : i32, !llvm.struct<(f32,f32,f32,f32,f32,f32,f32,f32)>, ...)
^bb1(%acc: i32, %160: !llvm.struct<(f32,f32,f32,f32,f32,f32,f32,f32)>, ...):
  %acc_2706 = llvm.icmp "slt" %acc, %14 : i32
  llvm.cond_br %acc_2706, ^bb2, ^bb3
^bb2:
  nvvm.cp.async.wait.group 6
  nvvm.barrier0
  // MMA（tt.dot → llvm.intr.fmuladd）：
  %acc_6088 = llvm.intr.fmuladd(%acc_5896, %acc_6024, %acc_5888) : (f32, f32, f32) -> f32
  %acc_6089 = llvm.intr.fmuladd(%acc_5897, %acc_6026, %acc_6088) : (f32, f32, f32) -> f32
  ... // 共数百条 fmuladd，对应 16x32x32 的矩阵乘展开
^bb3:
  // 全局存储（vector 4写，带条件掩码）：
  %374 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = []
      "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"
      %358, %363, %368, %373, %323, %339 : (i32, i32, i32, i32, !llvm.ptr<1>, i1) -> !llvm.void
  llvm.return
```

## 说明

**膨胀来源分析**（276 → 7710 行，增加 ~7434 行）：

- **循环前序（prologue）展开**：5 阶段软件流水线的 4 个预取迭代（K=0,1,2,3）在 lowering 时完全静态展开，每个预取包含 A、B 两路异步拷贝，每路 ~60 行（含寻址计算、mask 生成、cp.async 指令），4 个迭代共约 480 行。
- **主循环体展开**：`tt.dot %a, %b, %acc`（16×32 × 32×32 的矩阵乘）被展开为每线程持有 8 个 f32 累加值、每次读取 A（128 个 f32）和 B（128 个 f32），产生约 1024 条 `llvm.intr.fmuladd` + 相关 `llvm.extractvalue/insertvalue`，共约 3000 行。
- **常量初始化**：LLVM IR 要求所有常量显式声明，大量 `llvm.mlir.constant` 和 `llvm.mlir.undef` + `llvm.insertvalue` 序列构建初始 struct（约 500 行）。
- **内置类型转换桥接**（`builtin.unrealized_conversion_cast`）：TTG 张量类型在此 Pass 后仍有少量遗留，通过 unrealized cast 桥接（将在 Pass 64 消除）。

**`nvvm.kernel = 1 : ui1` 和 `nvvm.reqntid = array<i32: 64>`** 的含义：
- `nvvm.kernel = 1`：标记该 llvm.func 为 CUDA kernel，翻译为 PTX 的 `.visible .entry`。
- `nvvm.reqntid = [64]`：声明每个线程块（block）的线程数为 64（= 2 warp × 32 threads），翻译为 PTX `.reqntid 64, 1, 1`，使 CUDA 驱动可优化寄存器分配。

**MMA 策略**：由于 `num_warps=2`（64 threads），矩阵乘不使用 wgmma（Warp Group MMA，需要≥4 warp），而是使用标量 FMA（`llvm.intr.fmuladd`）实现，每个线程独立计算自己负责的 2×2 = 8 个输出元素的点积。
