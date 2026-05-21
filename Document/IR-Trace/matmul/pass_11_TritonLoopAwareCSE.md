# Pass 11：TritonLoopAwareCSE

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`triton-loop-aware-cse` ｜ 编译流水线第 11 个 Pass

## 这个 Pass 的作用

TritonLoopAwareCSE 是 Triton 特化的公共子表达式消除（Common Subexpression Elimination）Pass。与标准 CSE 不同，它能正确处理 Triton 循环中的 `scf.for` 迭代变量边界，避免在循环体内错误地将与归纳变量相关的表达式视为公共子表达式。该 Pass 识别并消除在不同上下文中计算结果完全相同的冗余指令。

本次变换将 IR 从 186 行减少到 185 行（消除了 1 条冗余指令）。

## IR 变化

**关键变化：** 循环内对 `tt.make_range` 的重复调用被消除，直接复用循环外已有的 `%rn_14`（`#ttg.slice<{dim = 0, parent = #blocked1}>`）结果。

**变换前（循环内有重复的 `tt.make_range`）：**

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x32xf32, #blocked>) {
  %a_k_idx_vals = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
  %a_k_idx_vals_26 = tt.expand_dims %a_k_idx_vals {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
  %a_k_idx_vals_27 = arith.muli %arg3, %c32_i32 : i32
  %a_k_idx_vals_28 = tt.splat %a_k_idx_vals_27 : i32 -> tensor<1x32xi32, #blocked1>
  %a_k_idx_vals_29 = arith.addi %a_k_idx_vals_26, %a_k_idx_vals_28 : tensor<1x32xi32, #blocked1>
```

**变换后（循环内去掉重复的 `tt.make_range`，直接使用外部的 `%rn_14`）：**

```mlir
%rn_14 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
...
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x32xf32, #blocked>) {
  %a_k_idx_vals = tt.expand_dims %rn_14 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
  %a_k_idx_vals_26 = arith.muli %arg3, %c32_i32 : i32
  %a_k_idx_vals_27 = tt.splat %a_k_idx_vals_26 : i32 -> tensor<1x32xi32, #blocked1>
  %a_k_idx_vals_28 = arith.addi %a_k_idx_vals, %a_k_idx_vals_27 : tensor<1x32xi32, #blocked1>
```

## 说明

`tt.make_range {end = 32 : i32, start = 0 : i32}` 是一个常量计算，其结果不依赖循环归纳变量 `%arg3`。在变换前，该指令在循环体内每次迭代都重新计算；变换后，它被提升到循环外（通过 CSE 识别为已有的 `%rn_14` 结果），循环体内直接引用 `%rn_14`，跳过 `tt.make_range` 这一步，直接执行 `tt.expand_dims`。

对于这个 K=4096 步长为 32 的矩阵乘法循环（128 次迭代），消除这条循环不变的 `tt.make_range` 在 GPU 上节省了 128 次张量初始化开销。虽然这只是微小的优化，但它是为后续 Pass 14（TritonLoopInvariantCodeMotion）做铺垫——LICM 将把更多循环不变量（包括 `tt.expand_dims`、地址基址计算等）整体提升到循环外。
