# Pass 14：TritonLoopInvariantCodeMotion

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`triton-licm` ｜ 编译流水线第 14 个 Pass

## 这个 Pass 的作用

TritonLoopInvariantCodeMotion 实现循环不变量代码外提（Loop-Invariant Code Motion，LICM）。它遍历 `scf.for` 循环体，识别那些计算结果不依赖循环归纳变量（`%arg3`）的指令，并将它们移到循环外的 `scf.for` 之前执行。这样每次迭代就不必重复计算这些不变量，减少了循环体的指令数量，从而为 GPU 上的流水线（Pipeline）调度提供更简洁的循环核心。

## IR 变化

**变换前（循环体内含大量循环不变计算）：**

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x32xf32, #blocked>) {
  %a_k_idx_vals = tt.expand_dims %rn_14 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
  %a_k_idx_vals_26 = arith.muli %arg3, %c32_i32 : i32
  %a_k_idx_vals_27 = tt.splat %a_k_idx_vals_26 : i32 -> tensor<1x32xi32, #blocked1>
  %a_k_idx_vals_28 = arith.addi %a_k_idx_vals, %a_k_idx_vals_27 : tensor<1x32xi32, #blocked1>
  %b_k_idx_vals = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
  %b_k_idx_vals_29 = tt.expand_dims %b_k_idx_vals {axis = 1 : i32} : ... -> tensor<32x1xi32, #blocked2>
  %b_k_idx_vals_30 = tt.splat %a_k_idx_vals_26 : i32 -> tensor<32x1xi32, #blocked2>
  %b_k_idx_vals_31 = arith.addi %b_k_idx_vals_29, %b_k_idx_vals_30 : tensor<32x1xi32, #blocked2>
  %idx_m_32 = tt.expand_dims %offs_a_m {axis = 1 : i32} : ... -> tensor<16x1xi32, #blocked1>
  %xindex_33 = arith.muli %idx_m_32, %cst_2 : tensor<16x1xi32, #blocked1>
  %xindex_34 = tt.broadcast %a_k_idx_vals_28 : tensor<1x32xi32, #blocked1> -> tensor<16x32xi32, #blocked1>
  %xindex_35 = tt.broadcast %xindex_33 : tensor<16x1xi32, #blocked1> -> tensor<16x32xi32, #blocked1>
  %xindex_36 = arith.addi %xindex_34, %xindex_35 : tensor<16x32xi32, #blocked1>
  %a = tt.splat %arg_A : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked1>
  %a_37 = tt.addptr %a, %xindex_36 : ...
  %a_38 = tt.load %a_37 : tensor<16x32x!tt.ptr<f32>, #blocked1>
  %idx_n_39 = tt.expand_dims %offs_b_n {axis = 0 : i32} : ... -> tensor<1x32xi32, #blocked2>
  %b = arith.muli %idx_n_39, %cst_3 : tensor<1x32xi32, #blocked2>
  %b_40 = tt.broadcast %b_k_idx_vals_31 : tensor<32x1xi32, #blocked2> -> tensor<32x32xi32, #blocked2>
  %b_41 = tt.broadcast %b : tensor<1x32xi32, #blocked2> -> tensor<32x32xi32, #blocked2>
  %b_42 = arith.addi %b_40, %b_41 : tensor<32x32xi32, #blocked2>
  %b_43 = tt.splat %arg_B : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked2>
  %b_44 = tt.addptr %b_43, %b_42 : ...
  %b_45 = tt.load %b_44 : tensor<32x32x!tt.ptr<f32>, #blocked2>
  ...
```

**变换后（循环不变量提升到循环外，循环体大幅精简）：**

```mlir
  %a_k_idx_vals = tt.expand_dims %rn_14 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
  %b_k_idx_vals = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
  %b_k_idx_vals_19 = tt.expand_dims %b_k_idx_vals {axis = 1 : i32} : ... -> tensor<32x1xi32, #blocked2>
  %idx_m = tt.expand_dims %offs_a_m {axis = 1 : i32} : ... -> tensor<16x1xi32, #blocked1>
  %xindex = arith.muli %idx_m, %cst_2 : tensor<16x1xi32, #blocked1>
  %xindex_20 = tt.broadcast %xindex : tensor<16x1xi32, #blocked1> -> tensor<16x32xi32, #blocked1>
  %a = tt.splat %arg_A : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked1>
  %idx_n = tt.expand_dims %offs_b_n {axis = 0 : i32} : ... -> tensor<1x32xi32, #blocked2>
  %b = arith.muli %idx_n, %cst_3 : tensor<1x32xi32, #blocked2>
  %b_21 = tt.broadcast %b : tensor<1x32xi32, #blocked2> -> tensor<32x32xi32, #blocked2>
  %b_22 = tt.splat %arg_B : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked2>
  %acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x32xf32, #blocked>) {
    %a_k_idx_vals_33 = arith.muli %arg3, %c32_i32 : i32
    %a_k_idx_vals_34 = tt.splat %a_k_idx_vals_33 : i32 -> tensor<1x32xi32, #blocked1>
    %a_k_idx_vals_35 = arith.addi %a_k_idx_vals, %a_k_idx_vals_34 : tensor<1x32xi32, #blocked1>
    %b_k_idx_vals_36 = tt.splat %a_k_idx_vals_33 : i32 -> tensor<32x1xi32, #blocked2>
    %b_k_idx_vals_37 = arith.addi %b_k_idx_vals_19, %b_k_idx_vals_36 : tensor<32x1xi32, #blocked2>
    %xindex_38 = tt.broadcast %a_k_idx_vals_35 : tensor<1x32xi32, #blocked1> -> tensor<16x32xi32, #blocked1>
    %xindex_39 = arith.addi %xindex_38, %xindex_20 : tensor<16x32xi32, #blocked1>
    %a_40 = tt.addptr %a, %xindex_39 : ...
    %a_41 = tt.load %a_40 : tensor<16x32x!tt.ptr<f32>, #blocked1>
    ...
```

## 说明

LICM 将 11 条循环不变指令从 128 次迭代的循环体中提升出去，包括：`tt.expand_dims`（A、B 的 K 维索引构造）、`tt.make_range`（B 的 K 维范围）、`idx_m` 的行偏移计算（`*4096`）、`tt.broadcast`（行偏移广播）、`tt.splat`（基址指针广播）等。

循环体内现在只保留真正依赖 `%arg3` 的计算：`%arg3 * 32`（K 偏移标量）、两个 `tt.splat`（将标量 K 偏移广播为向量）、两个 `arith.addi`（加 K 偏移）以及实际的地址和加载指令。循环体从约 20 条指令缩减为约 10 条，为后续的 TritonGPUPipeline Pass（软件流水线）提供了更规整的循环结构，使流水线分析可以更准确地划分 prologue/epilogue。
