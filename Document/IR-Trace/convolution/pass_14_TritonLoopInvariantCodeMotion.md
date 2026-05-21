# Pass 14：TritonLoopInvariantCodeMotion

> kernel：卷积 (Convolution) ｜ CLI：`triton-licm` ｜ 编译流水线第 14 个 Pass

## 这个 Pass 的作用

`TritonLoopInvariantCodeMotion`（循环不变代码外提，LICM）将 `scf.for` 循环体内不依赖循环变量的计算提升到循环之前，减少每次迭代的重复计算量。Triton 定制版本的 LICM 能正确处理带有 `iter_args` 的结构化 `scf.for`，并识别张量指针计算、掩码构造等 Triton 特有的模式。本次 IR 行数未变（292→292），但循环内部结构发生了显著重组：18 条操作从循环内移到了循环外。

## IR 变化

LICM 将与 `idx_x_c`（通道索引 range 0..16）、基础指针 `%x_base_36`/`%w_base_40`、权重掩码等相关的不变量全部外提：

```mlir
// 变换后（这些操作出现在 scf.for 之前，即被外提）
%idx_x_c = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%x_ptrs   = tt.expand_dims %idx_x_c {axis = 0 : i32} : tensor<16xi32, ...> -> tensor<1x16xi32, #blocked1>
%x_ptrs_41 = tt.broadcast %x_ptrs : tensor<1x16xi32, #blocked1> -> tensor<128x16xi32, #blocked1>
%x_ptrs_42 = tt.broadcast %x_base_36 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x16x!tt.ptr<f32>, #blocked1>
%mask_x    = arith.cmpi slt, %idx_n, %cst_6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%mask_x_43 = tt.expand_dims %mask_x {axis = 1 : i32} : tensor<128xi1, ...> -> tensor<128x1xi1, #blocked1>
%mask_x_44 = arith.cmpi slt, %idx_x_c, %cst_9 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%mask_x_45 = tt.expand_dims %mask_x_44 {axis = 0 : i32} : tensor<16xi1, ...> -> tensor<1x16xi1, #blocked1>
%mask_x_46 = tt.broadcast %mask_x_45 : tensor<1x16xi1, #blocked1> -> tensor<128x16xi1, #blocked1>
%w_ptrs    = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%w_ptrs_47 = tt.expand_dims %w_ptrs {axis = 1 : i32} : tensor<16xi32, ...> -> tensor<16x1xi32, #blocked2>
%w_ptrs_48 = tt.broadcast %w_base_40 : tensor<1x64x!tt.ptr<f32>, #blocked2> -> tensor<16x64x!tt.ptr<f32>, #blocked2>
%mask_w    = arith.cmpi slt, %w_ptrs_47, %cst_10 : tensor<16x1xi32, #blocked2>
%mask_w_49 = tt.expand_dims %idx_y_c_31 {axis = 0 : i32} : tensor<64xi32, ...> -> tensor<1x64xi32, #blocked2>
%mask_w_50 = arith.cmpi slt, %mask_w_49, %cst_11 : tensor<1x64xi32, #blocked2>
%mask_w_51 = tt.broadcast %mask_w : tensor<16x1xi1, #blocked2> -> tensor<16x64xi1, #blocked2>
%mask_w_52 = tt.broadcast %mask_w_50 : tensor<1x64xi1, #blocked2> -> tensor<16x64xi1, #blocked2>
%mask_w_53 = arith.andi %mask_w_51, %mask_w_52 : tensor<16x64xi1, #blocked2>
```

循环内部权重加载简化为：直接复用外提的 `%w_ptrs_48`（基础指针广播）和 `%mask_w_53`（完整权重掩码），每次迭代仅计算与 `i`、`j` 相关的偏移：

```mlir
// 变换后的循环内（仅保留依赖循环变量的计算）
%w_ptrs_94 = arith.muli %i, %c9_i32 : i32
%w_ptrs_95 = tt.splat %w_ptrs_94 : i32 -> tensor<16x1xi32, #blocked2>
%w_ptrs_96 = arith.addi %w_ptrs_47, %w_ptrs_95 : tensor<16x1xi32, #blocked2>
%w_ptrs_97 = arith.muli %j, %c3_i32 : i32
%w_ptrs_98 = tt.splat %w_ptrs_97 : i32 -> tensor<16x1xi32, #blocked2>
%w_ptrs_99 = arith.addi %w_ptrs_96, %w_ptrs_98 : tensor<16x1xi32, #blocked2>
%w_ptrs_100 = tt.broadcast %w_ptrs_99 : tensor<16x1xi32, #blocked2> -> tensor<16x64xi32, #blocked2>
%w_ptrs_101 = tt.addptr %w_ptrs_48, %w_ptrs_100 : tensor<16x64x!tt.ptr<f32>, #blocked2>, tensor<16x64xi32, #blocked2>
%matrix_w  = tt.load %w_ptrs_101, %mask_w_53, %cst_1 : tensor<16x64x!tt.ptr<f32>, #blocked2>
```

## 说明

本 kernel 的 K 循环（9 次迭代）中，输入通道维度的 range（`idx_x_c`，0..15）、通道数量掩码（`mask_x_c < 3`，即 RGB 3 通道掩码）以及权重矩阵的基础指针广播（`w_ptrs_48`）都不依赖循环变量 `%arg3`，LICM 将它们全部外提。对于权重掩码 `mask_w_53`，其值为 `(k_row < 3) AND (out_channel < 64)`，两个条件均与 `i`/`j` 无关，因此整个掩码张量可提升为常量——这意味着 9 次迭代共享同一份掩码，避免了 9×128=1152 次冗余的比较操作。外提后循环体更紧凑，为后续的 `TritonGPUAssignLatencies` 和流水线调度奠定了更清晰的数据依赖图。
