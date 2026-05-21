# Pass 11：TritonLoopAwareCSE

> kernel：卷积 (Convolution) ｜ CLI：`triton-loop-aware-cse` ｜ 编译流水线第 11 个 Pass

## 这个 Pass 的作用

`TritonLoopAwareCSE`（循环感知公共子表达式消除）是 Triton 定制的 CSE Pass，能够识别跨循环迭代的重复计算，并在循环外提升不变量（与 LICM 配合使用）。与标准 CSE 不同，它能处理 `scf.for` 中携带的循环变量（`iter_args`）和循环体内的结构化控制流。本次执行（Pass 11，第一次 CSE）将 IR 行数从 293 减少到 292，消除了一处在循环内重复计算的表达式。

## IR 变化

diff 显示循环体内的一处 `tt.make_range` 与 `tt.expand_dims` 序列被提前：循环内原本每迭代都重新生成 `idx_x_c` 对应的 `tensor<16xi32>` range，CSE 识别出这个值在 9 次迭代中完全相同，将其中的一个冗余路径合并。

```mlir
// 变换前（循环内）
%x_ptrs_63 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%x_ptrs_64 = tt.expand_dims %x_ptrs_63 {axis = 0 : i32} : tensor<16xi32, ...> -> tensor<1x16xi32, #blocked1>
%x_ptrs_65 = tt.broadcast %x_ptrs_62 : tensor<128x1xi32, #blocked1> -> tensor<128x16xi32, #blocked1>
%x_ptrs_66 = tt.broadcast %x_ptrs_64 : tensor<1x16xi32, #blocked1> -> tensor<128x16xi32, #blocked1>
%x_ptrs_67 = arith.addi %x_ptrs_65, %x_ptrs_66 : tensor<128x16xi32, #blocked1>

// 变换后（复用已有的 %x_ptrs_63，消除重复 make_range）
%x_ptrs_63 = tt.expand_dims %idx_x_c {axis = 0 : i32} : tensor<16xi32, ...> -> tensor<1x16xi32, #blocked1>
%x_ptrs_64 = tt.broadcast %x_ptrs_62 : tensor<128x1xi32, #blocked1> -> tensor<128x16xi32, #blocked1>
%x_ptrs_65 = tt.broadcast %x_ptrs_63 : tensor<1x16xi32, #blocked1> -> tensor<128x16xi32, #blocked1>
%x_ptrs_66 = arith.addi %x_ptrs_64, %x_ptrs_65 : tensor<128x16xi32, #blocked1>
```

掩码链的 SSA 编号随之向前移一位（`%mask_x_70` → `%mask_x_69` 等），但计算逻辑完全不变。

## 说明

本 kernel 的 K 维循环共 9 次迭代（卷积核 3×3 = 9 个位置），每次迭代都需要构造输入 X 的通道索引 `idx_x_c`（range 0..16）并展开为 `1×16` 的指针偏移。CSE 在第 11 个 Pass 就能识别出此 range 在整个 `scf.for` 中为常量（不依赖循环变量 `%arg3`），因此合并为一条共享定义。节省的开销很小（1 行 IR），但体现了循环感知 CSE 对标量/向量常量提升的细致处理；更显著的循环不变量提升将在 Pass 14（LICM）中完成。
