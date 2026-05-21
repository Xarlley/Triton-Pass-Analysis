# Pass 04：TritonGPURemoveLayoutConversions

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-remove-layout-conversions` ｜ 编译流水线第 04 个 Pass

## 这个 Pass 的作用

`TritonGPURemoveLayoutConversions` 消除冗余的 `ttg.convert_layout` 操作。该 Pass 对 IR 进行数据流分析，将那些源 layout 与目标 layout 可以直接对齐的转换链合并，或将 layout 统一到整个计算链的最优 layout，从而避免运行时不必要的 shared memory 数据重排开销。参见 [`RemoveLayoutConversions.md`](../../Passes/RemoveLayoutConversions.md)。本卷积 kernel 经此 Pass 后行数从 347 降至 293，减少了约 15%。

## IR 变化

整体 layout 命名空间被大幅简化：原有 9 个 `#blocked` 定义压缩为 4 个，并重新命名：

```mlir
// 变换前（9 个 layout）
#blocked  = <{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = <{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = <{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = <{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
... 以及 #blocked4..#blocked8

// 变换后（4 个 layout，重新编号）
#blocked  = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

循环体内指针计算和掩码计算中大量的 `ttg.convert_layout` 被直接删除，操作数 layout 统一：

```mlir
// 变换前（多步 convert_layout 链）
%x_base_22 = ttg.convert_layout %x_base_21 : tensor<128xi32, #blocked1> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
%x_base_23 = tt.expand_dims %x_base_22 ... -> tensor<128x1xi32, #blocked4>
%x_base_24 = ttg.convert_layout %x_base_23 : tensor<128x1xi32, #blocked4> -> tensor<128x1xi32, #blocked>

// 变换后（直接在目标 layout 上操作，消除中间转换）
%x_base_21 = arith.muli %idx_n, %x_base : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%x_base_22 = tt.expand_dims %x_base_21 {axis = 1 : i32} : tensor<128xi32, ...> -> tensor<128x1xi32, #blocked1>
%x_base_23 = tt.splat %arg_X : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
%x_base_24 = tt.addptr %x_base_23, %x_base_22 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
```

`tt.dot` 的累加器 layout 也从 `#blocked2`→`#blocked6` 的双层转换简化为直接在 `#blocked`（点积 layout）上操作：

```mlir
// 变换前
%19 = ttg.convert_layout %arg4 : tensor<128x64xf32, #blocked2> -> tensor<128x64xf32, #blocked6>
%acc_135 = tt.dot ... -> tensor<128x64xf32, #blocked6>
%20 = ttg.convert_layout %acc_135 : tensor<128x64xf32, #blocked6> -> tensor<128x64xf32, #blocked2>
scf.yield %20 : tensor<128x64xf32, #blocked2>

// 变换后（累加器直接保持在 #blocked，消除进出转换）
%acc_104 = tt.dot %matrix_x_102, %matrix_w_103, %arg4 : ... -> tensor<128x64xf32, #blocked>
scf.yield %acc_104 : tensor<128x64xf32, #blocked>
```

## 说明

这是卷积 kernel 编译中代码量减少最显著的 Pass 之一。Pass 00 分配 layout 时保守地使用了多套中间 layout 以保证正确性，Pass 01 Coalesce 又引入了额外的转换占位。Pass 04 在全图分析后发现：激活矩阵 X 的读取路径始终可以统一到 `#blocked1`（2×16 warp 分布），权重矩阵 W 的路径统一到 `#blocked2`（16×2 分布），累加器 layout 与点积 layout 直接对齐，无需在 `scf.for` 迭代边界做额外转换。这一优化避免了每次卷积 K 循环（9 次迭代）中对 shared memory 的无效 shuffle，对吞吐量有直接正向收益。
