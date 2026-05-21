# Pass 04：TritonGPURemoveLayoutConversions

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-remove-layout-conversions` ｜ 编译流水线第 4 个 Pass

## 这个 Pass 的作用

TritonGPURemoveLayoutConversions 消除 IR 中多余的 `ttg.convert_layout` 指令。该 Pass 通过布局传播算法，将各算子所需的布局"反向传播"到其输入，使相邻操作共享同一布局，从而无需插入显式的转换指令。参考文档：[`RemoveLayoutConversions.md`](../../Passes/RemoveLayoutConversions.md)。

在 Pass 01（TritonGPUCoalesce）之后，IR 中存在大量冗余的 `ttg.convert_layout`（load 前转换到合并布局、load 后再转回通用布局），该 Pass 将它们全部消除，并重新规划全局所需的布局集合，从 8 种 blocked 布局精简为 3 种。

## IR 变化

**整体变化概要：** IR 从 210 行压缩到 186 行，布局声明从 8 种缩减为 3 种（去掉了 `#blocked3`～`#blocked7`），保留 `#blocked`（dot 累加器布局）、`#blocked1`（矩阵 A 布局，行优先加载）、`#blocked2`（矩阵 B 布局，列优先加载）。

**变换前（模块顶部布局声明，8 种）：**

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
```

**变换后（模块顶部布局声明，3 种）：**

```mlir
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
```

**变换前（循环内 A 的加载路径，有冗余 convert_layout）：**

```mlir
%a_46 = ttg.convert_layout %a_45 : tensor<16x32x!tt.ptr<f32>, #blocked1> -> tensor<16x32x!tt.ptr<f32>, #blocked5>
%a_47 = tt.load %a_46 : tensor<16x32x!tt.ptr<f32>, #blocked5>
%a_48 = ttg.convert_layout %a_47 : tensor<16x32xf32, #blocked5> -> tensor<16x32xf32, #blocked1>
...
%a_61 = ttg.convert_layout %a_48 : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>>
```

**变换后（循环内 A 的加载路径，去除冗余，直接在 `#blocked1` 下 load）：**

```mlir
%a_38 = tt.addptr %a, %xindex_37 : tensor<16x32x!tt.ptr<f32>, #blocked1>, tensor<16x32xi32, #blocked1>
%a_39 = tt.load %a_38 : tensor<16x32x!tt.ptr<f32>, #blocked1>
...
%a_47 = ttg.convert_layout %a_39 : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
```

**变换前（dot 操作，累加器经过多次转换）：**

```mlir
%7 = ttg.convert_layout %arg4 : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #blocked7>
%acc_63 = tt.dot %a_61, %b_62, %7 : ... -> tensor<16x32xf32, #blocked7>
%8 = ttg.convert_layout %acc_63 : tensor<16x32xf32, #blocked7> -> tensor<16x32xf32, #blocked1>
scf.yield %8 : tensor<16x32xf32, #blocked1>
```

**变换后（dot 操作，累加器直接使用 `#blocked`，无需往返转换）：**

```mlir
%acc_49 = tt.dot %a_47, %b_48, %arg4 : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x32xf32, #blocked>
scf.yield %acc_49 : tensor<16x32xf32, #blocked>
```

## 说明

这是全连接 kernel 布局优化的关键 Pass。经过此 Pass 后，IR 形成了清晰的三布局分工：

- `#blocked1`（`sizePerThread=[1,4]`，`threadsPerWarp=[4,8]`，行优先）：专用于矩阵 A（激活）的合并加载，每 warp 8 个线程覆盖 32 列，恰好一次事务取 128 字节。
- `#blocked2`（`sizePerThread=[4,1]`，`threadsPerWarp=[8,4]`，列优先）：专用于矩阵 B（权重）的合并加载，适应列主序访问。
- `#blocked`（`sizePerThread=[2,2]`，`threadsPerWarp=[2,16]`，行优先）：专用于 dot 累加器，对应 MMA 指令的输出分片方式。

消除冗余 layout conversion 意味着 GPU 上不再需要额外的共享内存 transpose 操作或多次 warp shuffle 来完成数据重排，直接节省了寄存器搬移开销，对 K=4096 的 128 次循环迭代（`c128_i32`）效果尤为显著。
