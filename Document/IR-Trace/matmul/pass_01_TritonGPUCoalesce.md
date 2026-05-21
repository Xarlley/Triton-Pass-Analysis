# Pass 01：TritonGPUCoalesce

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-coalesce` ｜ 编译流水线第 1 个 Pass

## 这个 Pass 的作用

TritonGPUCoalesce 的目标是确保全局内存访问的合并（coalescing）。它分析每个 `tt.load` / `tt.store` 的张量布局，若当前布局导致 warp 内的线程访问不连续内存地址（即不合并），则重新分配一个更优的合并布局，并在 load/store 前后插入 `ttg.convert_layout` 以完成布局切换。参考文档：[`Coalesce.md`](../../Passes/Coalesce.md)。

该 Pass 在 `TritonGPUCoalesceUtils` 的基础上工作，核心逻辑是：对每个 load/store 指令，计算在当前布局下 warp 内相邻线程访问的内存步长，若不连续则找出令步长最小的轴顺序并重构布局。

## IR 变化

**关键变化：** 新增了两个合并布局 `#blocked5`（用于矩阵 A 加载）和 `#blocked6`（用于矩阵 B 加载），并在 load 前后插入了 `ttg.convert_layout`；输出存储也同样处理。

**变换前（A 的加载路径，布局为 `#blocked1`）：**

```mlir
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
...
%a_45 = tt.addptr %a, %xindex_44 : tensor<16x32x!tt.ptr<f32>, #blocked1>, tensor<16x32xi32, #blocked1>
%a_46 = tt.load %a_45 : tensor<16x32x!tt.ptr<f32>, #blocked1>
```

**变换后（A 的加载路径，在合并布局 `#blocked5` 下执行 load）：**

```mlir
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
...
%a_45 = tt.addptr %a, %xindex_44 : tensor<16x32x!tt.ptr<f32>, #blocked1>, tensor<16x32xi32, #blocked1>
%a_46 = ttg.convert_layout %a_45 : tensor<16x32x!tt.ptr<f32>, #blocked1> -> tensor<16x32x!tt.ptr<f32>, #blocked5>
%a_47 = tt.load %a_46 : tensor<16x32x!tt.ptr<f32>, #blocked5>
%a_48 = ttg.convert_layout %a_47 : tensor<16x32xf32, #blocked5> -> tensor<16x32xf32, #blocked1>
```

**变换前（B 的加载路径，布局为 `#blocked1`）：**

```mlir
%b_57 = tt.addptr %b_56, %b_55 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
%b_58 = tt.load %b_57 : tensor<32x32x!tt.ptr<f32>, #blocked1>
```

**变换后（B 的加载路径，在合并布局 `#blocked6` 下执行 load）：**

```mlir
#blocked6 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
...
%b_57 = tt.addptr %b_56, %b_55 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
%b_58 = ttg.convert_layout %b_57 : tensor<32x32x!tt.ptr<f32>, #blocked1> -> tensor<32x32x!tt.ptr<f32>, #blocked6>
%b_59 = tt.load %b_58 : tensor<32x32x!tt.ptr<f32>, #blocked6>
%b_60 = ttg.convert_layout %b_59 : tensor<32x32xf32, #blocked6> -> tensor<32x32xf32, #blocked1>
```

**输出存储也类似处理：**

```mlir
%4 = ttg.convert_layout %3 : tensor<16x32x!tt.ptr<f32>, #blocked1> -> tensor<16x32x!tt.ptr<f32>, #blocked5>
%5 = ttg.convert_layout %acc : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #blocked5>
%6 = ttg.convert_layout %mask_23 : tensor<16x32xi1, #blocked1> -> tensor<16x32xi1, #blocked5>
tt.store %4, %5, %6 : tensor<16x32x!tt.ptr<f32>, #blocked5>
```

## 说明

对于这个 VGG16 SNN 全连接 kernel，矩阵 A（激活，形状 `1×4096`，按行访问）和矩阵 B（权重，形状 `4096×4096`，按列访问）的内存访问模式不同。

- 矩阵 A 的新布局 `#blocked5`（`sizePerThread=[1,4]`，`threadsPerWarp=[4,8]`，`order=[1,0]` 行优先）：warp 内 8 个线程连续覆盖列维度（每线程 4 列），总计 32 个连续元素，与 GPU 全局内存的 128 字节事务对齐，实现完全合并。
- 矩阵 B 的新布局 `#blocked6`（`sizePerThread=[4,1]`，`threadsPerWarp=[8,4]`，`order=[0,1]` 列优先）：warp 内 4 个线程连续覆盖行维度（每线程 4 行），适合权重矩阵的列访问模式。

此时 IR 行数从 201 增加到 210，因为新增了合并布局并插入了额外的 `ttg.convert_layout` 指令。这些转换目前是冗余的（load 前转换、load 后再转回 `#blocked1`），将由后续的 RemoveLayoutConversions Pass 消除。
