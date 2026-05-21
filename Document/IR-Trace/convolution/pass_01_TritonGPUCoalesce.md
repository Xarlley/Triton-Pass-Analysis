# Pass 01：TritonGPUCoalesce

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-coalesce` ｜ 编译流水线第 01 个 Pass

## 这个 Pass 的作用

`TritonGPUCoalesce` 对全局内存访问的线程布局进行优化，确保 warp 内相邻线程访问连续的内存地址，从而使内存事务合并（coalesced access）。参见 [`Coalesce.md`](../../Passes/Coalesce.md)。该 Pass 通过分析 `tt.load` / `tt.store` 操作的指针计算路径，找到最优的线程分配 `order`，并在 IR 中引入新的 `#blocked` layout 定义，同时插入必要的 `ttg.convert_layout` 以桥接 layout 边界。对于本卷积 kernel，Pass 01 调整了权重矩阵 W 的加载布局和输出存储布局。

## IR 变化

**变换前**（Pass 00 产出，`#blocked6` 为点积父布局）：

```mlir
#blocked6 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

**变换后**（新增 `#blocked6`/`#blocked7`/`#blocked8`，旧 `#blocked6` 改名为 `#blocked7`）：

```mlir
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked7 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

循环体内的加载操作从单步 `tt.load` 变为带显式 layout 转换的四步：

```mlir
// 变换前
%matrix_x = tt.load %x_ptrs_81, %mask_x_112, %cst_6 : tensor<128x16x!tt.ptr<f32>, #blocked3>

// 变换后（插入三个 ttg.convert_layout 后再 load）
%matrix_x     = ttg.convert_layout %x_ptrs_81 : tensor<128x16x!tt.ptr<f32>, #blocked3> -> tensor<128x16x!tt.ptr<f32>, #blocked3>
%matrix_x_113 = ttg.convert_layout %mask_x_112 : tensor<128x16xi1, #blocked3> -> tensor<128x16xi1, #blocked3>
%matrix_x_114 = ttg.convert_layout %cst_6 : tensor<128x16xf32, #blocked3> -> tensor<128x16xf32, #blocked3>
%matrix_x_115 = tt.load %matrix_x, %matrix_x_113, %matrix_x_114 : tensor<128x16x!tt.ptr<f32>, #blocked3>
```

## 说明

Coalesce Pass 的核心是保证全局内存访问的合并。对于 VGG16 第 1 卷积的 128×16 激活子矩阵（`matrix_x`），线程需要按列连续访问以匹配 DRAM 的 128-byte 事务粒度；对于 16×64 权重矩阵（`matrix_w`），Pass 引入新的 `#blocked6`（`order = [0, 1]`，列优先），使 warp 内相邻线程沿 K 维（列方向）连续取数，提升 L2 命中率。

本阶段部分 `ttg.convert_layout` 看似是 layout→layout 自身（即 `#blocked3 -> #blocked3`），这是 Coalesce Pass 生成"占位"转换以标记内存访问点、供后续 Pass（`RemoveLayoutConversions`）决策消除的临时做法。这 30 个新增转换在 Pass 04 中会被大量消除，最终 IR 行数从 347 降至 293。
