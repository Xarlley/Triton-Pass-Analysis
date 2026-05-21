# Pass 01：TritonGPUCoalesce

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`tritongpu-coalesce` ｜ 编译流水线第 1 个 Pass

## 这个 Pass 的作用

TritonGPUCoalesce 分析 kernel 中的每次全局内存访问，计算其最优合并（coalesce）布局，并在读/写指令前后插入 `ttg.convert_layout` 操作将张量切换到合并友好的布局，以便后续 RemoveLayoutConversions Pass 将冗余转换消除后得到最终统一的高效布局。参见 [`Coalesce.md`](../../Passes/Coalesce.md) 获取该 Pass 的详细源码分析。

## IR 变化

此 Pass 引入了第二个布局 `#blocked1`（`sizePerThread = [2]`），并在每次 `tt.load` / `tt.store` 周围插入布局转换对：

**变化前**（直接在 `#blocked` 上 load）：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
...
%tmp0_12 = tt.addptr %tmp0_11, %tmp0_10 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp0_13 = tt.load %tmp0_12 : tensor<512x!tt.ptr<f32>, #blocked>
```

**变化后**（先转换到 `#blocked1` 再 load，结果再转回 `#blocked`）：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
...
%tmp0_13 = ttg.convert_layout %tmp0_12 : tensor<512x!tt.ptr<f32>, #blocked> -> tensor<512x!tt.ptr<f32>, #blocked1>
%tmp0_14 = tt.load %tmp0_13 : tensor<512x!tt.ptr<f32>, #blocked1>
%tmp0_15 = ttg.convert_layout %tmp0_14 : tensor<512xf32, #blocked1> -> tensor<512xf32, #blocked>
```

同样的模式出现在所有 4 个 `tt.load` 以及末尾的 `tt.store` 上（store 前插入两个 convert_layout）。IR 行数从 134 增至 145，增加的均是 `ttg.convert_layout` 操作。

## 说明

本 kernel 的 BLOCK_SIZE=512，CTA 共 256 线程（8 warp × 32 thread）。在初始 `sizePerThread=[1]` 布局下，每次 load 每线程只读 1 个 float（4 字节），无法构成 128-bit 宽内存事务，无法充分利用 GPU 全局内存带宽。Coalesce Pass 发现每个 warp 对 input 的访问模式（MaxPool 4 个候选位置都是连续步长访问），将 load 切换到 `sizePerThread=[2]`（每线程读 2 个 float = 8 字节），使每个 warp 一次读取 64×8=512 字节，生成 128-bit 对齐的 `ld.global.v2.b32` 指令（后续 PTX 中可见）。

插入的 `ttg.convert_layout` 此时是逻辑占位符，实际硬件开销几乎为零——因为 `#blocked` 和 `#blocked1` 的 order 相同，差异仅 sizePerThread；下一步 RemoveLayoutConversions Pass 会将这些 convert_layout 彻底消除，令所有操作统一运行在 `#blocked1` 布局上。
