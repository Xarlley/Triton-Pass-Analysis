# Pass 01：TritonGPUCoalesce

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`tritongpu-coalesce` ｜ 编译流水线第 1 个 Pass

## 这个 Pass 的作用

TritonGPUCoalesce 分析 kernel 中每条 `tt.load` / `tt.store` 指令的访存模式，推导出能使相邻线程访问连续内存地址（coalesced access）的最优 blocked 布局，然后在 load/store 周围插入 `ttg.convert_layout` 转换使访存指令在该最优布局下执行，计算结果再转换回全局布局。其目标是让全局内存访问尽可能合并为一次宽内存事务，从而最大化内存带宽利用率。参见 [`Coalesce.md`](../../Passes/Coalesce.md)。

## IR 变化

本 Pass 引入了两个新的布局定义，专门用于 load 指令：

```mlir
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
```

**变换前（load 直接在 `#blocked` 布局下执行）：**

```mlir
%tmp0_24 = tt.load %tmp0_22, %tmp0_23 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>, #blocked>
```

**变换后（load 在 `#blocked5` 布局下执行，周围包裹 convert_layout）：**

```mlir
%tmp0_24 = ttg.convert_layout %tmp0_22 : tensor<16x64x!tt.ptr<f32>, #blocked> -> tensor<16x64x!tt.ptr<f32>, #blocked5>
%tmp0_25 = ttg.convert_layout %tmp0_23 : tensor<16x64xi1, #blocked> -> tensor<16x64xi1, #blocked5>
%tmp0_26 = tt.load %tmp0_24, %tmp0_25 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>, #blocked5>
%tmp0_27 = ttg.convert_layout %tmp0_26 : tensor<16x64xf32, #blocked5> -> tensor<16x64xf32, #blocked>
```

同样的模式应用于全部 4 条 load 指令（`tmp0`、`tmp11`、`tmp22`、`tmp33`）以及末尾的 2 条 store 指令（也使用 `#blocked5`）。

## 说明

`#blocked5 = {sizePerThread=[1,4], threadsPerWarp=[2,16], warpsPerCTA=[4,1], order=[1,0]}` 是 BN+LIF kernel 中访存 coalescing 的关键布局：

- **order=[1,0]** 表示 x 维度（列方向，对应 64 个通道）是最内层连续维度，线程沿 x 方向连续排列。`threadsPerWarp=[2,16]` 意味着每个 warp 内有 16 个线程覆盖 x 维度，每个线程处理 4 个连续元素（`sizePerThread=[1,4]`），共覆盖 64 列，恰好是一行的宽度。
- 这意味着一个 warp 内的线程访问的地址是连续的，4 个线程 × 4 元素 = 一次 128 字节的对齐访问，满足 Blackwell sm_120 全局内存合并要求。
- 对比原始 `#blocked = {threadsPerWarp=[1,32], warpsPerCTA=[2,2]}` 布局：x 方向 32 个线程中只有 1/4 参与 x 方向连续访问，coalescing 效率较低。
- `#blocked5` 之外保留了 `#blocked6`（用于 store 的 out_ptr0 路径），表明 Pass 为读取路径（BN 输入缓冲区）和写入路径（out_ptr0）分别推导了最优布局。
- 整体代价：插入 12 个额外 `ttg.convert_layout` 操作（每次 load/store 各 3 个），但换来的是真正 coalesced 的 DRAM 访问，对 VGG16-SNN 推理中大量的特征图数据搬运至关重要。
