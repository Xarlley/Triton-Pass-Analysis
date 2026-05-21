# Pass 04：TritonGPURemoveLayoutConversions

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`tritongpu-remove-layout-conversions` ｜ 编译流水线第 4 个 Pass

## 这个 Pass 的作用

TritonGPURemoveLayoutConversions 通过数据流分析，将 Coalesce Pass 插入的冗余 `ttg.convert_layout` 操作消除。其策略是：从每个 load/store 的最优布局出发，向上传播该布局直到所有生产者都使用同一布局，从而使 `convert_layout` 变为恒等操作后可以删除。参见 [`RemoveLayoutConversions.md`](../../Passes/RemoveLayoutConversions.md) 获取该 Pass 的详细分析。

## IR 变化

**变化前**（存在 `#blocked` 和 `#blocked1` 两个布局，以及大量 convert_layout）：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
...
%tmp0_13 = ttg.convert_layout %tmp0_12 : tensor<512x!tt.ptr<f32>, #blocked> -> tensor<512x!tt.ptr<f32>, #blocked1>
%tmp0_14 = tt.load %tmp0_13 : tensor<512x!tt.ptr<f32>, #blocked1>
%tmp0_15 = ttg.convert_layout %tmp0_14 : tensor<512xf32, #blocked1> -> tensor<512xf32, #blocked>
```

**变化后**（统一使用 `#blocked1`，消除了所有 convert_layout）：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
...
%tmp0_12 = tt.addptr %tmp0_11, %tmp0_10 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp0_13 = tt.load %tmp0_12 : tensor<512x!tt.ptr<f32>, #blocked>
```

整个 `module` 现在只有一个布局 `#blocked`（原来的 `#blocked1`），`#blocked1` 标识符消失，所有 `ttg.convert_layout` 操作被删除。IR 行数从 145 降至 134（减少 11 行，正好是插入的 convert_layout 对数量）。

常量声明的顺序也发生了调整：`%c512_i32` 从末尾提前到顶部：

```mlir
// after：常量顺序重排
%c512_i32 = arith.constant 512 : i32 loc(#loc1)
%tmp5 = arith.constant dense<14400> : tensor<512xi32, #blocked>
```

## 说明

消除结果表明：对于这个 MaxPool+BN+LIF pointwise kernel，所有张量操作（4 次 load、9 组逐元素计算、1 次 store）都天然适合 `sizePerThread=[2]` 的合并布局，不需要任何实际的运行期布局转换。从 GPU 执行角度看，这意味着：

1. 每个线程持有 2 个连续的 float 值（索引 `2k` 和 `2k+1`）；
2. 同一 warp 内 32 个线程的访问地址连续，形成 256 字节（64 float × 4 字节）的合并访问，PTX 将生成 `ld.global.v2.b32` 128-bit 宽事务；
3. 无需任何 shared memory 中转，适合这个纯逐元素融合 kernel 的数据流特征。

此 Pass 是 sm_120 (Blackwell) 上实现最大内存带宽利用的关键步骤之一。
