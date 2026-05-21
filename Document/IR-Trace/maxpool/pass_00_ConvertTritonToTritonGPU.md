# Pass 00：ConvertTritonToTritonGPU

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`convert-triton-to-tritongpu` ｜ 编译流水线第 0 个 Pass

## 这个 Pass 的作用

ConvertTritonToTritonGPU 是整个 GPU 后端编译流水线的起点，负责将方言无关的 Triton IR（`tt.*` 操作、裸张量类型）转换为 Triton GPU IR（`ttg.*`），为后续所有 GPU 感知优化奠定基础。其核心工作是：根据编译选项（`num_warps=8`，`threads_per_warp=32`，`num_ctas=1`）确定一个初始 blocked layout，并将所有张量类型打上 `#blocked` 布局编码；同时在 `module` 属性上注入硬件目标信息（`ttg.num-warps`、`ttg.target`、`ttg.threads-per-warp`、`ttg.num-ctas`）。

## IR 变化

**变化前**（Triton IR，无布局）：

```mlir
module {
  tt.func public @triton_poi_fused_...() attributes {noinline = false} {
    %tmp5 = arith.constant dense<14400> : tensor<512xi32>
    %xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %tmp0_13 = tt.load %tmp0_12 : tensor<512x!tt.ptr<f32>>
    tt.store %1, %tmp6 : tensor<512x!tt.ptr<f32>>
  }
}
```

**变化后**（Triton GPU IR，带 `#blocked` 布局）：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_poi_fused_...() attributes {noinline = false} {
    %tmp5 = arith.constant dense<14400> : tensor<512xi32, #blocked>
    %xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %tmp0_13 = tt.load %tmp0_12 : tensor<512x!tt.ptr<f32>, #blocked>
    tt.store %1, %tmp6 : tensor<512x!tt.ptr<f32>, #blocked>
  }
}
```

关键变化：每个张量类型从 `tensor<512xi32>` 变为 `tensor<512xi32, #blocked>`；module 新增 `ttg.num-ctas`、`ttg.num-warps`、`ttg.target`、`ttg.threads-per-warp` 四个属性；所有函数操作体内的逻辑完全不变，仅类型标注增加布局后缀。

## 说明

初始布局 `#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>` 意味着：每个线程负责 1 个元素，warp 内 32 个线程，CTA 内 8 个 warp，总计 256 线程处理 256 个元素；而本 kernel 的 BLOCK_SIZE=512，因此实际上每线程处理 2 个元素（这个差距将由下一步 Coalesce Pass 修正为 `sizePerThread=[2]`）。

对于本 MaxPool+BN+LIF kernel 而言，这一步的意义在于：4 次全局内存读（MaxPool 的 2×2 窗口 4 个候选值）和 1 次写操作都被注册为 `#blocked` layout 下的合法张量操作，后续 Pass 才能基于此分析内存访问模式并做合并优化。`ttg.target = "cuda:120"` 标记了 Blackwell sm_120 目标，决定了后续 NVWS（Nvidia Warp Specialization）和 TMA 等 Blackwell 专属 Pass 的行为。
