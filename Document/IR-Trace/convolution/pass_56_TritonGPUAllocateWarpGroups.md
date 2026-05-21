# Pass 56：TritonGPUAllocateWarpGroups

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-allocate-warp-groups` ｜ 编译流水线第 56 个 Pass

## 这个 Pass 的作用

`TritonGPUAllocateWarpGroups`（Warp Group 分配）为 kernel 在 module 级属性中记录 Warp Group 的配置信息，为后续的 Warp Specialization lowering 做准备。对于本卷积 kernel，此 Pass 在 `module` 的属性列表中追加了 `"ttg.total-num-warps" = 4` 属性，表明整个 CTA 共有 4 个 warp。IR 行数保持 403 不变，仅修改了 module 的属性声明。

## IR 变化

**在 `module` 属性中追加 `ttg.total-num-warps`**：

```mlir
// 变换前
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {

// 变换后（新增 total-num-warps 属性）
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
```

## 说明

`ttg.num-warps` 和 `ttg.total-num-warps` 的区别在于：在支持 Warp Specialization 的 kernel 中，每个 Warp Group 可能只包含部分 warp（例如，Producer Warp Group 和 Consumer Warp Group 各占一半）；`ttg.num-warps` 可以指代每组的 warp 数，而 `ttg.total-num-warps` 则是整个 CTA 的 warp 总数。对于本卷积 kernel，由于目标是 Blackwell sm_120（`cuda:120`），编译器在前序的 Warp Specialization passes（Pass 20~28）中曾为 Producer/Consumer 分配不同的 warp cluster，但最终的 warp 数量仍为 4（`num_warps=4` 的编译参数）。`TritonGPUAllocateWarpGroups` 将这一总数显式写入 module 属性，使后续的 `ConvertWarpSpecializeToLLVM`（Pass 66）能够查询到正确的 warp 总数来生成正确的 PTX `%nthreads` 计算（4 × 32 = 128 threads per CTA）。
