# Pass 56：TritonGPUAllocateWarpGroups

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`tritongpu-allocate-warp-groups` ｜ 编译流水线第 56 个 Pass

## 这个 Pass 的作用

TritonGPUAllocateWarpGroups 负责为 kernel 分配 warp group 资源，并在 module 属性中记录总 warp 数（`ttg.total-num-warps`）。对于普通 kernel（无 warp 专化分组），此 Pass 将所有 warp 分配为单一的默认 group，并在 module 属性中写入 `"ttg.total-num-warps"` 以供后续 lowering 使用。

## IR 变化

此 Pass 仅修改了 **module 级别的属性**，函数体内容完全不变。IR 行数保持 134 行不变。

**变化前（before）**：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
```

**变化后（after）**：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
```

新增属性：`"ttg.total-num-warps" = 8 : i32`。

## 说明

`ttg.total-num-warps` 与 `ttg.num-warps` 的区别：

- `ttg.num-warps`：每个 CTA 中用于计算的 warp 数（本 kernel 为 8）；
- `ttg.total-num-warps`：CTA 中所有 warp 的总数（含专化 warp）。

对于有 warp 专化的 kernel（如含 WGMMA 的 matmul），`total-num-warps` 可能大于 `num-warps`（例如 num-warps=4 计算 warp + 4 专化 warp = total 8）。本 MaxPool+BN+LIF kernel 无 warp 专化，因此 `total-num-warps = num-warps = 8`。

此属性对 NVPTX 后端生成 PTX 的 `.reqntid` 指令至关重要：`nvvm.reqntid = array<i32: 256>`（即 `total-num-warps × threads-per-warp = 8 × 32 = 256`），在后续 Pass 63（ConvertTritonGPUToLLVM）中可以看到这一映射。

本 kernel 在 sm_120 上以 256 线程/CTA、8 warp、每线程处理 2 个元素（sizePerThread=2）的配置运行，BLOCK_SIZE=512 = 256线程 × 2元素/线程，计算资源利用率满载。
