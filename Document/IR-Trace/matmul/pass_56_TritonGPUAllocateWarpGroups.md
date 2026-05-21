# Pass 56：TritonGPUAllocateWarpGroups

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-allocate-warp-groups` ｜ 编译流水线第 56 个 Pass

## 这个 Pass 的作用

TritonGPUAllocateWarpGroups 为模块分配 warp 组信息，计算内核实际使用的总 warp 数量，并将结果作为 `"ttg.total-num-warps"` 属性附加到 `module` 上。该属性记录了所有 CTA 的 warp 数之和，用于后续代码生成阶段确定 CUDA 内核启动配置（kernel launch configuration）中的线程数。

对于本 kernel（`num_warps=2`，`num_ctas=1`），总 warp 数 = 2 × 1 = 2，因此添加属性 `"ttg.total-num-warps" = 2 : i32`。IR 行数保持 271 行不变，仅 `module attributes` 行新增该属性。

## IR 变化

**唯一变化：`module attributes` 新增 `"ttg.total-num-warps" = 2 : i32`。**

**变换前：**

```mlir
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 2 : i32,
  ttg.shared = 24576 : i32,
  ttg.target = "cuda:120",
  ttg.tensor_memory_size = 0 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttng.two-ctas" = false
} {
```

**变换后：**

```mlir
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 2 : i32,
  ttg.shared = 24576 : i32,
  ttg.target = "cuda:120",
  ttg.tensor_memory_size = 0 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.total-num-warps" = 2 : i32,   ← 新增
  "ttng.two-ctas" = false
} {
```

## 说明

`"ttg.total-num-warps"` 区别于 `"ttg.num-warps"`：
- `"ttg.num-warps" = 2`：每个 CTA 的 warp 数量。
- `"ttg.total-num-warps" = 2`：整个内核（所有 CTA）的总 warp 数量（= `num_warps × num_ctas = 2 × 1`）。

在多 CTA（TMA/CGA）配置中，`total-num-warps` 可能大于 `num-warps`（例如 `num_ctas=2, num_warps=4` 时，`total-num-warps = 8`）。该值将传递给 `ConvertTritonGPUToLLVM`（Pass 63），用于正确设置 `nvvm.reqntid`（required thread count）元数据，最终影响 PTX 的 `.reqntid` 指令和 CUDA 内核的线程块维度配置。
