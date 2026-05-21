# Pass 56：TritonGPUAllocateWarpGroups

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`tritongpu-allocate-warp-groups` ｜ 编译流水线第 56 个 Pass

## 这个 Pass 的作用

TritonGPUAllocateWarpGroups 为 Warp 专业化框架中的 warp group 分配最终的 warp 数量，并将结果记录在模块属性 `"ttg.total-num-warps"` 中。对于未进行 warp 专业化的 kernel，Pass 将所有 warp 归为一个 group，并向模块添加 `"ttg.total-num-warps"` 属性，其值等于 `"ttg.num-warps"`。

## IR 变化

本次变换极其简单，仅在模块属性中新增了一个字段：

**变换前：**

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
```

**变换后：**

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
```

新增属性：`"ttg.total-num-warps" = 4 : i32`

函数体内部所有操作均无变化，行数保持 233 行。

## 说明

`"ttg.total-num-warps"` 属性区分于 `"ttg.num-warps"`：
- `"ttg.num-warps"` = 4：每个 CTA（线程块）中用于**计算**的 warp 数量。
- `"ttg.total-num-warps"` = 4：整个 CTA 中**所有** warp 的总数，包括可能存在的专属 warp（如专用 producer warp）。

对于未经 warp 专业化的 kernel（如本 BN+LIF kernel），二者相等，均为 4。每个 warp 包含 32 个线程，因此整个 CTA 有 4×32=128 个线程，这与 Pass 63 after 中的 `nvvm.reqntid = array<i32: 128>` 完全对应。

在有 warp 专业化的 GEMM kernel 中，`"ttg.total-num-warps"` 可能更大（如 8），其中 `"ttg.num-warps"` = 4 个计算 warp + 4 个 producer warp（负责 TMA 异步加载）= 8 个总 warp。这个属性是后续 Pass（如 `ConvertTritonGPUToLLVM`）生成正确 NVVM kernel 属性的依据。
