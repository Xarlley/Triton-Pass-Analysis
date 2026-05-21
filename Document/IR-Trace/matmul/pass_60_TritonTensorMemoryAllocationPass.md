# Pass 60：TritonTensorMemoryAllocationPass

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`triton-tensor-memory-allocation` ｜ 编译流水线第 60 个 Pass

## 这个 Pass 的作用

TritonTensorMemoryAllocationPass 负责计算并记录内核所需的 Tensor Memory（TMem）总大小，将结果作为 `ttg.tensor_memory_size` 属性写入 `module attributes`。Tensor Memory 是 NVIDIA Blackwell（sm_90a+）架构引入的新型存储层级，专用于 Warp Group MMA（wgmma）操作中的 accumulator 存储，需要在内核启动前由驱动分配。

对于本 kernel（`num_warps=2`），没有使用 Warp Specialization 模式，也没有 TMem 分配（Warp Group MMA 要求 `num_warps≥4`，当前仅 2 个 warp），因此 `ttg.tensor_memory_size = 0 : i32`（TMem 用量为零）。IR 行数保持 276 行不变，仅 `module attributes` 行新增该属性。

## IR 变化

**唯一变化：`module attributes` 新增 `ttg.tensor_memory_size = 0 : i32`。**

**变换前：**

```mlir
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 2 : i32,
  ttg.shared = 24576 : i32,
  ttg.target = "cuda:120",
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.total-num-warps" = 2 : i32
} {
```

**变换后：**

```mlir
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 2 : i32,
  ttg.shared = 24576 : i32,
  ttg.target = "cuda:120",
  ttg.tensor_memory_size = 0 : i32,   ← 新增
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.total-num-warps" = 2 : i32
} {
```

## 说明

`ttg.tensor_memory_size` 与 `ttg.shared` 的对比：
- `ttg.shared = 24576`：共享内存（SMEM）用量，以字节为单位，由 L1 cache/SMEM 共享的物理存储提供。
- `ttg.tensor_memory_size = 0`：Tensor Memory（TMem）用量，以某种内部单位记录，是 Blackwell 新增的独立存储资源（位于 SM 内，专用于 wgmma accumulator）。

对于真正使用 Warp Specialization + wgmma 的 kernel（`num_warps≥4`，通常 `num_warps=8` 或 `num_warps=16`），`ttg.tensor_memory_size` 会记录非零值，对应 TMem accumulator buffer 的大小。该值将在 ConvertTritonGPUToLLVM（Pass 63）中被用于设置内核的 TMem 请求大小，通过特殊的 `nvvm.setmaxnreg` 或 PTX `.reqnctaid` 等指令传递给 CUDA 运行时。

对于本 kernel，该 Pass 实际是 no-op（仅写入 `0`），但它是编译流水线中必须执行的标准步骤，以确保 module attributes 的完整性。
