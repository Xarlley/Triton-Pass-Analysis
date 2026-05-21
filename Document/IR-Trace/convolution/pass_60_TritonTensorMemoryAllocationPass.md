# Pass 60：TritonTensorMemoryAllocationPass

> kernel：卷积 (Convolution) ｜ CLI：`triton-tensor-memory-allocation` ｜ 编译流水线第 60 个 Pass

## 这个 Pass 的作用

`TritonTensorMemoryAllocationPass`（Tensor Memory 分配 Pass）分析 IR 中 Tensor Memory（TMEM）操作的使用情况，计算所需的 Tensor Memory 大小，并将结果写入 module 属性 `ttg.tensor_memory_size`。Tensor Memory 是 NVIDIA Blackwell (sm_120) 架构上的新型片上存储单元，专用于矩阵运算的累加器数据。对于本卷积 kernel，由于累加器直接使用寄存器（`tensor<128x64xf32, #blocked>`）而非 TMEM，所需的 Tensor Memory 为 0。IR 行数保持 408 不变，仅修改 module 属性。

## IR 变化

**在 module 属性中记录 Tensor Memory 大小**：

```mlir
// 变换前
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 36864 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {

// 变换后（新增 ttg.tensor_memory_size = 0）
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 36864 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
```

## 说明

Blackwell sm_120 引入了专用的 Tensor Memory（TMEM）用于存放 Warp MMA（矩阵乘法）的累加器。使用 TMEM 的优势在于：TMEM 位于 SM 片上，具有比 shared memory 更高的带宽和更低的延迟，且可以持久存在于多次 warp 调度之间（寄存器则随 warp 上下文保存/恢复有额外开销）。

本卷积 kernel 的 `tt.dot` 使用了传统的寄存器累加（`tensor<128x64xf32, #blocked3>`），而非 Tensor Memory 累加，因此 `ttg.tensor_memory_size = 0`。在更高级的 Blackwell 内核中（如需要极大矩阵输出的 matmul），编译器可能会将累加器分配到 TMEM 以节省寄存器压力。

`ttg.tensor_memory_size = 0` 的记录告知后续的 ConvertTritonGPUToLLVM（Pass 63）不需要生成 TMEM 分配的 PTX 指令（如 `tcgen05.alloc`），从而简化代码生成路径。
