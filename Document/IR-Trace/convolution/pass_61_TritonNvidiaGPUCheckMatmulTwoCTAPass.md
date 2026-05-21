# Pass 61：TritonNvidiaGPUCheckMatmulTwoCTAPass

> kernel：卷积 (Convolution) ｜ CLI：`triton-nvidia-check-matmul-two-cta` ｜ 编译流水线第 61 个 Pass

## 这个 Pass 的作用

`TritonNvidiaGPUCheckMatmulTwoCTAPass`（Two-CTA Matmul 检查 Pass）检查 kernel 是否采用了 NVIDIA Blackwell sm_120 上的 Two-CTA 协同矩阵乘法模式（即两个 CTA 协同计算一个更大的矩阵分块）。检查结果以 `"ttng.two-ctas"` 属性写入 module。对于本卷积 kernel（单 CTA，`num_ctas=1`），检查结果为 `false`。IR 行数保持 408 不变。

## IR 变化

**在 module 属性中记录 Two-CTA 状态**：

```mlir
// 变换前
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 36864 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {

// 变换后（新增 ttng.two-ctas = false）
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 36864 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, "ttng.two-ctas" = false} {
```

## 说明

Two-CTA 矩阵乘法是 Blackwell 架构的高级特性：两个相邻的 CTA 可以通过 SM-to-SM 互联以更高带宽共享数据，协同计算大型矩阵分块，从而突破单 CTA 的 shared memory 容量限制。这种模式在大型 Transformer 模型的 attention 计算和大矩阵 GEMM 中特别有益。

本卷积 kernel 的矩阵分块为 128×16（激活）× 16×64（权重），在 4-warp 单 CTA 的配置下完全可以放入 36KB shared memory，无需 Two-CTA 协作。因此 `ttng.two-ctas = false` 是预期的正确结果。

此属性将被 Pass 63（ConvertTritonGPUToLLVM）用于选择正确的代码生成路径：当 `two-ctas = false` 时，生成标准的单 CTA LLVM IR；当 `two-ctas = true` 时，需要生成 `tcgen05.mma.cta_pair` 等 Two-CTA 专用指令。
