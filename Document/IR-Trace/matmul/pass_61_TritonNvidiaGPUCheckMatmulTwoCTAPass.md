# Pass 61：TritonNvidiaGPUCheckMatmulTwoCTAPass

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`triton-nvidia-gpu-check-matmul-two-cta` ｜ 编译流水线第 61 个 Pass

## 这个 Pass 的作用

TritonNvidiaGPUCheckMatmulTwoCTAPass 检查内核是否启用了双 CTA（Two-CTA）矩阵乘法模式——这是 NVIDIA Blackwell 架构（sm_120）的新特性，允许两个 CTA 协作完成一次矩阵乘法，以提高 SM 利用率和 wgmma 吞吐量。该 Pass 将检查结果以 `"ttng.two-ctas"` 布尔属性写入 `module attributes`。

对于本 kernel（`num_ctas=1`），仅使用单个 CTA，不启用 Two-CTA 协作模式，因此写入 `"ttng.two-ctas" = false`。IR 行数保持 276 行不变，仅 `module attributes` 行新增该属性。

## IR 变化

**唯一变化：`module attributes` 新增 `"ttng.two-ctas" = false`。**

**变换前：**

```mlir
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 2 : i32,
  ttg.shared = 24576 : i32,
  ttg.target = "cuda:120",
  ttg.tensor_memory_size = 0 : i32,
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
  ttg.tensor_memory_size = 0 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.total-num-warps" = 2 : i32,
  "ttng.two-ctas" = false               ← 新增
} {
```

## 说明

Two-CTA 模式（`"ttng.two-ctas" = true`）是 NVIDIA Hopper/Blackwell（sm_90a+/sm_120）引入的高级并行化特性：在此模式下，两个 CTA 通过共享内存和 `nvgpu.cluster_barrier_arrive/wait` 等 CGA（Cooperative Thread Array Group）指令协同工作，共同处理一个输出 tile，从而提高每个 SM 的算力利用率。

对于本 kernel，由于：
1. `num_ctas=1`：仅分配 1 个 CTA，无法形成 2-CTA 协作组
2. `num_warps=2`：warp 数量较少，不满足 Two-CTA 模式对 warp 组数量的要求

因此 `"ttng.two-ctas" = false`。

在 `"ttng.two-ctas" = true` 的 kernel 中（通常需要 `num_ctas=2` 且 `num_warps≥4`），后续 ConvertTritonGPUToLLVM（Pass 63）会生成额外的 `nvvm.barrier` 和跨 CTA 的 shared memory 同步代码。对本 kernel 该分支不激活。

经过 Pass 59、60、61，`module attributes` 已完整记录了内核的所有资源配置（SMEM 大小、TMem 大小、Two-CTA 标志、warp 数、CTA 数、thread-per-warp、total-num-warps），为 Pass 63 的 LLVM lowering 提供了完整的元信息。
