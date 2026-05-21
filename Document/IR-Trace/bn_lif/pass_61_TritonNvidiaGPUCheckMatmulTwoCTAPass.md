# Pass 61：TritonNvidiaGPUCheckMatmulTwoCTAPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`triton-nvidia-check-matmul-two-cta` ｜ 编译流水线第 61 个 Pass

## 这个 Pass 的作用

TritonNvidiaGPUCheckMatmulTwoCTAPass 检查 kernel 是否使用了 Blackwell 架构特有的"双 CTA 协作矩阵乘法"（Two-CTA Matmul）模式。在 Blackwell 上，某些 MMA 操作可以由两个相邻 CTA 协同完成以提升效率（配合 `ttng.two-ctas` 特性）。Pass 检测 kernel 中是否存在对应的操作，并将结论写入模块属性 `"ttng.two-ctas"`。

## IR 变化

本次变换仅新增一个布尔型模块属性：

**变换前：**

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32}
```

**变换后：**

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, "ttng.two-ctas" = false}
```

新增属性：`"ttng.two-ctas" = false`

函数体内所有操作无变化，行数保持 233 行。

## 说明

`"ttng.two-ctas" = false` 表明本 kernel **不使用双 CTA 协作模式**，预期结果。

Blackwell 的 Two-CTA Matmul 是 sm_120 引入的新特性，允许两个 CTA 通过片上高速互连（cluster interconnect）共享数据，将矩阵分块计算分散到两个 CTA 以减少每个 CTA 的 SMEM 占用并提升带宽利用率。这种模式需要：
1. `"ttg.num-ctas" > 1` 的 cluster 配置（本 kernel 为 1）
2. 专用的 `ttng.async_tma_gather` 或 `ttng.mma_tma` 类指令（本 kernel 均无）
3. 专用的 Two-CTA MMA 布局（本 kernel 无）

BN+LIF pointwise kernel 是彻底的单 CTA 操作，每个 CTA 独立处理一个 `16×64` 的 tile，无需跨 CTA 通信。`"ttng.two-ctas" = false` 告知后续的 LLVM 降级（Pass 63）不需要生成 cluster 相关的同步指令，直接生成标准单 CTA 的 kernel 代码。

至此，模块属性已完整：
```
num-ctas=1, num-warps=4, shared=4096, target=cuda:120,
tensor_memory_size=0, threads-per-warp=32, total-num-warps=4, two-ctas=false
```
这是 TTGIR 降级为 LLVMIR 所需的全部硬件元信息。
