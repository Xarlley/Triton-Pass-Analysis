# Pass 61：TritonNvidiaGPUCheckMatmulTwoCTAPass

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`triton-nvidia-check-matmul-two-cta` ｜ 编译流水线第 61 个 Pass

## 这个 Pass 的作用

TritonNvidiaGPUCheckMatmulTwoCTAPass 检查 kernel 是否使用了 Blackwell 的"双 CTA 模式"（two-CTA matmul），即通过两个 CTA 协同完成一个 tile 的矩阵乘法以提高带宽利用率。检查结果以布尔属性 `"ttng.two-ctas"` 写入 module，供后续 lowering 使用。

## IR 变化

此 Pass 仅修改了 **module 级别属性**，函数体内容不变。IR 行数保持 134 行不变。

**变化前（before）**：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
```

**变化后（after）**：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32, "ttng.two-ctas" = false} {
```

新增属性：`"ttng.two-ctas" = false`。

## 说明

`"ttng.two-ctas" = false` 表明本 kernel 不使用双 CTA matmul 模式。这对 MaxPool+BN+LIF pointwise kernel 是必然结果：

1. 双 CTA 模式仅适用于含 warpgroup matmul（WGMMA）且 CTA 配置为 `num_ctas=2` 的 kernel；
2. 本 kernel 的 `ttg.num-ctas = 1`，根本不在双 CTA 配置下；
3. 本 kernel 无任何 matmul 操作。

经过此 Pass，module 的所有目标相关属性均已就位：
```mlir
"ttg.num-ctas" = 1
"ttg.num-warps" = 8
ttg.shared = 0
ttg.target = "cuda:120"
ttg.tensor_memory_size = 0
"ttg.threads-per-warp" = 32
"ttg.total-num-warps" = 8
"ttng.two-ctas" = false
```

这 8 个属性完整描述了本 kernel 在 Blackwell sm_120 上的执行配置，是 Pass 63（ConvertTritonGPUToLLVM）生成正确 LLVM IR（包括 `nvvm.reqntid`、内核签名、地址空间注解）所需的全部元信息。至此，kernel 准备好进入最终的 LLVM 代码生成阶段。
