# Pass 25：SCCPPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`sccp` ｜ 编译流水线第 25 个 Pass

## 这个 Pass 的作用

SCCP（Sparse Conditional Constant Propagation）是 MLIR 的标准常量传播 Pass。它分析程序中所有值的可能取值，如果某个值在所有执行路径上都是常量，则直接将其替换为常量字面量，并随之折叠依赖该值的下游操作（constant folding）。对于 tensor 常量，SCCP 还可以将整个 tensor 折叠为更紧凑的常量表示。在本流水线位置，SCCP 主要用于处理双份 IR 中的常量，为后续 CSE 做准备。

## IR 变化

本次变换的主要效果是**重新整理两份 IR 中的常量池顺序和布局别名编号**，同时将一些用于 BN 参数偏移的整数常量从计算式合并为直接常量：

**变换前（第一份 IR 中的常量，带完整 BN 偏移量）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
    %cst = arith.constant dense<5.000000e-01> : tensor<16x64xf32, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<16x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
```

**变换后（常量池重组，整数偏移量提升为独立常量 tensor）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
    %cst = arith.constant dense<50176> : tensor<1x64xi32, #blocked>
    %cst_0 = arith.constant dense<9633792> : tensor<1x64xi32, #blocked1>
    %cst_1 = arith.constant dense<6422528> : tensor<1x64xi32, #blocked1>
    %cst_2 = arith.constant dense<3211264> : tensor<1x64xi32, #blocked1>
    %cst_3 = arith.constant dense<64> : tensor<16x1xi32, #blocked1>
    %cst_4 = arith.constant dense<64> : tensor<1x64xi32, #blocked>
    %cst_5 = arith.constant dense<64> : tensor<1x64xi32, #blocked1>
```

整数偏移常量（3211264 = 50176×64、6422528 = 50176×128、9633792 = 50176×192）现在显式出现为 tensor 常量，分配到对应布局中。浮点常量（0.5、1.0、0.0）则在两份 IR 中分别保留各自的布局版本。

## 说明

这次变换揭示了 BN+LIF kernel 中数据布局的本质：
- **偏移量 3211264 = 64×50176**：对应 VGG16 第一个时间步的激活图偏移（64 通道 × 224×224 = 3211264 个 float），这是 LIF 神经元从时间步 t-1 读取历史膜电位所需的地址偏移量。
- **偏移量 6422528、9633792**：分别对应时间步 t-2 和 t-3 的偏移（2× 和 3× 的单步偏移），用于读取多个时间步的脉冲历史。

SCCP 此时的主要作用是为两份 IR 中的常量生成统一的布局注解，使得后续 CSE Pass（Pass 26）能识别跨两份 IR 的相同常量，从而将 469 行的双份 IR 合并回 233 行的单份 IR。
