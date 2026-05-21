# Pass 25：SCCPPass

> kernel：卷积 (Convolution) ｜ CLI：`sccp` ｜ 编译流水线第 25 个 Pass

## 这个 Pass 的作用

`SCCPPass`（Sparse Conditional Constant Propagation，稀疏条件常量传播）是标准 MLIR/LLVM Pass，对 IR 进行数据流分析，将那些在编译时可知为常量的值替换为字面量，并删除由此变为死代码的操作。本次执行（Pass 25，第一次 SCCP）处于 Warp Specialization 之后，主要对已重排 layout 的常量定义进行传播。IR 行数不变（587→587），但内部常量表示被规范化。

## IR 变化

SCCP 将零初始化的浮点张量常量（`dense<0.000000e+00>`）替换为具体 layout 下的非零整数常量（卷积相关的步长和尺寸参数），并对 `#blocked` 编号进行重排以匹配 Warp Specialization 后的分区视图：

```mlir
// 变换前（浮点零张量作为累加器初值）
%cst   = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked1>
%cst_1 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked2>
%c9_i32 = arith.constant 9 : i32
%c1_i32 = arith.constant 1 : i32
%c64_i32 = arith.constant 64 : i32
%c3_i32 = arith.constant 3 : i32

// 变换后（SCCP 将常量传播并重组为 layout 感知常量）
%cst    = arith.constant dense<64>       : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_0  = arith.constant dense<3211264>  : tensor<128x1xi32, #blocked>
%cst_1  = arith.constant dense<64>       : tensor<128x1xi32, #blocked>
%cst_2  = arith.constant dense<14336>    : tensor<128x1xi32, #blocked>
%cst_11 = arith.constant dense<27>       : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%cst_12 = arith.constant dense<150528>   : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
```

## 说明

SCCP 在此阶段主要完成两件事：一是将 layout 重排后可以静态确定值的常量（如卷积步长 672、图像尺寸 224、通道数 64、权重组大小 27 = 3×3×3 等）传播到使用点；二是在 Warp Specialization 后，将 producer 分区（cluster=3）和 consumer 分区（cluster=0）各自的常量视图分开处理，避免跨分区共享可能导致的错误传播。数值 3211264 = 64×224×224 是单张图像的输出激活总元素数（out_channels × H × W），150528 = 3×224×224 是单张图像的输入激活总元素数。这些常量已在 Pass 14（LICM）中被提升到循环外，SCCP 在此进一步确认其为编译期常量。
