# Pass 52：SCCPPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`sccp` ｜ 编译流水线第 52 个 Pass

## 这个 Pass 的作用

这是流水线中第二次执行 SCCP（Sparse Conditional Constant Propagation）。在此位置，SCCP 对经过所有 GPU 优化 Pass 处理后的最终 TTGIR 进行常量传播和常量折叠，主要目标是将计算链中仍以变量形式存在的常量值直接替换为字面量，为 LLVM 降级做最后准备。

## IR 变化

本次变换对 IR 的**布局别名顺序**再次进行了重组，同时对常量池结构做了重新排列。行数保持 233 行不变，但布局别名的 `#blocked` 和 `#blocked1` 发生了对调：

**变换前：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
    %cst = arith.constant dense<50176> : tensor<1x64xi32, #blocked>
    %cst_0 = arith.constant dense<9633792> : tensor<1x64xi32, #blocked1>
    ...
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked1>
    %cst_7 = arith.constant dense<1.000000e+00> : tensor<16x64xf32, #blocked1>
    %cst_8 = arith.constant dense<5.000000e-01> : tensor<16x64xf32, #blocked1>
```

**变换后（布局别名互换，常量以新布局名重排）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
    %cst   = arith.constant dense<5.000000e-01> : tensor<16x64xf32, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<16x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked>
    %c16_i32 = arith.constant 16 : i32
    %cst_2 = arith.constant dense<64> : tensor<1x64xi32, #blocked>
    %cst_3 = arith.constant dense<64> : tensor<1x64xi32, #blocked1>
    %cst_4 = arith.constant dense<64> : tensor<16x1xi32, #blocked>
    %cst_5 = arith.constant dense<3211264> : tensor<1x64xi32, #blocked>
    %cst_6 = arith.constant dense<6422528> : tensor<1x64xi32, #blocked>
    %cst_7 = arith.constant dense<9633792> : tensor<1x64xi32, #blocked>
    %cst_8 = arith.constant dense<50176> : tensor<1x64xi32, #blocked1>
```

注意：浮点常量（0.5、1.0、0.0）现在排在整数常量之前，且都属于计算主路径的 `#blocked`（读取布局），整数偏移常量（3211264、6422528、9633792）也移到了 `#blocked`，只有 `50176` 属于 `#blocked1`（写入路径布局）。

## 说明

Pass 52 的 SCCP 是在所有 GPU 优化完成后（加速 matmul、布局去重、流水线等均为 no-op 后）执行的一次"最终常量整理"。对本 BN+LIF kernel，SCCP 的实质工作是：

1. **重新确认布局别名命名优先级**：将使用频率最高的布局（读取路径 `{sizePerThread=[1,4],...}`）设为 `#blocked`，将写入路径（`{sizePerThread=[4,1],...}`）设为 `#blocked1`。这只影响别名文本，不影响机器码。

2. **常量排列优化**：将 BN+LIF 计算中最频繁使用的浮点常量（0.5、1.0、0.0）排到常量池顶部，对应于 LIF ATan 替代梯度和 Heaviside 函数的参数（如 `cmpf oge, %tmp2, %cst_0` 中的阈值 1.0）。

3. **无常量折叠机会**：kernel 中没有编译期可知的条件分支（所有 `cmpf` 的输入都依赖运行时数据），因此 SCCP 没有折叠任何 LIF 计算。

经此整理后，IR 进入最终的 TTGIR 形态，准备进入 Pass 63 的 `ConvertTritonGPUToLLVM` 降级。
