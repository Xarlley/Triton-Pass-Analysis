# Pass 52：SCCPPass

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`sccp` ｜ 编译流水线第 52 个 Pass

## 这个 Pass 的作用

此为流水线中第二次运行 SCCPPass（Sparse Conditional Constant Propagation）。与 Pass 25 类似，此 Pass 在 lowering 前的最后清理阶段再次对常量进行规范化整理。这次 SCCP 在 TritonNvidiaGPUMMALoweringPass（Pass 51）之后运行，主要作用是将常量池重排为"按使用顺序"的排列，以便 LLVM 后端生成更高质量的寄存器分配。

## IR 变化

常量池的**排列顺序发生了逆转**：Pass 25 将常量按值从小到大排列（64, 112, 128, …），Pass 52 将其**按使用顺序或降序重排**（14400, 14336, 28672, 128, 7168, 112, 64），并且 `%c512_i32` 从列表末尾提前到列表最前面。

**变化前（before，Pass 25 的升序排列）**：
```mlir
%cst   = arith.constant dense<64>    : tensor<512xi32, #blocked> loc(#loc1)
%cst_0 = arith.constant dense<112>   : tensor<512xi32, #blocked> loc(#loc1)
%cst_1 = arith.constant dense<7168>  : tensor<512xi32, #blocked> loc(#loc1)
%cst_2 = arith.constant dense<128>   : tensor<512xi32, #blocked> loc(#loc1)
%cst_3 = arith.constant dense<28672> : tensor<512xi32, #blocked> loc(#loc1)
%cst_4 = arith.constant dense<14336> : tensor<512xi32, #blocked> loc(#loc1)
%cst_5 = arith.constant dense<14400> : tensor<512xi32, #blocked> loc(#loc1)
%c512_i32 = arith.constant 512 : i32 loc(#loc1)
```

**变化后（after，使用顺序/降序排列）**：
```mlir
%c512_i32 = arith.constant 512 : i32 loc(#loc1)
%cst   = arith.constant dense<14400> : tensor<512xi32, #blocked> loc(#loc1)
%cst_0 = arith.constant dense<14336> : tensor<512xi32, #blocked> loc(#loc1)
%cst_1 = arith.constant dense<28672> : tensor<512xi32, #blocked> loc(#loc1)
%cst_2 = arith.constant dense<128>   : tensor<512xi32, #blocked> loc(#loc1)
%cst_3 = arith.constant dense<7168>  : tensor<512xi32, #blocked> loc(#loc1)
%cst_4 = arith.constant dense<112>   : tensor<512xi32, #blocked> loc(#loc1)
%cst_5 = arith.constant dense<64>    : tensor<512xi32, #blocked> loc(#loc1)
```

随之，所有引用常量的操作更新编号（如 `arith.remsi %xindex_8, %cst` → `arith.remsi %xindex_8, %cst_5`，因为 64 现在是 `%cst_5`）。IR 行数保持 134 行不变。

## 说明

这次常量顺序变化是 Pass 52 的 SCCP 规范化副作用——第二次 SCCP 与第一次（Pass 25）使用了不同的常量排序策略（第一次升序，第二次降序或按首次使用位置的倒序）。这种不一致性是编译器内部实现细节，最终 PTX 生成结果不受影响，因为常量值本身未变。

注意 `%c512_i32`（标量整数 512）提前到列表最前面，是因为它在函数体中最先被使用（用于 `arith.muli %xoffset, %c512_i32`），SCCP 在此次运行中将标量常量与张量常量分开并优先排列。这一排列将持续到 Pass 63（ConvertTritonGPUToLLVM），影响 LLVM IR 中常量的初始化顺序。
