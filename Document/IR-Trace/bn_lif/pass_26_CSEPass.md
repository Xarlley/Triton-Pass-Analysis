# Pass 26：CSEPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`cse` ｜ 编译流水线第 26 个 Pass

## 这个 Pass 的作用

CSE（Common Subexpression Elimination，公共子表达式消除）Pass 识别并消除程序中计算结果相同的重复操作。当同一个表达式在代码中出现多次且结果不变时，CSE 保留第一次计算，后续出现的替换为第一次结果的引用，从而减少冗余计算。在本流水线位置，CSE 的主要任务是消除因 Pass 20（AutomaticWarpSpecialization）产生的双份 IR——两份完全相同的函数体中有大量重复的操作序列。

## IR 变化

本次变换是行数最大幅度变化之一：从 469 行降至 233 行（减少 236 行），即将双份 IR 合并回单份：

**变换前（469 行，包含两份相同的 IR）：**

```mlir
// 第一份 IR
module attributes {"ttg.num-ctas" = 1 : i32, ...} {
  tt.func public @triton_poi_fused_...(...) {
    %cst = arith.constant dense<50176> : tensor<1x64xi32, #blocked>
    ...（约 100+ 行计算）...
  }
}
#loc121 = loc("tmp43"(#loc55))


// -----// IR Dump Before VerifyWarpSpecializationPartitions //----- //
module attributes {"ttg.num-ctas" = 1 : i32, ...} {
  tt.func public @triton_poi_fused_...(...) {
    %cst = arith.constant dense<50176> : tensor<1x64xi32, #blocked>
    ...（完全相同的约 100+ 行计算）...
  }
}
```

**变换后（233 行，单份 IR）：**

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_poi_fused_...(...) {
    %cst = arith.constant dense<50176> : tensor<1x64xi32, #blocked>
    ...（单份完整计算）...
  }
}
```

布局别名在 after.mlir 中以新的顺序出现（因为 CSE 重新扫描时遇到的顺序不同），但代表的布局与 Pass 25 after 中相同：

```mlir
// after.mlir
#blocked  = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

## 说明

这次 CSE 消除的不是 BN+LIF 计算内部的公共子表达式（kernel 本身的算术操作并无明显重复），而是因 Pass 20 的内部调试输出机制产生的两份完全相同的模块级重复。CSE 以模块为粒度识别了两个完全相同的 `tt.func` 定义，将其合并为一个。

对于 BN+LIF kernel 自身的计算，CSE 理论上可以识别的公共子表达式包括：4 次 LIF 发放循环中都使用的 `%cst_2`（0.5）、`%cst_1`（1.0）、`%cst_0`（0.0）等张量常量，但这些已在前序 Pass 中作为共享常量存在，此处不再重复消除。消除双份 IR 后，流水线回到干净的单 kernel IR 状态，为后续的 `NVWSLowerAref`（Pass 27）做好准备。
