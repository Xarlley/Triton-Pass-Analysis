# Pass 26：CSEPass

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`cse` ｜ 编译流水线第 26 个 Pass

## 这个 Pass 的作用

CSEPass（Common Subexpression Elimination，公共子表达式消除）负责识别并消除 IR 中具有相同操作数和属性的重复计算，将后续出现的重复操作替换为对首次计算结果的引用，从而减少冗余运算和代码体积。

## IR 变化

此 Pass 的核心变化是**消除了 after.mlir 中多余的诊断 dump 副本**，将 271 行的 IR 压缩回 134 行。

before.mlir（271 行）包含两份完整的 module IR：
- 第 1 份（行 1–133）：实际 kernel 的 IR（SCCP 规范化后）；
- 第 2 份（行 134–271）：`// IR Dump Before VerifyWarpSpecializationPartitions` 的旧版本（带旧常量命名），两者是完全不同名称的相同语义内容。

CSE 通过值语义识别这两份 IR 中的公共子表达式，最终 after.mlir 只保留一份规范化版本（134 行），内容为 SCCP 产出的新常量顺序。

关键的 before → after 对比（函数体起始部分）：

**before（行 143–151，第二份 dump 的旧常量命名）**：
```mlir
%tmp5 = arith.constant dense<14400> : tensor<512xi32, #blocked>
%tmp3 = arith.constant dense<14336> : tensor<512xi32, #blocked>
%cst = arith.constant dense<28672> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<128> : tensor<512xi32, #blocked>
%x2 = arith.constant dense<7168> : tensor<512xi32, #blocked>
%x1 = arith.constant dense<112> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<64> : tensor<512xi32, #blocked>
```

**after（单份，SCCP 规范化命名）**：
```mlir
%cst = arith.constant dense<64> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<112> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<7168> : tensor<512xi32, #blocked>
%cst_2 = arith.constant dense<128> : tensor<512xi32, #blocked>
%cst_3 = arith.constant dense<28672> : tensor<512xi32, #blocked>
%cst_4 = arith.constant dense<14336> : tensor<512xi32, #blocked>
%cst_5 = arith.constant dense<14400> : tensor<512xi32, #blocked>
```

## 说明

从行数变化（271→134）可以看出，CSE 消除的并非用户代码中的重复计算，而是**编译器内部调试 dump 引入的重复 IR 块**。Triton 的 NVWS warp 专化分析流程在捕获 IR 快照时会将同一 kernel 内容写入两次，CSE 将这两份语义等价的 module 合并为一份。

对于本 MaxPool+BN+LIF kernel 的实际计算逻辑，CSE 没有消除任何 pointwise 操作（每个 load 的地址不同、每个 cmpf/select 的操作数不同，均无真正的重复子表达式）。这与 kernel 的设计一致：Inductor 生成的 pointwise kernel 本身已是最简形式，每条计算链对应 MaxPool 2×2 窗口中一个独立候选值的处理路径，不存在跨路径的共同子表达式。

此 Pass 后，编译器进入 NVWSLowerAref（Pass 27），开始真正处理 NVWS 框架下的 aref 降级。
