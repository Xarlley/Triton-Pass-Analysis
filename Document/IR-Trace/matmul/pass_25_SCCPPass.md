# Pass 25：SCCPPass

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`sccp` ｜ 编译流水线第 25 个 Pass

## 这个 Pass 的作用

SCCPPass（Sparse Conditional Constant Propagation）执行稀疏条件常量传播，将编译时可确定值的变量直接替换为常量，并消除由此产生的冗余声明。此外，该 Pass 还对内存中的常量 dense 张量（如 `dense<4096>`）进行了布局和声明重排，统一了常量命名规范。在本 kernel 中，最关键的变化是将 `%width = arith.constant 1024 : i32`（具名符号常量）折叠替换为标准的 `%c1024_i32 = arith.constant 1024 : i32`，并将所有引用 `%width` 的地方替换为 `%c1024_i32`。

Pass 25 的 before 文件为 367 行（来自 Pass 24 after 的第一段 dump），after 文件为 362 行（减少 5 行）。

## IR 变化

**关键变化：常量 `%width` 被消除，改为标准化的 `%c1024_i32`；布局名称重排；常量声明顺序调整。**

**变换前（before，具名 `%width` 常量）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8],  warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4],  warpsPerCTA = [1, 2], order = [0, 1]}>
...
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %width = arith.constant 1024 : i32    ← 具名符号常量
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant dense<4> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %cst_1 = arith.constant dense<4096> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %cst_2 = arith.constant dense<4096> : tensor<16x1xi32, #blocked1>
    %cst_3 = arith.constant dense<4096> : tensor<1x32xi32, #blocked2>
    %cst_4 = arith.constant dense<4096> : tensor<1x32xi32, #blocked1>
    %cst_5 = arith.constant dense<4> : tensor<16x1xi32, #blocked1>
    ...
    %group_id = arith.divsi %pid, %width : i32    ← 引用 %width
    ...
    %pid_n = arith.remsi %pid, %width : i32       ← 引用 %width
```

**变换后（after，`%width` 折叠为 `%c1024_i32`；布局名称重排）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8],  warpsPerCTA = [2, 1], order = [1, 0]}>  ← 原 #blocked1
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4],  warpsPerCTA = [1, 2], order = [0, 1]}>  ← 原 #blocked2
#blocked2 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>  ← 原 #blocked
...
    %cst    = arith.constant dense<4>    : tensor<16x1xi32, #blocked>
    %cst_0  = arith.constant dense<4096> : tensor<1x32xi32, #blocked>
    %cst_1  = arith.constant dense<4096> : tensor<1x32xi32, #blocked1>
    %cst_2  = arith.constant dense<4096> : tensor<16x1xi32, #blocked>
    %cst_3  = arith.constant dense<4096> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_4  = arith.constant dense<4>   : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c8_i32   = arith.constant 8   : i32
    %c32_i32  = arith.constant 32  : i32
    %c16_i32  = arith.constant 16  : i32
    %c0_i32   = arith.constant 0   : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32   ← 标准化具名常量，替代 %width
    %c1_i32   = arith.constant 1   : i32
    %cst_5  = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #blocked2>
    ...
    %group_id = arith.divsi %pid, %c1024_i32 : i32   ← %width → %c1024_i32
    ...
    %pid_n    = arith.remsi  %pid, %c1024_i32 : i32  ← %width → %c1024_i32
```

## 说明

SCCP 确认了两处 `%width` 引用可以替换为 `%c1024_i32`，并重新整理了常量声明顺序（将 dense tensor 常量置前，标量常量居后）。同时，与 Pass 24 after 文件第二段 dump 一致，布局名称发生了重排：原来的 accumulator 布局（`sizePerThread=[2,2]`）从 `#blocked` 改称 `#blocked2`，A-load 布局（`sizePerThread=[1,4]`）改称 `#blocked`，B-load 布局（`sizePerThread=[4,1]`）改称 `#blocked1`。

这种重排是 MLIR pass manager 在 `VerifyWarpSpecializationPartitions` 内部重新规范化属性别名顺序时产生的，不影响 IR 的语义正确性。之后 `scf.for` 的 iter_args 类型从 `tensor<16x32xf32, #blocked>`（原 accumulator 名称）改为 `tensor<16x32xf32, #blocked2>`（新名称），dot 指令的输出类型也相应更新为 `#ttg.dot_op<{opIdx = 0, parent = #blocked2}>` 和 `#ttg.dot_op<{opIdx = 1, parent = #blocked2}>`。

这 5 行的净减少来自：消除 `%width` 声明（1 行）+ 常量声明合并/重排缩减（4 行）。
