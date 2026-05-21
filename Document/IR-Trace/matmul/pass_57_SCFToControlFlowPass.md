# Pass 57：SCFToControlFlowPass

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`convert-scf-to-cf` ｜ 编译流水线第 57 个 Pass

## 这个 Pass 的作用

SCFToControlFlowPass（结构化控制流转控制流图）将 MLIR 的结构化控制流（Structured Control Flow，SCF）操作（`scf.for`, `scf.if`, `scf.while`）转换为低层的控制流图（Control Flow Graph，CFG）形式——即基本块（basic block）+ 分支指令（`cf.br`, `cf.cond_br`）。这是从 Triton 高层 IR 下降到 LLVM IR 的必要步骤，因为 LLVM IR 使用 CFG 而非结构化控制流。

对于本 kernel，`scf.for` 循环（128 次迭代的 K 维 reduction 循环）被展开为 3 个基本块：`^bb0`（循环初始化 + 跳转）、`^bb1`（循环头：检查终止条件）、`^bb2`（循环体：计算 + 更新状态 + 跳回 `^bb1`）、`^bb3`（循环后）。IR 行数从 271 行增加到 276 行（增加 5 行：新增基本块标签和分支指令）。

## IR 变化

**变换前（结构化 `scf.for`）：**

```mlir
%acc:11 = scf.for %acc_98 = %c0_i32 to %c128_i32 step %c1_i32
    iter_args(%arg4 = %cst, %acc_99 = %c3_i32, %acc_100 = %c-1_i32,
              %a_101 = %a_38, ..., %b_108 = %b_86)
    -> (tensor<16x32xf32, #blocked>, i32, i32, !ttg.async.token x8)  : i32 {
  // 循环体
  scf.yield %acc_118, %acc_121, %acc_112, %a_102, ..., %b_141 : ...
} loc(#loc75)
%acc_87 = ttg.async_wait {num = 0 : i32}
```

**变换后（CFG 形式，3 个基本块）：**

```mlir
// 初始分支（循环前一次性初始化状态）
cf.br ^bb1(%c0_i32, %cst, %c3_i32, %c-1_i32, %a_38, ..., %b_86 :
           i32, tensor<16x32xf32, #blocked>, i32, i32, !ttg.async.token x8) loc(#loc75)

^bb1(%acc: i32, %2: tensor<16x32xf32, #blocked>, %acc_87: i32, %acc_88: i32,
     %a_89: !ttg.async.token, ..., %b_96: !ttg.async.token):  // 2 个前驱：^bb0, ^bb2
  %acc_97 = arith.cmpi slt, %acc, %c128_i32 : i32 loc(#loc75)   ← 循环条件检查
  cf.cond_br %acc_97, ^bb2, ^bb3 loc(#loc75)

^bb2:  // 循环体（^bb1 的后继）
  // ...循环体内所有计算...
  cf.br ^bb1(%acc_107, %acc_110, %acc_101, %a_90, ..., %b_144 :
             i32, tensor<16x32xf32, #blocked>, i32, i32, !ttg.async.token x8) loc(#loc75)

^bb3:  // 循环出口
  %acc_87_post = ttg.async_wait {num = 0 : i32} loc(#loc75)
  ...
```

## 说明

`scf.for` 的 `iter_args`（循环携带值）对应 CFG 中基本块参数（block arguments）：`^bb1` 接受所有循环状态（归纳变量 `%acc`、dot 累加器 `%2`、槽索引 `%acc_87/%acc_88`、8 个 async token）作为参数，每次迭代通过 `cf.br ^bb1(...)` 传递更新后的值，形成循环。

循环终止条件 `%acc_98 < %c128_i32` 被编码为 `cf.cond_br`，条件为真时进入循环体 `^bb2`，为假时跳出到循环后 `^bb3`。

注意：变换后循环归纳变量 `%acc_98`（原 `scf.for` 的循环变量）成为 `^bb1` 的第一个块参数 `%acc`，其步进（`+1`）需要在循环体 `^bb2` 内显式计算并在 `cf.br ^bb1(...)` 中传递。这是将结构化 `for` 循环（含归纳变量自动递增）降低为显式 CFG 的必要变换。
