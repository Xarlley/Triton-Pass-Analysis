# Pass 57：SCFToControlFlowPass

> kernel：卷积 (Convolution) ｜ CLI：`convert-scf-to-cf` ｜ 编译流水线第 57 个 Pass

## 这个 Pass 的作用

`SCFToControlFlowPass`（结构化控制流到平坦控制流转换）将高层次的结构化循环 `scf.for` 转换为基本块（basic block）和分支指令 `cf.br` / `cf.cond_br`。这是向 LLVM IR 下沉的关键步骤：LLVM 不理解 `scf.for`，只能理解显式的 CFG（控制流图）。IR 行数从 403 增至 408（增加 5 行，因为需要插入循环头部的条件分支基本块 `^bb1` 和循环体基本块 `^bb2`）。参见 [`Prefetch.md`](../../Passes/Prefetch.md) 中对软件流水线循环结构的描述。

## IR 变化

**`scf.for` 替换为 `cf.br` + `cf.cond_br` + 基本块**：

```mlir
// 变换前（结构化 scf.for 循环）
%acc:9 = scf.for %acc_149 = %c0_i32 to %c9_i32 step %c1_i32
  iter_args(%arg4 = %cst, %acc_150 = %c2_i32, %acc_151 = %c-1_i32,
            %matrix_x_152 = %matrix_x_82, ..., %matrix_w_157 = %matrix_w_134)
  -> (tensor<128x64xf32, #blocked>, i32, i32, !ttg.async.token × 6) : i32 {
    // 循环体（9 次迭代的 steady-state 流水线）
    scf.yield %acc_167, %acc_170, %acc_163, ... : ...
}

// 变换后（平坦 CFG，三个基本块）
cf.br ^bb1(%c0_i32, %cst, %c2_i32, %c-1_i32, %matrix_x_82, ..., %matrix_w_134 : i32, tensor<...>, i32, i32, !ttg.async.token × 6)

^bb1(%acc: i32, %0: tensor<128x64xf32, #blocked>, %acc_135: i32, %acc_136: i32,
     %matrix_x_137: !ttg.async.token, ..., %matrix_w_142: !ttg.async.token):  // 2 preds: ^bb0, ^bb2
  %acc_143 = arith.cmpi slt, %acc, %c9_i32 : i32         // 循环条件 i < 9
  cf.cond_br %acc_143, ^bb2, ^bb3                         // 条件跳转

^bb2:  // 循环体（pred: ^bb1）
  // ... 全部循环体操作 ...
  %acc_204 = arith.addi %acc, %c1_i32 : i32              // i++
  cf.br ^bb1(%acc_204, %acc_153, %acc_156, %acc_147, ...) // 跳回循环头

^bb3:  // 循环出口（pred: ^bb1）
  %acc_205 = ttg.async_wait {num = 0 : i32}
  // ... epilogue ...
```

## 说明

`SCFToControlFlowPass` 是从 Triton 的高层 IR（具有结构化控制流的 SSACFG）到 LLVM IR（平坦 CFG）的关键桥梁。经过此 Pass 后，9 次迭代的卷积 K 循环被展开为如下 CFG 结构：

- **`^bb0`**（函数入口）：执行所有循环前的 prologue 操作（2 次预热 load），然后通过 `cf.br ^bb1(初始参数)` 跳入循环头。
- **`^bb1`**（循环头/phi 节点）：接受迭代变量作为基本块参数（`%acc, %0, %acc_135, ...`），在 LLVM 层面这些将成为 phi 指令。执行循环条件判断 `acc < 9`，条件成立跳 `^bb2`（循环体），否则跳 `^bb3`（出口）。
- **`^bb2`**（循环体）：执行 `ttg.async_wait`、`ttg.local_load`、`tt.dot` 和下一次 load 的 `ttg.async_copy_global_to_local`，最后通过 `cf.br ^bb1(更新后的参数)` 回到循环头。
- **`^bb3`**（循环出口/epilogue）：执行 `ttg.async_wait {num=0}` 等待所有 in-flight load，然后执行边界处理和输出写回。

5 行增加来自新增的 3 个基本块标签（`^bb1`, `^bb2`, `^bb3`）加上新增的 `cf.br` 跳转指令和 `cf.cond_br` 条件跳转指令，而 `scf.for` 和 `scf.yield` 各删去 1 行，净增 5 行。
