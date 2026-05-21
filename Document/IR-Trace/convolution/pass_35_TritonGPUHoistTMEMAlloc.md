# Pass 35：TritonGPUHoistTMEMAlloc

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-hoist-tmem-alloc` ｜ 编译流水线第 35 个 Pass

## 这个 Pass 的作用

`TritonGPUHoistTMEMAlloc`（第二次执行，Pass 35）将 Tensor Memory 分配操作（`ttg.local_alloc`）提升到 `scf.for` 循环外部，以避免每次循环迭代都重新分配和释放 shared memory。参见 [`HoistTMEMAlloc.md`](../../Passes/HoistTMEMAlloc.md)。经过 Pass 32（Pipeline）的展开，循环体的结构已经固定，此时将 shared memory 分配提升到函数入口处可以确保缓冲区在整个循环期间保持有效。IR 行数不变（425→425），但分配操作的位置发生了变化。

## IR 变化

本 Pass 将循环体内（稳态循环的 `iter_args` 部分）残留的 `ttg.local_alloc` 操作提升到循环前：

```mlir
// 变换前（在软件流水线的 prologue 中分配 shared memory）
// Pass 32 已将分配放在 prologue 之前，但 Pass 35 确保其位置严格在循环之外
...
%acc_154:12 = scf.for %acc_169 = %c0_i32 to %c9_i32 step %c1_i32
  iter_args(..., %matrix_x_175 = %matrix_x_83, ...) -> ... {
    // 循环内使用这些 shared memory handle
    %matrix_x_185 = ttg.async_wait ...
    %matrix_x_187 = ttg.local_load %matrix_x_186 ...
```

在 Pass 35 之后，`ttg.local_alloc` 调用被确保出现在函数体的最顶端（所有循环之前），与 Pass 32 产出的 `%matrix_x` 和 `%matrix_w` 分配位置相同，确保 shared memory 的生命周期覆盖整个 kernel 执行。在此版本中，`iter_args` 中的 token 传递机制（`%matrix_x_175..%matrix_w_180`）发生了轻微调整——部分 token 的初始值从 prologue 中的 async copy token 改为通过 `ub.poison` 占位的方式来处理边界条件。

## 说明

在 Blackwell sm_120 上，shared memory 分配在 kernel 启动时一次性确定（通过 PTX 的 `.shared .align` 指令），不能在运行时动态分配。`HoistTMEMAlloc` 确保所有 `ttg.local_alloc` 在 IR 层面也处于最外层作用域，与最终 PTX 的语义保持一致。对本 kernel 而言，这两个 shared memory 缓冲（`3×128×16×4=24576` 字节的激活缓冲 + `3×16×64×4=12288` 字节的权重缓冲 = 共 36864 字节）在函数开始时就被分配，在函数结束时（`ttg.local_dealloc`）释放，保证了整个卷积 K 循环（9次迭代）期间数据始终有效。
