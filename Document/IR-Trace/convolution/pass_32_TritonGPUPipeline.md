# Pass 32：TritonGPUPipeline

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-pipeline` ｜ 编译流水线第 32 个 Pass

## 这个 Pass 的作用

`TritonGPUPipeline` 将 Pass 31 产出的带有流水线注解的循环（包含 `ttg.local_alloc` + `ttg.async_copy_global_to_local` 原语）转化为完整的软件流水线结构，包括：prologue（预热阶段，提前发出前几次迭代的 load）、pipelined steady-state loop（稳态循环，每次迭代同时 load 下一批数据并计算当前批次）和 epilogue（排空阶段，等待最后几批数据完成并执行最后的 dot）。IR 行数从 815 降至 425（主 IR 提取，去掉调度注解副本）。参见 [`Prefetch.md`](../../Passes/Prefetch.md) 的相关概念。

## IR 变化

Pass 32 产出的 IR 中可以清晰看到软件流水线的三段式结构：

**Prologue**（预热，迭代 0 和 1 的 load）：

```mlir
// Prologue: 提前加载 K=0 的激活和权重
%matrix_x = ttg.local_alloc : () -> !ttg.memdesc<3x128x16xf32, #shared, #smem, mutable>
%matrix_w = ttg.local_alloc : () -> !ttg.memdesc<3x16x64xf32, #shared1, #smem, mutable>
// 槽位 0
%matrix_x_81 = ttg.memdesc_index %matrix_x[%c0_i32] : ... -> !ttg.memdesc<128x16xf32, ...>
%matrix_x_82 = ttg.async_copy_global_to_local %x_ptrs_66, %matrix_x_81 mask %mask_x_80 ...
// 槽位 1
%matrix_x_112 = ttg.memdesc_index %matrix_x[%c1_i32] : ... -> !ttg.memdesc<128x16xf32, ...>
%matrix_x_113 = ttg.async_copy_global_to_local %x_ptrs_97, %matrix_x_112 mask %mask_x_111 ...
```

**Steady-state 循环**（稳态，每次迭代 load + dot）：

```mlir
%acc_154:12 = scf.for %acc_169 = %c0_i32 to %c9_i32 step %c1_i32
              iter_args(%arg4 = %cst_19, %acc_170 = %acc, %acc_171 = %acc_0,
                        %acc_172 = %c3_i32, %acc_173 = %c3_i32, %acc_174 = %c3_i32,
                        %matrix_x_175 = %matrix_x_83, ..., %matrix_w_180 = %matrix_w_153) -> ... {
  // consumer: 等待并取出 slot[i-3] 的数据
  %matrix_x_185 = ttg.async_wait %matrix_x_175, %matrix_w_178 {num = 4 : i32}
  %matrix_x_186 = ttg.memdesc_index %matrix_x[%acc_184] : ...
  %matrix_x_187 = ttg.local_load %matrix_x_186 token %matrix_x_185 ...
  // dot
  %matrix_x_190 = ttg.convert_layout %matrix_x_187 : ... -> tensor<..., #ttg.dot_op<{opIdx = 0, ...}>>
  %matrix_w_191 = ttg.convert_layout %matrix_w_189 : ... -> tensor<..., #ttg.dot_op<{opIdx = 1, ...}>>
  %acc_192 = tt.dot %matrix_x_190, %matrix_w_191, %arg4 : ...
  // producer: 发出 load[i+3]（已在迭代开始时预测）
  ...
}
```

## 说明

经过此 Pass，卷积 kernel 的 K 循环（9 次迭代）变成了完整的 4 级软件流水线：

- **Prologue（2次预热）**：在进入稳态循环前，提前发出 K=0 和 K=1 两次异步 load，将数据提前送入 shared memory 槽位 0 和 1。
- **Steady-state（9次迭代）**：每次迭代中，consumer 通过 `ttg.async_wait {num=4}` 等待 4 个 in-flight 的异步拷贝中最老的那个完成（即当前迭代所需的数据），同时 producer 立即发出下一次迭代的 load。
- **Epilogue（排空）**：循环结束后，通过 `ttg.async_wait {num=0}` 等待所有 in-flight load 完成，然后执行最后几次 dot，并释放 shared memory（`ttg.local_dealloc`）。

`iter_args` 列表中同时传递了累加器张量、槽位索引（`acc_170..acc_174`）和异步 token（`matrix_x_175..matrix_w_180`），这是软件流水线状态机的标准表示。对于本卷积 kernel，`async_wait {num=4}` 中的 4 对应于 2 条数据流（激活 X + 权重 W）各 2 个 in-flight 请求，即 4 级流水线的 stage depth。
