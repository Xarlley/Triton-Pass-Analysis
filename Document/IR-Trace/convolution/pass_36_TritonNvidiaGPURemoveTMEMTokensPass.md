# Pass 36：TritonNvidiaGPURemoveTMEMTokensPass

> kernel：卷积 (Convolution) ｜ CLI：`triton-nvidia-gpu-remove-tmem-tokens` ｜ 编译流水线第 36 个 Pass

## 这个 Pass 的作用

`TritonNvidiaGPURemoveTMEMTokensPass` 清理 Tensor Memory 异步操作中用于同步的 token 值（`!ttg.async.token`）。在 Pass 32（Pipeline）中，`ttg.async_copy_global_to_local` 和 `ttg.async_commit_group` 等操作通过 token 链传递同步信息；当这些 token 不再被任何操作依赖时（即同步点已经固化为显式的 `ttg.async_wait`），该 Pass 将其删除，减少 `scf.for` 的 `iter_args` 数量，简化后续代码生成。IR 行数从 425 增至 426（增加了一行 `ub.poison` 占位）。

## IR 变化

Pass 36 的核心变化是将 `scf.for` 的 `iter_args` 中不再需要的 6 个 `!ttg.async.token` 参数进行清理，并在边界条件处插入 `ub.poison`（未定义行为占位符，表示该 token 值在首次迭代时未定义，但不会被使用）：

```mlir
// 变换前（iter_args 中携带 6 个 async.token）
%acc_154:12 = scf.for %acc_169 = %c0_i32 to %c9_i32 step %c1_i32
  iter_args(%arg4 = %cst_19, %acc_170 = %acc, %acc_171 = %acc_0,
            %acc_172 = %c3_i32, %acc_173 = %c3_i32, %acc_174 = %c3_i32,
            %matrix_x_175 = %matrix_x_83,   // async.token
            %matrix_x_176 = %matrix_x_114,  // async.token
            %matrix_x_177 = %matrix_x_147,  // async.token
            %matrix_w_178 = %matrix_w_88,   // async.token
            %matrix_w_179 = %matrix_w_120,  // async.token
            %matrix_w_180 = %matrix_w_153)  // async.token
  -> (tensor<128x64xf32, #blocked3>, i32, i32, i32, i32, i32,
      !ttg.async.token, !ttg.async.token, !ttg.async.token,
      !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {

// 变换后（增加 ub.poison，token 被整理）
%0 = ub.poison : !ttg.async.token  // 用于表示边界迭代中未定义的 token 初始值
%1 = arith.muli %idx_w, %cst_3 : tensor<128x1xi32, #blocked2>  // 原 %0 变为 %1
```

## 说明

软件流水线（Pass 32）在稳态循环中通过 `iter_args` 传递 async token 以维护 in-flight 异步操作的同步状态。然而在 Pass 36 时，编译器已经能够确认：这 6 个 token（3 个激活 X 的，3 个权重 W 的，对应三重缓冲的 3 个槽位）的同步信息已经被 `ttg.async_wait {num=4}` 操作显式捕获，token 链本身已无需在 `iter_args` 中维护。`ub.poison` 的插入（1 行新增，使 IR 从 425 行增至 426 行）是为了给 epilogue 阶段的最后几次计算提供一个形式上合法但语义上不使用的 token 初始值。这一清理使 Pass 37（Canonicalize）能够进一步化简循环参数。
