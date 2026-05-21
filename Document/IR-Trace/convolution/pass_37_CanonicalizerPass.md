# Pass 37：CanonicalizerPass

> kernel：卷积 (Convolution) ｜ CLI：`canonicalize` ｜ 编译流水线第 37 个 Pass

## 这个 Pass 的作用

`CanonicalizerPass`（规范化 Pass，第三次执行）在 Pass 36（RemoveTMEMTokens）之后清理遗留的冗余操作，主要包括：删除 `ub.poison` 占位 token、将 `ttg.local_load` 的输出 layout 从中间布局直接转化为 `ttg.dot_op` 布局（从而消除后续的 `ttg.convert_layout`），以及化简其他规范化模式。IR 行数从 426 降至 423，净减少 3 行。

## IR 变化

**删除 `ub.poison` 占位符**（1 行减少）：

```mlir
// 变换前
%0 = ub.poison : !ttg.async.token

// 变换后（删除）
（已删除）
```

**将 `ttg.local_load` 的输出直接对齐到 `dot_op` layout**（每处 2 行变 1 行）：

```mlir
// 变换前（两步：先 local_load 到 #blocked1，再 convert_layout 到 dot_op）
%matrix_x_184 = ttg.local_load %matrix_x_183 token %matrix_x_182
    : !ttg.memdesc<128x16xf32, #shared, #smem, mutable>
    -> tensor<128x16xf32, #blocked1>
%matrix_x_187 = ttg.convert_layout %matrix_x_184
    : tensor<128x16xf32, #blocked1>
    -> tensor<128x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>>

// 变换后（一步：local_load 直接输出 dot_op layout）
%matrix_x_184 = ttg.local_load %matrix_x_183 token %matrix_x_182
    : !ttg.memdesc<128x16xf32, #shared, #smem, mutable>
    -> tensor<128x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>>
```

同样地，权重矩阵 W 的 `ttg.local_load` 也从 `#blocked` → `dot_op<{opIdx=1}>` 的两步变为一步。

## 说明

Canonicalize Pass 在此处完成了两个重要的代码质量改进：

1. **清理 `ub.poison`**：Pass 36 插入的 `ub.poison` 是为了给 `iter_args` 中被删除的 token 提供形式上合法的初始值，但 canonicalize 确认这些值确实从未被使用（因为 epilogue 路径不会读取这些 token），故将其彻底删除。

2. **消除 `local_load → convert_layout` 链**：从 shared memory 读取数据后需要转换到 `ttg.dot_op` layout 才能喂给 `tt.dot`。规范化 Pass 发现 `ttg.local_load` 可以直接产出任意 layout（包括 `dot_op` layout），因此将两步操作合并为一步，省去了寄存器间的 layout 转换开销。对于 128×16 的激活矩阵，这相当于在寄存器层面节省了约 2048 次不必要的数据移动。
