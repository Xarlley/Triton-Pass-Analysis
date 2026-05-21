# Pass 38：TritonLoopAwareCSE

> kernel：卷积 (Convolution) ｜ CLI：`triton-loop-aware-cse` ｜ 编译流水线第 38 个 Pass

## 这个 Pass 的作用

`TritonLoopAwareCSE`（循环感知公共子表达式消除）在软件流水线展开后的 prologue 区域中识别并消除重复计算。Pass 32（Pipeline）在生成 prologue 时，为 K=0 和 K=1 两次预热 load 分别独立计算了激活矩阵 X 的行索引（`%idx_x_h`）和列索引（`%idx_x_w`），但由于 K=0 与 K=1 的 H 维度坐标完全相同（均来自同一个 `%idx_y_h + %cst`），这些计算实际上是冗余的。此 Pass 将 prologue K=1 中与 K=0 相同的子表达式替换为对已有结果的直接引用，IR 行数从 423 降至 404，净减少 19 行。

## IR 变化

**消除 prologue K=1 中与 K=0 相同的行索引计算**（删除 2 行）：

```mlir
// 变换前（prologue K=1 重复计算 %idx_x_h_89，与 K=0 的 %idx_x_h 完全相同）
%idx_x_h = arith.addi %idx_y_h, %cst : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>     // K=0（第 97 行）
// ... K=0 的 prologue 展开 ...
%idx_x_h_89 = arith.addi %idx_y_h, %cst : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>  // K=1 重复
%x_ptrs_90 = arith.muli %idx_x_h_89, %cst_12 : ...
%x_ptrs_91 = tt.expand_dims %x_ptrs_90 {axis = 1 : i32} : ... -> tensor<128x1xi32, #blocked1>
// ...（完整的 H 行索引展开链，共 3 行）
// mask 也重复：%mask_x_98..%mask_x_109（基于 %idx_x_h_89 的完整 H/W 边界检查链，12 行）

// 变换后（K=1 直接复用 K=0 的 %x_ptrs_60，%mask_x_72）
// 删除了 %idx_x_h_89、%x_ptrs_90、%x_ptrs_91（已重复）
// %x_ptrs_91（before）→ %x_ptrs_60（after，K=0 的现有结果，直接引用）
%x_ptrs_89 = arith.muli %idx_y_w, %cst_11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%x_ptrs_90 = tt.expand_dims %x_ptrs_89 {axis = 1 : i32} : ... -> tensor<128x1xi32, #blocked1>
%x_ptrs_91 = arith.addi %x_ptrs_60, %x_ptrs_90 : tensor<128x1xi32, #blocked1>  // 直接用 %x_ptrs_60
```

**消除 prologue K=1 中重复的 H 维度 mask 计算**（删除约 8 行）：

```mlir
// 变换前（K=1 独立计算与 K=0 完全相同的 H 维度边界 mask 链）
%mask_x_98 = arith.cmpi sge, %idx_x_h_89, %cst_8 : ...   // H >= 0
%mask_x_99 = tt.expand_dims %mask_x_98 {axis = 1} : ...
%mask_x_100 = arith.andi %mask_x_46, %mask_x_99 : ...
%mask_x_101 = arith.cmpi slt, %idx_x_h_89, %cst_16 : ... // H < 224
%mask_x_102 = tt.expand_dims %mask_x_101 {axis = 1} : ...
%mask_x_103 = arith.andi %mask_x_100, %mask_x_102 : ...
// 然后再拼接 W 维度 mask...

// 变换后（直接引用 K=0 已计算好的 %mask_x_72，省去 6 行 H 维度 mask 计算）
// %mask_x_72 = andi(andi(%mask_x_46, H>=0_mask), H<224_mask)  ← K=0 的结果
%mask_x_95 = arith.cmpi sge, %idx_y_w, %cst_8 : ...   // 只保留 W 维度 mask（这是不同的）
%mask_x_96 = tt.expand_dims %mask_x_95 {axis = 1} : ...
%mask_x_97 = arith.andi %mask_x_72, %mask_x_96 : ...   // 直接引用 %mask_x_72
```

**变化后的 scf.for iter_args 计数更新**（SSA 编号下移）：

```mlir
// 变换前
%acc_154:9 = scf.for %acc_169 = %c0_i32 to %c9_i32 step %c1_i32
  iter_args(..., %matrix_x_172 = %matrix_x_83, %matrix_x_173 = %matrix_x_114, %matrix_x_174 = %matrix_x_147, ...) ...

// 变换后（编号因删除 19 行而整体前移）
%acc_136:9 = scf.for %acc_151 = %c0_i32 to %c9_i32 step %c1_i32
  iter_args(..., %matrix_x_154 = %matrix_x_83, %matrix_x_155 = %matrix_x_105, %matrix_x_156 = %matrix_x_129, ...) ...
```

## 说明

在软件流水线的 prologue 中，每个预热迭代都需要计算下一次 load 的全局内存地址和边界 mask。对于本卷积 kernel，激活矩阵 X 的地址由两个分量决定：H 维度（行号，与卷积核行偏移有关）和 W 维度（列号，与卷积核列偏移有关）。由于 prologue K=0 和 K=1 处于同一个 kernel 行位置（3×3 卷积核的同一行，K 的步进只改变 W/channel 分量），H 维度分量 `%idx_x_h = %idx_y_h + %cst` 在两次 prologue 中完全相同，对应的 H 边界 mask（`h >= 0 && h < 224`）也完全相同。

`TritonLoopAwareCSE` 识别到这一跨 prologue-pipelined 块的重复性，将 K=1 prologue 中的 3 个 H 索引计算操作（`arith.addi`、`arith.muli`、`tt.expand_dims`）和 6 个 H 维度 mask 操作（`arith.cmpi` × 2、`tt.expand_dims` × 2、`arith.andi` × 2）替换为对 K=0 对应结果（`%x_ptrs_60` 和 `%mask_x_72`）的直接引用，合计删除约 11 个操作，节省 19 行 IR。这 19 行在 PTX 层面对应于约 19 条 SASS 指令，对于每次 kernel 启动只执行一次的 prologue 区域，减少的是确实的指令开销，而非仅仅是 IR 层面的冗余。

"循环感知"（Loop-Aware）体现在此 Pass 能够跨越 prologue 和 epilogue 的 region 边界进行 CSE，而普通 CSE（Pass 25）只在单个基本块内部工作。这一能力对软件流水线展开后产生的 prologue/epilogue 代码尤为重要，因为这些区域包含大量从循环体复制来的相同计算模式。
