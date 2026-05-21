# Pass 19：TritonGPUScheduleLoops

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-schedule-loops` ｜ 编译流水线第 19 个 Pass

## 这个 Pass 的作用

TritonGPUScheduleLoops 根据 Pass 18 标注的延迟信息（`tt.latency`），对循环体内的每条指令分配流水线 stage（`loop.stage`）和 cluster（`loop.cluster`）属性。这些属性指定了每条指令应在软件流水线的哪个 stage 执行，以及属于哪个依赖 cluster，为后续的 TritonGPUPipeline Pass（负责真正展开循环并插入异步加载）提供调度信息。`loop.stage` 的最大值被记录为 `tt.scheduled_max_stage` 属性附加到 `scf.for` 上。

## IR 变化

**关键变化：** 循环体内每条指令被添加了 `{loop.cluster = N : i32, loop.stage = M : i32}` 属性；`scf.for` 被添加了 `{tt.scheduled_max_stage = 4 : i32}`。

**变换前（循环体内指令无调度属性）：**

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x32xf32, #blocked>) {
  %a_k_idx_vals_33 = arith.muli %arg3, %c32_i32 : i32
  %a_k_idx_vals_34 = tt.splat %a_k_idx_vals_33 : i32 -> tensor<1x32xi32, #blocked1>
  %a_k_idx_vals_35 = arith.addi %a_k_idx_vals, %a_k_idx_vals_34 : tensor<1x32xi32, #blocked1>
  ...
  %a_41 = tt.load %a_40 {tt.latency = 4 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked1>
  ...
  %a_46 = ttg.convert_layout %a_41 : ... -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
  %acc_48 = tt.dot %a_46, %b_47, %arg4 : ...
  scf.yield %acc_48 : tensor<16x32xf32, #blocked>
} loc(#loc78)
```

**变换后（每条指令带 `loop.stage` / `loop.cluster`，`scf.for` 带 `tt.scheduled_max_stage`）：**

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x32xf32, #blocked>) {
  %a_k_idx_vals_33 = arith.muli %arg3, %c32_i32 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
  %a_k_idx_vals_34 = tt.splat %a_k_idx_vals_33 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32 -> tensor<1x32xi32, #blocked1>
  %a_k_idx_vals_35 = arith.addi %a_k_idx_vals, %a_k_idx_vals_34 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x32xi32, #blocked1>
  %b_k_idx_vals_36 = tt.splat %a_k_idx_vals_33 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32 -> tensor<32x1xi32, #blocked2>
  %b_k_idx_vals_37 = arith.addi %b_k_idx_vals_19, %b_k_idx_vals_36 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<32x1xi32, #blocked2>
  %xindex_38 = tt.broadcast %a_k_idx_vals_35 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : ...
  %xindex_39 = arith.addi %xindex_38, %xindex_20 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : ...
  %a_40 = tt.addptr %a, %xindex_39 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : ...
  %a_41 = tt.load %a_40 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked1>
  ...
  %a_46 = ttg.convert_layout %a_41 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  %b_47 = ttg.convert_layout %b_45 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  %acc_48 = tt.dot %a_46, %b_47, %arg4 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  scf.yield %acc_48 : tensor<16x32xf32, #blocked>
} {tt.scheduled_max_stage = 4 : i32} loc(#loc78)
```

## 说明

调度结果将循环指令划分为两个 stage 分组：

- **stage 0 / cluster 4**（预取阶段）：包含地址计算（K 偏移乘法、加法、broadcast）和 `tt.load`。这些指令在迭代 i 执行时，实际加载的是 i+4 次迭代所需的数据，即提前 4 个迭代预取。
- **stage 4 / cluster 0**（消费阶段）：包含 `ttg.convert_layout`（将加载数据转为 dot_op 布局）和 `tt.dot`（MMA 矩阵乘法）。这些指令使用 stage 0 在 4 次迭代前发出的加载结果。

`tt.scheduled_max_stage = 4` 记录了流水线深度，意味着在循环展开后，将有一个 4 次迭代的 prologue（纯预取，无 dot）和 1 次迭代的 epilogue（纯消费，无新预取）。这个 5 阶段流水线与 `num_stages=5` 的编译参数完全对应，对 K=4096 的 128 次迭代能最大限度隐藏全局内存延迟。
