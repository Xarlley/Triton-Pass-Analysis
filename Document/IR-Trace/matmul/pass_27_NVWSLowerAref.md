# Pass 27：NVWSLowerAref

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`nvws-lower-aref` ｜ 编译流水线第 27 个 Pass

## 这个 Pass 的作用

NVWSLowerAref（NVIDIA Warp Specialization Lower Address Reference）将上一阶段（NVWSInsertTmemAref）可能插入的 TMem 地址引用（`aref`）操作降低（lower）为实际的 TMem 内存操作，并完成 warp 专用化分区相关的内存地址绑定工作。

对于本 kernel（`num_warps=2`），Pass 24 未插入任何 `aref` 操作，因此 Pass 27 同样是一次 no-op：IR 内容不变，仍为 179 行的干净单段 TTGIR。该 Pass 的 after 文件为 362 行，是因为 `VerifyWarpSpecializationPartitions` 再次被调用并追加了一段 IR dump，内容与 before 完全相同。

## IR 变化

before（179 行）与 after 第一段（179 行）内容完全相同，IR 结构无变化：

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst_5) -> (tensor<16x32xf32, #blocked2>)  : i32 {
  %a_k_idx_vals_33 = arith.muli %arg3, %c32_i32 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
  ...
  %a_41 = tt.load %a_40 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked>
  ...
  %a_46 = ttg.convert_layout %a_41 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : tensor<16x32xf32, #blocked> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
  %b_47 = ttg.convert_layout %b_45 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
  %acc_48 = tt.dot %a_46, %b_47, %arg4 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ... -> tensor<16x32xf32, #blocked2>
  scf.yield %acc_48 : tensor<16x32xf32, #blocked2>
} {tt.scheduled_max_stage = 4 : i32}
```

after 文件 362 行 = 第一段 179 行（NVWSLowerAref 本身，无变化）+ 第二段 183 行（`VerifyWarpSpecializationPartitions` 追加 dump，内容语义相同）。

## 说明

NVWSLowerAref 在有 aref 操作时会将其展开为：
- `nvws.tmem_alloc`：分配 Tensor Memory 缓冲区
- `nvws.tmem_store`/`nvws.tmem_load`：TMem 的读写操作
- 相关 barrier 插入

由于本 kernel 没有 aref，上述操作均未发生，IR 保持不变。Pass 27 之后，VerifyWarpSpecializationPartitions 的第二段 dump 将被 Pass 28（NVWSAssignStagePhase）的 after 文件中的 CSE 步骤清理。整个 NVWS（Passes 20, 24, 27, 28）对本 kernel 实际均为 no-op，仅在每次执行时追加 VerifyWarpSpecializationPartitions 验证 dump，确认分区正确性。
