# Pass 24：NVWSInsertTmemAref

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`nvws-insert-tmem-aref` ｜ 编译流水线第 24 个 Pass

## 这个 Pass 的作用

NVWSInsertTmemAref（NVIDIA Warp Specialization Insert Tensor Memory Aref）负责在已完成 warp 专用化分区的循环中，为 Tensor Memory（TMem，Blackwell sm_120 专属的片上张量存储）插入地址引用（`aref`）。该 Pass 会检测循环中的 `tt.load` → `tt.dot` 模式，若硬件支持且分区结构合适，则将 MMA 操作的累加器分配到 TMem，并插入相应的 `aref` 占位符。

然而对于本 kernel（`num_warps=2`），Pass 24 判断不满足 TMem 分配的最低要求（Blackwell TMem 通常需要至少 4 个 warp 才能有效利用），因此 IR 内容没有发生实质性变化。与 Pass 20（TritonGPUAutomaticWarpSpecialization）相同，该 Pass 的 after 文件包含两段 IR dump：第一段来自 NVWSInsertTmemAref 本身（IR 未变），第二段来自内部 `VerifyWarpSpecializationPartitions` 验证步骤。

## IR 变化

**第一段 dump（NVWSInsertTmemAref 本身，before → after 第一段）：IR 内容完全相同，185 行 → 185 行。**

核心循环结构保持不变：

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x32xf32, #blocked>)  : i32 {
  %a_k_idx_vals_33 = arith.muli %arg3, %c32_i32 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
  ...
  %a_41 = tt.load %a_40 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked1>
  ...
  %a_46 = ttg.convert_layout %a_41 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
  %b_47 = ttg.convert_layout %b_45 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : tensor<32x32xf32, #blocked2> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
  %acc_48 = tt.dot %a_46, %b_47, %arg4 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  scf.yield %acc_48 : tensor<16x32xf32, #blocked>
} {tt.scheduled_max_stage = 4 : i32}
```

**第二段 dump（VerifyWarpSpecializationPartitions 内部验证）：** after 文件第 188～367 行为同一 IR，但经过布局名称重排（`#blocked`→A load 布局，`#blocked1`→B load 布局，`#blocked2`→dot accumulator 布局），并将 `%width = arith.constant 1024 : i32` 折叠为 `%c1024_i32`：

```mlir
// 变换后常量声明（第二段 dump 中）：
%c1024_i32 = arith.constant 1024 : i32
...
%group_id = arith.divsi %pid, %c1024_i32 : i32
%pid_n    = arith.remsi  %pid, %c1024_i32 : i32
```

before 文件 372 行（来自 Pass 20 after 的双 dump），after 文件 367 行（第一段 185 行 + 第二段 182 行）。

## 说明

本 kernel 不进行 TMem 分配的原因：Blackwell TMem 是专为 warp 专用化场景设计的，需要至少将循环拆分为「加载 warp 组」和「计算 warp 组」两个分区，且 `num_warps` 需足够大（通常 ≥ 4）才能形成有效的加载/计算 warp 分组。`num_warps=2` 时，Pass 20（TritonGPUAutomaticWarpSpecialization）已判定不进行分区，因此 NVWSInsertTmemAref 也无法为 `tt.dot` 的累加器分配 TMem aref，IR 保持不变。

after 文件中第二段 IR dump 的常量折叠（`%width` → `%c1024_i32`）以及布局名称重排，是 `VerifyWarpSpecializationPartitions` 内部运行的局部 SCCP 优化和布局规范化产生的副产品，将由后续 Pass 25（SCCPPass）正式固化。
