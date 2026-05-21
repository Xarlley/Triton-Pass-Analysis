# Pass 20：TritonGPUAutomaticWarpSpecialization

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`tritongpu-automatic-warp-specialization` ｜ 编译流水线第 20 个 Pass

## 这个 Pass 的作用

TritonGPUAutomaticWarpSpecialization 分析带有 `loop.stage` 和 `loop.cluster` 属性的循环，尝试将不同 cluster 的指令分配给不同的 warp 组（warp group）来并行执行，即"warp 专用化"（warp specialization）。在 Blackwell sm_120 架构上，这对应于将加载 warp 和计算 warp 分离，加载 warp 专门执行预取，计算 warp 专门执行 MMA。

然而对于本 kernel，Pass 判断此 kernel 的结构（仅 `num_warps=2`，且依赖关系不支持完全分离）**不适合**自动 warp 专用化，因此 IR 实际上没有被修改——该 Pass 只是验证了分区的正确性（`VerifyWarpSpecializationPartitions`）并通过。after 文件中包含两段相同的 IR dump，分别来自 Pass 本身和内部验证步骤，这证实了 IR 内容未发生变化。

## IR 变化

该 Pass 的 before 与 after IR 内容完全相同（185 行 → 185 行）。核心循环结构保持不变：

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16x32xf32, #blocked>) {
  %a_41 = tt.load %a_40 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked1>
  %b_45 = tt.load %b_44 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked2>
  %a_46 = ttg.convert_layout %a_41 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  %b_47 = ttg.convert_layout %b_45 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  %acc_48 = tt.dot %a_46, %b_47, %arg4 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  scf.yield %acc_48 : tensor<16x32xf32, #blocked>
} {tt.scheduled_max_stage = 4 : i32}
```

## 说明

本 kernel 不进行 warp 专用化的原因：`num_warps=2` 在 sm_120 上配合 NVWS（NVIDIA Warp Specialization）需要至少 4 个 warp（通常为加载 warp + 计算 warp 各 2 组），而仅 2 个 warp 不足以有效分组。因此 Pass 通过内部 `VerifyWarpSpecializationPartitions` 验证后，确认不需要分区，IR 保持不变。

这与后续 Pass 31（第二次 TritonGPUScheduleLoops）处理的情况不同——在那个阶段，loop 的 stage 信息将被用于真正的软件流水线展开。此 Pass 对本 kernel 是一次验证性的 no-op，但对于更大 warp 数的 kernel（如 `num_warps=8`），同样的 Pass 会将循环分裂为多个 `nvws.warp_group` 区域。
