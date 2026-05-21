# Pass 28：NVWSAssignStagePhase

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`nvws-assign-stage-phase` ｜ 编译流水线第 28 个 Pass

## 这个 Pass 的作用

NVWSAssignStagePhase（NVIDIA Warp Specialization Assign Stage Phase）为 warp 专用化分区中的每个操作分配具体的「阶段相位」（stage phase）标记，用于协调加载 warp 和计算 warp 之间的同步点。在 Blackwell 的 NVWS 模型中，不同 warp 组执行不同 stage 的指令，stage phase 确保加载 warp 在正确时刻通知计算 warp 数据已就绪。

与 Pass 20、24、27 相同，本 kernel 因 `num_warps=2` 不满足 warp 专用化最低要求，Pass 28 对 IR 不产生任何实质性修改。after 文件 179 行，内容与 before 第一段（179 行）相同；before 文件 362 行（来自 Pass 27 after 的双 dump）。

## IR 变化

该 Pass 的 before 与 after IR 实质内容完全相同（均为 179 行有效 TTGIR，before 文件含 VerifyWarpSpecializationPartitions 追加的第二段 dump）。

核心数据流（无变化）：

```mlir
// 循环外（stage phase 在有分区时会在此插入 barrier/phase 初始化，但本 kernel 无此操作）
%acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst_5) -> (tensor<16x32xf32, #blocked2>) : i32 {
  // cluster 4 / stage 0：预取阶段（在有 stage phase 时，加载完成后会 signal）
  %a_41 = tt.load %a_40 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked>
  %b_45 = tt.load %b_44 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked1>
  // cluster 0 / stage 4：计算阶段（在有 stage phase 时，dot 开始前会 wait）
  %a_46 = ttg.convert_layout %a_41 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  %b_47 = ttg.convert_layout %b_45 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  %acc_48 = tt.dot %a_46, %b_47, %arg4 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ...
  scf.yield %acc_48 : tensor<16x32xf32, #blocked2>
} {tt.scheduled_max_stage = 4 : i32}
```

before 文件 362 行（含 VerifyWarpSpecializationPartitions 第二段 dump），after 文件 179 行（仅 IR 主体，无多余 dump）。

## 说明

在 warp 专用化生效的 kernel 中（`num_warps≥4`），NVWSAssignStagePhase 会在循环的每个 `loop.cluster`/`loop.stage` 边界处插入 `nvws.set_stage_phase` 和 `nvws.wait_stage_phase` 操作，形成加载 warp 和计算 warp 之间的握手协议：加载 warp 完成数据搬运后更新 phase，计算 warp 等待 phase 达到预期值后才开始 MMA。

对于本 kernel，由于 Passes 20/24/27/28 均判定不适合 warp 专用化，整个 NVWS pass 组（`nvws-*`）的四次 Pass 合计对 IR 产生的净变化仅为：Pass 25（SCCPPass）折叠了 `%width` 常量并规范化布局名称，Pass 26（CSEPass）去除了多余的 dump，其余均为 no-op。

Pass 28 是 NVWS pass 组的最后一个 Pass。之后 IR 将进入 Pass 31（第二次 TritonGPUScheduleLoops）开始软件流水线展开阶段。
