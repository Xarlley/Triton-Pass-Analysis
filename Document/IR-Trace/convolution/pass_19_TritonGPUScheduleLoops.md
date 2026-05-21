# Pass 19：TritonGPUScheduleLoops

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-schedule-loops` ｜ 编译流水线第 19 个 Pass

## 这个 Pass 的作用

`TritonGPUScheduleLoops`（第一次执行）根据 Pass 18 标注的延迟信息，为循环体内每条操作分配 `loop.stage`（流水线阶段编号）和 `loop.cluster`（操作分组，用于 Warp 特化）属性。这些属性描述了指令在多缓冲流水线中应出现在哪个阶段，是后续 `TritonGPUPipeline` Pass 将单循环展开为多缓冲结构的依据。此为第一次 ScheduleLoops（Pass 19），在 WarpSpecialization 之前；第二次（Pass 31）在 WarpSpecialization 之后。

## IR 变化

循环体内所有操作均被打上 `loop.stage` 和 `loop.cluster` 属性，形成明确的流水线时序图：

```mlir
// 变换前（无调度注解）
%i = arith.divsi %arg3, %c3_i32 : i32
%idx_x_h_68 = tt.splat %idx_x_h : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
...
%matrix_x = tt.load %x_ptrs_79, %mask_x_93, %cst_0 {tt.latency = 3 : i32} : tensor<128x16x!tt.ptr<f32>, #blocked1>
%matrix_w = tt.load %w_ptrs_101, %mask_w_53, %cst_1 {tt.latency = 3 : i32} : tensor<16x64x!tt.ptr<f32>, #blocked2>
%acc_104  = tt.dot %matrix_x_102, %matrix_w_103, %arg4 : ...

// 变换后（每条操作都有 loop.stage 和 loop.cluster）
%i = arith.divsi %arg3, %c3_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
%idx_x_h_68 = tt.splat %idx_x_h {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32 -> tensor<128xi32, ...>
...
%x_ptrs_79 = tt.addptr %x_ptrs_42, %x_ptrs_78 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : ...
%matrix_x  = tt.load %x_ptrs_79, %mask_x_93, %cst_0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.latency = 3 : i32} : tensor<128x16x!tt.ptr<f32>, #blocked1>
...
%matrix_w  = tt.load %w_ptrs_101, %mask_w_53, %cst_1 {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.latency = 3 : i32} : tensor<16x64x!tt.ptr<f32>, #blocked2>
...
%acc_104   = tt.dot %matrix_x_102, %matrix_w_103, %arg4 {loop.cluster = 3 : i32, loop.stage = 3 : i32} : ...
```

关键调度结论：
- 指针计算、load 操作：`loop.stage = 0`（第 0 阶段，尽早发出）
- `tt.dot` 计算：`loop.stage = 3`（第 3 阶段，等待 load 完成后执行）
- 所有操作均属于 `loop.cluster = 3`（单一 warp group，尚未进行 warp 特化分割）

## 说明

对于 `num_stages=4` 的 4 级流水线，调度器将循环展成 0→3 共 4 个流水阶段：Stage 0 发出内存预取，Stage 3 执行矩阵乘法（dot）。中间两个阶段（Stage 1、2）用于等待内存请求完成。这意味着当第 `k` 次迭代的 dot 在 Stage 3 执行时，第 `k+1`、`k+2`、`k+3` 次迭代的 load 已经分别在 Stage 0 发出——实现了 load 与 compute 的完全重叠（latency hiding）。`loop.cluster = 3` 此阶段为占位值，Pass 20（`TritonGPUAutomaticWarpSpecialization`）会根据此信息将 warp 分组并分配到不同角色（producer / consumer）。
