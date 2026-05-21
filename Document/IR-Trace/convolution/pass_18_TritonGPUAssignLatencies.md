# Pass 18：TritonGPUAssignLatencies

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-assign-latencies` ｜ 编译流水线第 18 个 Pass

## 这个 Pass 的作用

`TritonGPUAssignLatencies` 为循环体内的内存操作（`tt.load`）标注延迟信息（`tt.latency` 属性），为后续的流水线调度（`TritonGPUScheduleLoops`、`TritonGPUPipeline`）提供数据。该属性告知调度器：某个 load 指令需要多少个流水线阶段才能完成，从而安排足够的预取（prefetch）距离。`tt.latency` 值直接来源于编译选项 `num_stages=4`：对 sm_120 上的全局内存访问，编译器估算需要 3 个中间阶段（latency = num_stages - 1 = 3）。

## IR 变化

本 Pass 只修改了两条 `tt.load` 指令，为其添加 `{tt.latency = 3 : i32}` 属性：

```mlir
// 变换前
%matrix_x = tt.load %x_ptrs_79, %mask_x_93, %cst_0 : tensor<128x16x!tt.ptr<f32>, #blocked1>
...
%matrix_w = tt.load %w_ptrs_101, %mask_w_53, %cst_1 : tensor<16x64x!tt.ptr<f32>, #blocked2>

// 变换后
%matrix_x = tt.load %x_ptrs_79, %mask_x_93, %cst_0 {tt.latency = 3 : i32} : tensor<128x16x!tt.ptr<f32>, #blocked1>
...
%matrix_w = tt.load %w_ptrs_101, %mask_w_53, %cst_1 {tt.latency = 3 : i32} : tensor<16x64x!tt.ptr<f32>, #blocked2>
```

IR 行数不变（292→292），仅增加两处属性注解。

## 说明

卷积 kernel 的 K 循环（9 次迭代）中，每次迭代需要从全局内存加载 128×16 的激活子矩阵（`matrix_x`）和 16×64 的权重子矩阵（`matrix_w`）。在 Blackwell sm_120 上，全局内存访问延迟约为数百个 GPU 时钟周期。通过标注 `tt.latency = 3`，下游的 `TritonGPUScheduleLoops`（Pass 19）知道需要在第 `k` 次使用数据时，提前 3 个阶段（即从第 `k-3` 次迭代开始）发出 load 请求，实现数据预取。`num_stages=4` 是用户通过 `torch.compile` max_autotune 搜索得到的最优值——4 级流水线（3 个 latency 阶段 + 1 个计算阶段）既能充分隐藏内存延迟，又不会因缓冲区过大而耗尽 shared memory。
