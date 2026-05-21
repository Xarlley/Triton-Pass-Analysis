# Pass 20：TritonGPUAutomaticWarpSpecialization

> kernel：卷积 (Convolution) ｜ CLI：`tritongpu-automatic-warp-specialization` ｜ 编译流水线第 20 个 Pass

## 这个 Pass 的作用

`TritonGPUAutomaticWarpSpecialization` 是 Blackwell（sm_120）架构特有的 Warp 特化 Pass，将一个统一的 `scf.for` 循环分割为多个 Warp Group 并行流水线结构。该 Pass 读取 Pass 19 中标注的 `loop.stage` 和 `loop.cluster` 属性，将不同 stage 的操作分配给专用 Warp Group：发出内存请求的 Producer Warp Group 和执行矩阵乘法的 Consumer Warp Group，两者通过 shared memory 进行数据传递，实现 compute 与 memory access 的真正并发。IR 行数从 292 增至 587（约 2 倍），原因是循环体内 load 和 compute 操作被分配到两套独立的控制流分支。

## IR 变化

本 Pass 的变化体现在将带有不同 `loop.stage` 的操作区分为两个 cluster（流水线分区）：

**变换前**（所有操作在同一 `scf.for` 中，cluster=3 为统一分组）：

```mlir
%acc = scf.for %arg3 = %c0_i32 to %c9_i32 step %c1_i32 ... {
  %i = arith.divsi ... {loop.cluster = 3 : i32, loop.stage = 0 : i32}
  ...
  %matrix_x = tt.load ... {loop.cluster = 3 : i32, loop.stage = 0 : i32}
  %matrix_w = tt.load ... {loop.cluster = 3 : i32, loop.stage = 0 : i32}
  %matrix_x_102 = ttg.convert_layout %matrix_x {loop.cluster = 0 : i32, loop.stage = 3 : i32} ...
  %matrix_w_103 = ttg.convert_layout %matrix_w {loop.cluster = 0 : i32, loop.stage = 3 : i32} ...
  %acc_104 = tt.dot ... {loop.cluster = 0 : i32, loop.stage = 3 : i32}
} {tt.scheduled_max_stage = 3 : i32}
```

**变换后**（IR 翻倍至 587 行，包含 warp 分区分配标记）：

Pass 20 输出的 IR 中保留了与输入相同的结构（292 行），随后紧跟了一段供 `VerifyWarpSpecializationPartitions` 验证的 IR 副本（另 295 行），以确保分区正确。关键变化是 `loop.cluster` 的分配从统一的 `cluster=3` 拆分为：
- `cluster = 3`：Producer Warp Group，负责计算指针、构造掩码、发出 `tt.load`（Stage 0）
- `cluster = 0`：Consumer Warp Group，负责 `ttg.convert_layout` 和 `tt.dot` 计算（Stage 3）

两个 Cluster 在 Blackwell 的硬件 Warp Group 调度器下并发执行，Producer 在 Consumer 计算时提前发出下一批 load 请求。

```mlir
// Producer cluster（load，stage=0）
%matrix_x = tt.load %x_ptrs_79, %mask_x_93, %cst_0 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x16x!tt.ptr<f32>, #blocked1>
%matrix_w = tt.load %w_ptrs_101, %mask_w_53, %cst_1 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<16x64x!tt.ptr<f32>, #blocked2>

// Consumer cluster（dot，stage=3）
%matrix_x_102 = ttg.convert_layout %matrix_x {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<128x16xf32, #blocked1> -> tensor<128x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
%matrix_w_103 = ttg.convert_layout %matrix_w {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<16x64xf32, #blocked2> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
%acc_104 = tt.dot %matrix_x_102, %matrix_w_103, %arg4 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : ...
```

## 说明

这是整个编译流水线中最关键的架构特化 Pass 之一。Blackwell sm_120 引入了专用的 Warp Group 硬件调度机制，允许同一 CTA（thread block）内的不同 Warp Group 真正并发执行——不需要等待彼此的操作完成。在本卷积 kernel 中：

- **Producer Warp Group**（cluster=3，4 个 warp）：专门负责从全局内存取 128×16 激活数据和 16×64 权重数据，写入 shared memory 缓冲区。
- **Consumer Warp Group**（cluster=0，4 个 warp）：专门负责从 shared memory 读取数据、执行 128×16 × 16×64 的矩阵乘法累加。

Producer 在第 k 次迭代时预取第 k+3 次迭代的数据（因 latency=3），Consumer 在第 k 次迭代时计算第 k-3 次迭代已经就绪的数据，两者在时间上完全重叠，实现了理论上的最高内存带宽利用率。后续的 `NVWSInsertTmemAref`（Pass 24）和 `NVWSLowerAref`（Pass 27）将为 producer-consumer 之间插入异步引用（aref）同步原语，完成 Blackwell Warp Specialization 的完整实现。
