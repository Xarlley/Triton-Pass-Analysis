# Pass 28：NVWSAssignStagePhase

> kernel：卷积 (Convolution) ｜ CLI：`nvws-assign-stage-phase` ｜ 编译流水线第 28 个 Pass

## 这个 Pass 的作用

`NVWSAssignStagePhase`（NVIDIA Warp Specialization Assign Stage Phase）为每个 Warp Specialization 分区（partition）中的操作分配具体的流水线阶段（stage）和相位（phase）编号。在 Warp Specialization 架构中，producer 和 consumer 分区各自独立运行循环，需要通过"相位"概念来实现 N 缓冲（N-buffering）同步——producer 在 phase 0 写入槽位 0，在 phase 1 写入槽位 1，……；consumer 依据 phase 信息读取对应槽位。该 Pass 将 `loop.stage` / `loop.cluster` 注解转化为具体的分区内部 phase 编号，并删除已经用完的高层调度注解。IR 行数从 587 恢复为 292。

## IR 变化

Pass 28 的主要变化是将 Pass 27 中重新引入的验证副本（295 行）再次消除，并将 `loop.stage` / `loop.cluster` 注解中的信息提取为实际的分区相位标记（`phase` 属性），写入相关的 `nvws` 原语中。最终 IR 回到单副本 292 行结构。

核心变化模式：操作上的 `{loop.stage = N, loop.cluster = M}` 注解被消费（consumed）后删除：

```mlir
// 变换前（带有流水线调度注解）
%matrix_x = tt.load ... {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.latency = 3 : i32} : tensor<128x16x!tt.ptr<f32>, #blocked2>
%acc_104   = tt.dot ... {loop.cluster = 0 : i32, loop.stage = 3 : i32} : ...

// 变换后（注解被消费，分区 phase 信息写入 nvws barrier 原语，操作本身回归干净形式）
%matrix_x = tt.load ... : tensor<128x16x!tt.ptr<f32>, #blocked2>
%acc_104   = tt.dot ... : ...
```

`nvws.aref_get` / `nvws.aref_put` 等操作中写入了具体的 `phase = 0..3` 值，描述 4 个 shared memory 槽位的轮转方式。

## 说明

Pass 28 完成了 Warp Specialization 模型的最后一层"注解到实现"转化。经过 Pass 19（ScheduleLoops 标注）→ Pass 20（WarpSpecialization 分区）→ Pass 24（TmemAref 插入）→ Pass 27（LowerAref 具体化）→ Pass 28（AssignStagePhase 相位赋值）的完整链条，本卷积 kernel 的 K 循环已经被完整地转化为 4 级异步流水线结构：

- Phase 0：producer 加载 K=0 的数据到 shared memory 槽 0
- Phase 1：producer 加载 K=1 的数据到 shared memory 槽 1，consumer 开始计算 K=0 的 dot
- Phase 2、3：依此类推，producer 和 consumer 在时间上完全重叠

这解释了为何最终 PTX 中会看到大量的 `cp.async`（异步拷贝）和 `bar.arrive`（屏障到达）指令——它们正是此流水线结构的硬件实现。
