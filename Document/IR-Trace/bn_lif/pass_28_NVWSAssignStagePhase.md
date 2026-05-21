# Pass 28：NVWSAssignStagePhase

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`nvws-assign-stage-phase` ｜ 编译流水线第 28 个 Pass

## 这个 Pass 的作用

NVWSAssignStagePhase 是 Warp 专业化流水线（NVWS）的阶段分配 Pass。在多阶段（multi-stage）的软件流水线中，每个操作需要被标记为属于哪个 pipeline stage 和 phase（用于同步 barrier 的相位控制）。对于已经完成 warp 专业化分区的 kernel，Pass 会为每个操作附加 `stage` 和 `phase` 属性；对于未分区的 kernel，Pass 会消除 LowerAref 产生的验证副本，回到单份 IR。

## IR 变化

本次变换使行数从 469 行回到 233 行，即**消除了 Pass 27 追加的第二份 IR 副本**：

**变换前（469 行，双份 IR）：**

```mlir
// 第一份（功能 IR）
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", ...} {
  tt.func public @triton_poi_fused_...(...) {
    ...（完整计算体）...
  }
}
#loc121 = loc("tmp43"(#loc55))


// -----// IR Dump Before VerifyWarpSpecializationPartitions //----- //
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", ...} {
  tt.func public @triton_poi_fused_...(...) {
    ...（相同的计算体副本）...
  }
}
```

**变换后（233 行，单份 IR）：**

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_poi_fused_...(...) {
    ...（单份完整计算体，无任何 stage/phase 标注）...
  }
}
```

BN+LIF 的计算指令（`arith.mulf`、`arith.cmpf`、`tt.load`、`tt.store` 等）均无变化，也未添加任何 stage 属性。

## 说明

NVWSAssignStagePhase 对本 kernel 的唯一实质效果是**清理双份 IR**。Pass 判断该 kernel 未经 warp 专业化分区，因此没有任何 stage/phase 需要分配，验证副本的存在已无意义，直接丢弃。

在真正被 warp 专业化的 GEMM kernel 中，此 Pass 会为 producer warp 中的 `tt.load` 标注 `stage=0, phase=0`，为 consumer warp 中的 MMA 操作标注 `stage=1, phase=1`，配合 barrier 机制实现 ping-pong buffering。对于逐元素的 BN+LIF kernel，所有 4 个 warp 在完全相同的 stage 执行完全相同的操作，无法也无需进行 stage 分配。

经过 Pass 28 后，流水线进入 TritonGPUPartitionLoops（Pass 29）和 NVWSLowerWarpGroup（Pass 30）阶段，这两个 Pass 对本 kernel 同样是 no-op，之后进入 TritonGPUScheduleLoops（Pass 31）开始软件流水线调度。
