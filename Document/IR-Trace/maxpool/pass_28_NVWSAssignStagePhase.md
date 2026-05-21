# Pass 28：NVWSAssignStagePhase

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`nvws-assign-stage-phase` ｜ 编译流水线第 28 个 Pass

## 这个 Pass 的作用

NVWSAssignStagePhase 负责为 warp 专化流水线中的每个操作分配阶段（stage）和相位（phase）编号，使 producer/consumer warp 之间的数据传递能按正确的时序顺序执行。在无 warp 专化的 kernel 中，此 Pass 是空操作；但其运行后会触发 `VerifyWarpSpecializationPartitions` 验证，可能影响 IR dump 的行数。

## IR 变化

本 kernel 同样无需阶段/相位分配。after.mlir 的行数从 271 **降回 134 行**，这是因为 Pass 28 在完成（空）阶段分配后，`VerifyWarpSpecializationPartitions` 验证器清理了之前 NVWSLowerAref 遗留的额外诊断 dump，产出只包含实际 kernel IR 的单份文件。

before（271 行，含 NVWSLowerAref 追加的诊断 dump）到 after（134 行，纯净的单份 IR）：

**after 的有效内容（不变的函数体片段）**：
```mlir
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_poi_fused_...() {
    %cst = arith.constant dense<64> : tensor<512xi32, #blocked>
    %cst_0 = arith.constant dense<112> : tensor<512xi32, #blocked>
    %cst_1 = arith.constant dense<7168> : tensor<512xi32, #blocked>
    %cst_2 = arith.constant dense<128> : tensor<512xi32, #blocked>
    %cst_3 = arith.constant dense<28672> : tensor<512xi32, #blocked>
    %cst_4 = arith.constant dense<14336> : tensor<512xi32, #blocked>
    %cst_5 = arith.constant dense<14400> : tensor<512xi32, #blocked>
    ...
  }
}
```

函数体的逻辑内容与 Pass 25（SCCP）后完全相同，无任何 stage/phase 属性被插入（因为没有 warp 专化分区）。

## 说明

NVWSAssignStagePhase 的核心作用在矩阵 kernel 中体现：为 `nvws.warp_group` 内的每个 `tt.load`、WGMMA 等操作标记 `{stage = N, phase = K}` 属性，控制软件流水线的调度顺序。对于本 MaxPool+BN+LIF pointwise kernel，无任何 warp group 分区，Pass 直接放行。

从 271→134 的行数变化标志着 NVWS 相关的 Pass 组（20、24、25、26、27、28）全部结束。此后编译器进入 Pass 29（TritonGPUPartitionLoops）和 Pass 31（TritonGPUScheduleLoops），开始处理软件流水线调度——但对于本 kernel（无循环结构），这些 Pass 也将大部分是空操作或诊断行为，直到 Pass 36 才出现下一个有实质语义意义的变化（TritonNvidiaGPURemoveTMEMTokensPass）。
