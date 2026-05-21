# Pass 32：TritonGPUPipeline

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`tritongpu-pipeline` ｜ 编译流水线第 32 个 Pass

## 这个 Pass 的作用

TritonGPUPipeline 实现软件流水线变换，将 loop 内的内存访问（尤其是 global memory load）提前到前一个迭代，通过 prefetch 隐藏内存延迟。对于含循环的 kernel，此 Pass 会引入 epilogue/prologue 代码并修改 loop 结构；对于无循环的 pointwise kernel，此 Pass 清理 Pass 31 留下的诊断 dump，使 IR 恢复到单份 134 行的纯净状态。

## IR 变化

Pass 32 将 after.mlir 的行数从 258 **降回 134 行**，清理掉了 Pass 31 追加的 `SoftwarePipeliner internal IR Dump After: LowerLoops` 诊断块。

**before（258 行，含内部 dump，展开布局注解）**：
```mlir
// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
...
tt.func public @triton_poi_fused_...(%arg0: !tt.ptr<f32> ..., %arg1: !tt.ptr<f32> ...) {
    %cst = arith.constant dense<64> : tensor<512xi32, #ttg.blocked<{sizePerThread = [2], ...}>>
    ...
}
```

**after（134 行，单份规范 IR，引用 `#blocked`）**：
```mlir
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
...
tt.func public @triton_poi_fused_...(%in_ptr0: !tt.ptr<f32> ..., %out_ptr0: !tt.ptr<f32> ...) {
    %cst = arith.constant dense<64> : tensor<512xi32, #blocked>
    %cst_0 = arith.constant dense<112> : tensor<512xi32, #blocked>
    ...
}
```

函数体逻辑与 Pass 25（SCCP）后完全相同，参数名恢复为具名形式（`%in_ptr0`、`%out_ptr0`、`%xnumel`）。

## 说明

对本 MaxPool+BN+LIF kernel，Pipeline Pass 的实质工作为零，清理行为是 Pass 框架的标准 teardown。这一对（31 增加 dump，32 清理 dump）的往返行为反映了 Triton 软件流水线框架的工作模式：ScheduleLoops 先收集调度信息并产出内部表示，Pipeline 随后依据该信息做实际变换并清理临时 IR。

在真正需要流水线的 matmul kernel 中，Pass 32 会将 before 中的 loop 拆分为 prologue + pipelined body + epilogue 三段，并插入 async load 和 barrier，使 GPU 的 load 单元在执行当前迭代计算的同时预取下一迭代数据。本 kernel 无此需求，因此 before 和 after 的 134 行有效内容完全一致。

此后编译流水线进入 Pass 36（TritonNvidiaGPURemoveTMEMTokensPass），开始处理 Blackwell 专属的 Tensor Memory token 清理工作。
