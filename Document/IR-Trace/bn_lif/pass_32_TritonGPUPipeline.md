# Pass 32：TritonGPUPipeline

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`tritongpu-pipeline` ｜ 编译流水线第 32 个 Pass

## 这个 Pass 的作用

TritonGPUPipeline 是软件流水线的实施 Pass，它基于 ScheduleLoops 生成的调度信息，将循环中的 load 操作转换为异步预取（async prefetch），并在循环迭代之间插入 barrier 以协调数据就绪。对于无循环的 kernel，Pipeline Pass 通常是 no-op，但仍会清理 ScheduleLoops 追加的内部 dump，将双份 IR 压缩回单份。

## IR 变化

本次变换将行数从 457 行压缩回 233 行，即**消除了 ScheduleLoops 追加的 SoftwarePipeliner 内部 dump**，同时将 ScheduleLoops 内部 dump 中匿名化的参数名、内联展开的布局注解全部恢复为标准形式：

**变换前（457 行，含内部 dump 的尾部片段）：**

```mlir
// 第一份（功能 IR）
tt.func public @triton_poi_fused_...(%in_out_ptr0: !tt.ptr<f32> ...) { ... }
#loc121 = loc("tmp43"(#loc55))

// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
module attributes {"ttg.num-ctas" = 1 : i32, ...} {
  tt.func public @triton_poi_fused_...(%arg0: !tt.ptr<f32> ...) {
    %cst_0 = arith.constant dense<9633792> : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 4], ...}>>
    ...（内联布局的匿名参数版本）...
  }
}
```

**变换后（233 行，单份功能 IR，别名布局，命名参数）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_poi_fused_...(%in_out_ptr0: !tt.ptr<f32> ..., %in_ptr0: !tt.ptr<f32> ...) {
    ...（与 Pass 28 after 相同的完整计算体）...
  }
}
```

## 说明

对于本 BN+LIF pointwise kernel，Pipeline Pass 判定**不需要软件流水线**：因为没有循环结构，无法创建传统的 prologue-mainloop-epilogue 流水线框架。4 条独立的 `tt.load` 指令（加载 4 个时间步的激活数据）在没有循环的情况下不能被转换为异步预取序列。

从硬件角度看，在 Blackwell sm_120 上，对于无循环的逐元素 kernel，load 延迟隐藏通过以下方式实现：
1. 编译器生成的指令调度（由 PTXAS 处理），将 load 指令尽可能提前发射；
2. 4 个 warp 之间的 latency hiding（warp interleaving）。

流水线变换不适用于本 kernel，但 Pipeline Pass 完成了重要的"清理"工作：将整个 NVWS 验证/调试工具链（Pass 20→27 产生的双份/多份 IR）最终收敛到干净的单份 IR，为后续的 TMem token 移除（Pass 36）和规范化（Pass 37）做好准备。
