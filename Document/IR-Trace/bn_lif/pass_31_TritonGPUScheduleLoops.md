# Pass 31：TritonGPUScheduleLoops

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`tritongpu-schedule-loops` ｜ 编译流水线第 31 个 Pass

## 这个 Pass 的作用

TritonGPUScheduleLoops 是软件流水线（software pipelining）的调度准备 Pass，它分析 kernel 中的内存访问延迟，并将 load 操作标注为可异步执行（async）的候选，然后将 IR 转换为适合流水线的形式（如 LowerLoops）。对于无循环的 pointwise kernel，Pass 会输出其内部 IR dump（标注为 `SoftwarePipeliner internal IR Dump After: LowerLoops`）以供后续 Pipeline Pass 使用，行数从 233 行增至 457 行（追加了一份内部 dump）。

## IR 变化

本 Pass 追加了一份标注为 `SoftwarePipeliner internal IR Dump After: LowerLoops` 的内部 IR dump，该 dump 中函数参数名称从命名参数变为匿名参数（`%arg0` 等），布局注解从别名形式展开为内联形式：

**变换前（参数有命名，布局为别名）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
tt.func public @triton_poi_fused_...(%in_out_ptr0: !tt.ptr<f32> ..., %in_ptr0: !tt.ptr<f32> ...) {
    %cst_0 = arith.constant dense<9633792> : tensor<1x64xi32, #blocked1>
```

**变换后的内部 dump（参数匿名，布局内联展开）：**

```mlir
tt.func public @triton_poi_fused_(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    ...) {
    %cst_0 = arith.constant dense<9633792> : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
```

load 指令在内部 dump 中已展开为无 `evictionPolicy` 的简化形式：

```mlir
// 外部 IR（Pass 28 after）
%tmp0_30 = tt.load %tmp0_27, %tmp0_28 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>, #blocked1>

// 内部 dump（LowerLoops 后）
%30 = tt.load %27, %28 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], ...}>>
```

## 说明

ScheduleLoops 在本 BN+LIF pointwise kernel 上无法执行真正的软件流水线（因为没有循环），但仍然通过内部 `LowerLoops` 子程序输出了展平的调度 IR。这份内部 dump 是给 `TritonGPUPipeline`（Pass 32）消费的——Pipeline Pass 会检查这份 dump，判断 load 指令是否可以被异步化并提前发射。

对于本 kernel，所有 4 条 `tt.load` 指令（加载 4 个时间步的膜电位数据）本质上是相互独立的：它们加载不同地址（偏移量 0、3211264、6422528、9633792），计算也不相互依赖（每个时间步的 LIF 计算是独立的）。这为 Pass 32 的流水线决策提供了依据——尽管没有循环，4 条 load 也有可能被调度为重叠执行以隐藏 DRAM 延迟。

行数从 233 增至 457 行（增加 224 行），完全来自内部 dump 追加，功能 IR 未变。
