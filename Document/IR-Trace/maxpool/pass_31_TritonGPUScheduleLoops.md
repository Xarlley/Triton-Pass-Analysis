# Pass 31：TritonGPUScheduleLoops

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`tritongpu-schedule-loops` ｜ 编译流水线第 31 个 Pass

## 这个 Pass 的作用

TritonGPUScheduleLoops 负责为含循环的 kernel 生成软件流水线调度计划，通过分析 loop 内各操作的延迟（latency）和依赖关系，确定哪些操作可以提前发起（prefetch），从而将内存访问延迟与计算重叠。对于无循环的 pointwise kernel，此 Pass 会调用内部的 SoftwarePipeliner 但发现无可调度的循环，随后在 after.mlir 中附加内部 IR dump。

## IR 变化

Pass 31 将 after.mlir 的行数从 134 **增至 258 行**，原因是追加了 `// -----// SoftwarePipeliner internal IR Dump After: LowerLoops` 诊断块。

主要差异是此内部 dump 的函数签名中参数名称发生了变化（具名 `%in_ptr0`、`%out_ptr0`、`%xnumel` → 匿名 `%arg0`、`%arg1`、`%arg2`），且所有类型注解中的 `#blocked` 布局展开为内联形式：

**before（命名参数 + 引用 `#blocked`）**：
```mlir
tt.func public @triton_poi_fused_...(%in_ptr0: !tt.ptr<f32> ..., %out_ptr0: !tt.ptr<f32> ...) {
    %cst = arith.constant dense<64> : tensor<512xi32, #blocked>
```

**after 诊断 dump 中（匿名参数 + 展开布局）**：
```mlir
tt.func public @triton_poi_fused_...(%arg0: !tt.ptr<f32> ..., %arg1: !tt.ptr<f32> ..., %arg2: i32 ...) {
    %cst = arith.constant dense<64> : tensor<512xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>>
```

函数体中的逻辑计算内容（7 个常量、地址计算、4 次 load、3 组 max 比较、1 次 store）保持完全一致，仅格式有所不同。

## 说明

此诊断 dump 是 `SoftwarePipeliner` 子系统的 `LowerLoops` 步骤产出——对于无循环 kernel，`LowerLoops` 直接通过（no-op），并记录当前 IR 状态。布局类型从别名 `#blocked` 展开为完整的内联形式，是 SoftwarePipeliner 内部 IR 序列化风格的差异。

参数名从 `%in_ptr0`/`%out_ptr0`/`%xnumel` 变为 `%arg0`/`%arg1`/`%arg2` 是 SoftwarePipeliner 创建了一份内部 copy（去除外部属性注解）的结果，并不影响最终 Pass 32（Pipeline）的输入：Pipeline Pass 的实际输入仍是 before 中那份 134 行的规范 IR，诊断 dump 仅供人工审查。

对于 MaxPool+BN+LIF 这种单轮迭代 pointwise kernel，软件流水线没有发挥空间——所有 4 次 load 都在同一"波"中发出，计算链（3 级 max reduction）紧随其后，不需要跨迭代的指令重排。
