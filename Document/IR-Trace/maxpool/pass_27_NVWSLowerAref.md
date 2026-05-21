# Pass 27：NVWSLowerAref

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`nvws-lower-aref` ｜ 编译流水线第 27 个 Pass

## 这个 Pass 的作用

NVWSLowerAref 负责将 NVWS（Nvidia Warp Specialization）框架中的抽象 `nvws.aref`（asynchronous reference）操作降级为具体的同步/屏障原语。在含 TMem/warp 专化的 kernel 中，`nvws.aref` 标记了 producer warp 和 consumer warp 之间的数据依赖点，此 Pass 将其转换为实际的 barrier 等待指令。

## IR 变化

本 kernel 没有任何 `nvws.aref` 操作（Pass 24 已确认 NVWSInsertTmemAref 为空操作），因此 NVWSLowerAref 对函数体没有任何修改。

然而，after.mlir 的行数从 134 **增至 271 行**，原因是此 Pass 再次追加了一份内部诊断 dump：

```
// -----// IR Dump Before (anonymous namespace)::VerifyWarpSpecializationPartitions () ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
...
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:120", ...} {
  tt.func public @triton_poi_fused_...() {
    %cst = arith.constant dense<64> : tensor<512xi32, #blocked>
    %cst_0 = arith.constant dense<112> : tensor<512xi32, #blocked>
    ...  // 与 before 完全相同的函数体（SCCP 规范化后的常量顺序）
  }
}
```

这是 NVWSLowerAref 在运行前触发的 `VerifyWarpSpecializationPartitions` 验证产生的调试快照，内容与 before 的实际 IR（第 1–134 行）完全相同。

## 说明

NVWSLowerAref 在本 kernel 上同样是空操作。after.mlir 中多出的 137 行诊断 dump 是 Triton NVWS 框架的验证机制：每次进入 NVWSLowerAref 之前，`VerifyWarpSpecializationPartitions` 会先验证当前 warp 专化分区状态的合法性，并将当前 IR 快照写入 dump。

这种"隐式 dump 膨胀"模式在 NVWS 相关 Pass（24、25、26、27、28）中反复出现，是 Blackwell sm_120 编译路径的特有行为——为了确保 warp 专化正确性，框架在关键节点多次验证和记录 IR 状态。对于不需要 warp 专化的 pointwise kernel，这些诊断记录是纯编译器内部开销，不影响最终生成的 PTX。

实际上，NVWSLowerAref 降级的目标操作（`nvws.aref`、`nvws.get_aref`、`nvws.put_aref`）在本 kernel 中一个都没有，Pass 的实质工作量为零。
