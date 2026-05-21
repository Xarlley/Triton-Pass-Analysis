# Pass 20：TritonGPUAutomaticWarpSpecialization

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`tritongpu-automatic-warp-specialization` ｜ 编译流水线第 20 个 Pass

## 这个 Pass 的作用

TritonGPUAutomaticWarpSpecialization 针对 Blackwell（sm_120）及更新架构，尝试将 kernel 中的操作自动拆分为多个 warp 组（warp group），令不同 warp 组专门执行不同任务（如数据搬运组 vs 计算组），以实现计算与访存的重叠流水。若 kernel 不满足专化条件（例如不含需要专化的 matmul loop），此 Pass 仅为后续 NVWS 系列 Pass 的验证留下内部 IR dump，不改变 kernel 实际逻辑。

## IR 变化

本 kernel 的 IR 内容完全未变，但 `after.mlir` 文件**行数从 133 增至 271 行**。这是因为 after 文件中追加了一份 MLIR 内部诊断 dump：

```
// -----// IR Dump Before (anonymous namespace)::VerifyWarpSpecializationPartitions () ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
...
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:120", ...} {
  tt.func public @triton_poi_fused_...() ... {
    // 与 before 完全相同的函数体
  }
}
```

这份额外 dump 是 Triton 的 `VerifyWarpSpecializationPartitions` 子 Pass 触发的诊断记录，表示该 Pass 已运行但判断此 kernel 无需 warp 专化分区（kernel 是纯 pointwise 逐元素操作，无 matmul tile loop，不满足 warp 专化的前提条件）。

## 说明

Warp Specialization 针对的场景是：同一个 CTA 内，部分 warp 专门做 TMA 异步搬运（producer），另一部分 warp 做矩阵计算（consumer），两者通过 `nvws.aref` / barrier 同步，隐藏内存延迟。本 MaxPool+BN+LIF kernel 是纯逐元素 pointwise kernel——4 次全局内存 load（MaxPool 2×2 窗口）+ 三级 max 比较 + 一次 store，没有任何矩阵乘法或需要跨 warp 协作的 reduction，因此 AutomaticWarpSpecialization 判定无需专化，原样放行。

after.mlir 中多出的 137 行仅是编译器内部调试记录（`IR Dump`），在 `VerifyWarpSpecializationPartitions` 验证后该状态不会进入下一 Pass 的输入。下一 Pass（NVWS 系列）接收的实际 IR 仍是 134 行的无专化版本。此行为是 Triton sm_120 编译路径的特有现象，体现了 Blackwell 后端对所有 kernel 都过一遍 warp 专化检查的策略。
