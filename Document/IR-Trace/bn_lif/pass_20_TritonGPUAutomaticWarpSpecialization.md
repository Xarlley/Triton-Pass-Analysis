# Pass 20：TritonGPUAutomaticWarpSpecialization

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`tritongpu-automatic-warp-specialization` ｜ 编译流水线第 20 个 Pass

## 这个 Pass 的作用

TritonGPUAutomaticWarpSpecialization 尝试对 kernel 进行 warp 专业化分区（warp specialization partition），将 kernel 拆分为多个 warp group，每个 group 执行不同的工作（如 producer warp 负责数据加载，consumer warp 负责计算）。若 kernel 不满足 warp 专业化的前提条件（如无法从中分离出独立的 producer/consumer 子图），Pass 则不修改 IR，但仍会向输出文件中追加一份用于内部验证（VerifyWarpSpecializationPartitions）的 IR 副本。

## IR 变化

本 Pass 对实际 IR **未作任何功能性变换**，但输出文件行数从 233 行增至 469 行，原因是 Pass 在输出中追加了一份完整的 IR dump，标注头为：

```
// -----// IR Dump Before (anonymous namespace)::VerifyWarpSpecializationPartitions () ('builtin.module' operation) //----- //
```

变换前后两份 IR 的内容完全相同：

```mlir
// 两份 IR 均为相同的函数体，布局不变
#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", ...} {
  tt.func public @triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4(...) {
    ...
    %tmp0_30 = tt.load %tmp0_27, %tmp0_28 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>, #blocked>
    ...
  }
}
```

## 说明

对于本 BN+LIF kernel，自动 warp 专业化没有被激活，原因在于这是一个**逐元素（pointwise）kernel**，没有 matmul 或带有 producer-consumer 数据依赖的流水线结构，所有 warp 执行完全相同的工作。Warp 专业化通常用于 attention 或 GEMM 这类 kernel，在其中可以将加载 K/V 与计算 QK^T 分配给不同 warp group。

追加的 IR 副本是编译器内部完整性检查（VerifyWarpSpecializationPartitions）的输入，用于验证分区结果（此处结论为：无需分区）。后续的 `SCCPPass`（Pass 25）会处理这两份 IR，而 `CSEPass`（Pass 26）则会将重复的 IR 消除回单份。

这种"不分区但仍输出双份 IR"的行为是 Triton 编译框架调试工具链的产物——当 `TRITON_KERNEL_DUMP` 环境变量启用时，每个内部分析步骤都会将当时的 IR 写入捕获文件，因此行数翻倍。
