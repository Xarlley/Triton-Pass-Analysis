# Pass 26：CSEPass

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`cse` ｜ 编译流水线第 26 个 Pass

## 这个 Pass 的作用

CSEPass（公共子表达式消除，Common Subexpression Elimination）消除 IR 中重复计算的相同子表达式，以减少冗余指令。在本 kernel 的这个阶段，Pass 26 完成了两件事：

1. **消除 VerifyWarpSpecializationPartitions 产生的第二段 IR dump**：Pass 25 的 before 文件包含两段内容相同的 IR（原始 + VerifyWarpSpecializationPartitions 验证 dump），CSEPass 将其规范化为单段 IR，从 362 行减少到 179 行。
2. **消除真正的重复计算**：跨两段 dump 中存在的任何计算相同结果的重复指令，统一引用同一结果值。

经此 Pass 后，IR 从 362 行缩减至 179 行，成为后续所有 NVWS lowering Pass 和 Pipeline Pass 的稳定输入。

## IR 变化

before（362 行）是 SCCP 后的双段 dump；after（179 行）是干净的单段 IR，功能与 before 第一段完全相同：

**变换后的完整 IR 结构（after，179 行）：**

```mlir
#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8],  warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4],  warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:120", ...} {
  tt.func public @triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46(...) {
    %cst    = arith.constant dense<4>    : tensor<16x1xi32, #blocked>
    %cst_0  = arith.constant dense<4096> : tensor<1x32xi32, #blocked>
    %cst_1  = arith.constant dense<4096> : tensor<1x32xi32, #blocked1>
    %cst_2  = arith.constant dense<4096> : tensor<16x1xi32, #blocked>
    %cst_3  = arith.constant dense<4096> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_4  = arith.constant dense<4>   : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_5  = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #blocked2>
    ...（pid 计算、行列偏移计算、循环外不变量）...
    %acc = scf.for %arg3 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg4 = %cst_5) -> (tensor<16x32xf32, #blocked2>) : i32 {
      %a_k_idx_vals_33 = arith.muli %arg3, %c32_i32 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
      %a_k_idx_vals_34 = tt.splat %a_k_idx_vals_33 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32 -> tensor<1x32xi32, #blocked>
      %a_k_idx_vals_35 = arith.addi %a_k_idx_vals, %a_k_idx_vals_34 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x32xi32, #blocked>
      ...
      %a_41 = tt.load %a_40 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<16x32x!tt.ptr<f32>, #blocked>
      ...
      %b_45 = tt.load %b_44 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %a_46 = ttg.convert_layout %a_41 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : tensor<16x32xf32, #blocked> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
      %b_47 = ttg.convert_layout %b_45 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
      %acc_48 = tt.dot %a_46, %b_47, %arg4 {loop.cluster = 0 : i32, loop.stage = 4 : i32} : ... -> tensor<16x32xf32, #blocked2>
      scf.yield %acc_48 : tensor<16x32xf32, #blocked2>
    } {tt.scheduled_max_stage = 4 : i32}
    ...（store 结果、return）...
  }
}
```

## 说明

经过 Pass 25（SCCPPass）和 Pass 26（CSEPass），IR 经历了一次完整的"规范化-去重"周期：

- **SCCP**（Pass 25）：折叠了 `%width` 常量，统一了布局名称顺序（`#blocked`=A load，`#blocked1`=B load，`#blocked2`=dot accumulator）。
- **CSE**（Pass 26）：去除了 VerifyWarpSpecializationPartitions 产生的第二段完整 IR dump，将 IR 从 362 行降为 179 行。

此后，布局对应关系为：
- `#blocked`（`sizePerThread=[1,4], threadsPerWarp=[4,8], warpsPerCTA=[2,1]`）：A 矩阵加载布局，行方向连续，配合 A 的内存行主序。
- `#blocked1`（`sizePerThread=[4,1], threadsPerWarp=[8,4], warpsPerCTA=[1,2]`）：B 矩阵加载布局，列方向连续，配合 B 转置后的内存布局。
- `#blocked2`（`sizePerThread=[2,2], threadsPerWarp=[2,16], warpsPerCTA=[2,1]`）：dot 累加器布局，`scf.for` 的 iter_args 和 `tt.dot` 输出均使用此布局。

Pass 26 后，IR 进入稳定状态，将由 Pass 31（第二次 TritonGPUScheduleLoops，实际上等于 TritonGPUPipeline 的准备阶段）进行真正的软件流水线展开。
