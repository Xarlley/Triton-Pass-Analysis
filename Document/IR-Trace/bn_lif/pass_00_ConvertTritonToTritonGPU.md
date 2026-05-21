# Pass 00：ConvertTritonToTritonGPU

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`convert-triton-to-tritongpu` ｜ 编译流水线第 0 个 Pass

## 这个 Pass 的作用

ConvertTritonToTritonGPU 是整个 GPU 编译流水线的入口 Pass，它将与硬件无关的 Triton IR（`tt.func`/`tt.` 方言）转换为带有 GPU 线程块布局信息的 Triton GPU IR（`ttg.` 方言）。Pass 的核心工作是：为模块添加 GPU 元信息属性（num-warps、num-ctas、threads-per-warp、target），并为每个 tensor 类型注入默认的 `#ttg.blocked` 布局，同时将 `tt.expand_dims` + `tt.broadcast` 等形状操作替换为先转换布局再操作的版本，以便后续 Pass 可以基于布局进行优化。

## IR 变化

**变换前（TTIR，无布局注解）：**

```mlir
module {
  tt.func public @triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4(...) {
    %cst = arith.constant dense<50176> : tensor<1x64xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %yindex = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %yindex_5 = tt.expand_dims %yindex {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %tmp0_24 = tt.load %tmp0_17, %tmp0_18 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>>
```

**变换后（TTGIR，含布局注解）：**

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4(...) {
    %cst = arith.constant dense<50176> : tensor<1x64xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked>
    %yindex = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked2>
    %yindex_5 = ttg.convert_layout %yindex : tensor<16xi32, #blocked2> -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %yindex_6 = tt.expand_dims %yindex_5 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi32, #blocked3>
    %yindex_7 = ttg.convert_layout %yindex_6 : tensor<16x1xi32, #blocked3> -> tensor<16x1xi32, #blocked1>
    %tmp0_24 = tt.load %tmp0_22, %tmp0_23 evictionPolicy = evict_last : tensor<16x64x!tt.ptr<f32>, #blocked>
```

模块属性也从无标注变为完整的 GPU 元信息：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32}
```

## 说明

这次变换完成了从硬件无关 IR 到 GPU 感知 IR 的关键跨越。对于本 BN+LIF kernel，有几点值得关注：

1. **默认布局选择**：Pass 为 `tensor<16x64xf32>` 选择了 `#blocked = {sizePerThread=[1,1], threadsPerWarp=[1,32], warpsPerCTA=[2,2], order=[1,0]}`，即每个线程只负责 1 个元素，128 个线程（4 warp × 32 线程）共同覆盖 16×64=1024 个元素。这是最保守的初始布局，留给后续 Coalesce Pass 优化。

2. **expand_dims 的处理**：原始 TTIR 中 `tt.make_range → tt.expand_dims` 是两步操作；转换后增加了 `ttg.convert_layout` 作为中间步骤，将 1D 的 `#blocked2` 布局转换为符合 `expand_dims` 期望的 `#ttg.slice` 布局。这是布局传播的必要准备。

3. **目标标注**：`ttg.target = "cuda:120"` 标记了 sm_120（Blackwell 架构），这影响后续 Pass 可以使用的硬件特性（如 TMA、tensor memory 等）。

4. **kernel 语义不变**：BN 归一化（`arith.mulf`、`arith.subf` 等）和 LIF 发放（`arith.cmpf oge`、`arith.uitofp`）的所有计算操作完全不变，仅添加了布局类型注解。行数从 219 增加到 230 行，增量完全来自布局别名定义和 `ttg.convert_layout` 插入。
