# Pass 00：ConvertTritonToTritonGPU

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`convert-triton-to-tritongpu` ｜ 编译流水线第 0 个 Pass

## 这个 Pass 的作用

ConvertTritonToTritonGPU 是整个编译流水线的第一个 Pass，也是最重要的结构性转换之一。它将设备无关的 Triton IR（`tt.*` 方言）提升为携带 GPU 线程布局信息的 Triton GPU IR（`ttg.*` 方言）。具体来说，该 Pass 根据编译参数（`num_warps=2`，`threads_per_warp=32`）为所有张量类型附加 `#ttg.blocked` 编码属性，同时向 `module` 添加 `ttg.num-warps`、`ttg.num-ctas`、`ttg.target` 等全局属性。此外，由于 `tt.dot` 的操作数需要特定的 dot_op 编码，Pass 会在 dot 之前插入 `ttg.convert_layout` 指令，将操作数从通用 blocked 布局转换为 `ttg.dot_op` 布局。

## IR 变化

**变换前（Triton IR，无布局信息）：**

```mlir
module {
  tt.func public @triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46(...) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
    %cst_1 = arith.constant dense<4096> : tensor<1x32xi32>
    %cst_2 = arith.constant dense<4096> : tensor<16x1xi32>
    ...
    %rm_9 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    ...
    %acc = scf.for ... iter_args(%arg4 = %cst_0) -> (tensor<16x32xf32>) {
      ...
      %a_33 = tt.load %a_32 : tensor<16x32x!tt.ptr<f32>>
      %b_40 = tt.load %b_39 : tensor<32x32x!tt.ptr<f32>>
      %acc_41 = tt.dot %a_33, %b_40, %arg4 : tensor<16x32xf32> * tensor<32x32xf32> -> tensor<16x32xf32>
      scf.yield %acc_41 : tensor<16x32xf32>
    }
```

**变换后（Triton GPU IR，附加布局编码）：**

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46(...) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #blocked1>
    %cst_1 = arith.constant dense<4096> : tensor<1x32xi32, #blocked1>
    %cst_2 = arith.constant dense<4096> : tensor<16x1xi32, #blocked>
    ...
    %rm_9 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked2>
    ...
    %acc = scf.for ... iter_args(%arg4 = %cst_0) -> (tensor<16x32xf32, #blocked1>) {
      ...
      %a_46 = tt.load %a_45 : tensor<16x32x!tt.ptr<f32>, #blocked1>
      %b_56 = tt.load %b_55 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %a_57 = ttg.convert_layout %a_46 : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>>
      %b_58 = ttg.convert_layout %b_56 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>>
      %4 = ttg.convert_layout %arg4 : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #blocked5>
      %acc_59 = tt.dot %a_57, %b_58, %4 : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked5}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked5}>> -> tensor<16x32xf32, #blocked5>
      %5 = ttg.convert_layout %acc_59 : tensor<16x32xf32, #blocked5> -> tensor<16x32xf32, #blocked1>
      scf.yield %5 : tensor<16x32xf32, #blocked1>
    }
```

## 说明

Pass 为所有张量分配了 6 种不同的 blocked 布局（`#blocked` 到 `#blocked5`），分别对应矩阵 A（行优先加载）、矩阵 B（列优先加载）、累加器（dot 结果）等不同角色。`num_warps=2`、`threads_per_warp=32` 的配置体现在每种布局的 `warpsPerCTA` 和 `threadsPerWarp` 字段中。

对于这个 VGG16 SNN 全连接 kernel（输入形状 `1×4096`，权重形状 `4096×4096`），矩阵 A（激活）和矩阵 B（权重）的加载被分配了不同的布局：A 使用 `#blocked1`（行优先，每线程 `[1,1]`，warp 内 1×32 布局），B 的加载也使用 `#blocked1`。在进入 `tt.dot` 之前，Pass 插入了三个 `ttg.convert_layout` 将 A、B 和累加器转换为 `#blocked5`（`sizePerThread=[2,2]`，适合 MMA 的分块模式）所需的 `dot_op` 编码，并在 `tt.dot` 之后将结果转回 `#blocked1` 以便 `scf.yield`。此刻这些 layout conversion 并非最优，后续 Pass（如 Coalesce、RemoveLayoutConversions）会进一步优化。
