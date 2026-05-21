# Pass 00：ConvertTritonToTritonGPU

> kernel：卷积 (Convolution) ｜ CLI：`convert-triton-to-tritongpu` ｜ 编译流水线第 00 个 Pass

## 这个 Pass 的作用

`ConvertTritonToTritonGPU` 是整个 Triton GPU 编译流水线的起点。它将与硬件无关的 `tt` 方言 IR（Triton IR，TTIR）转换为带有 GPU 线程布局注解的 `ttg` 方言 IR（TritonGPU IR，TTGIR）。该 Pass 的核心任务是根据编译参数（`num_warps=4`、`num_stages=4`、目标 `cuda:120`）为每个张量操作推断并附加 `#blocked` 布局属性，同时在 `module` 上写入 `ttg.num-warps`、`ttg.num-ctas`、`ttg.threads-per-warp` 等属性。此 Pass 之后，IR 进入 GPU 感知阶段，后续所有 Pass 均在带布局的 TTGIR 上运行。

## IR 变化

**变换前**（TTIR，无布局注解，module 无 GPU 属性）：

```mlir
module {
  tt.func public @triton_tem_fused_convolution_view_2(...) attributes {noinline = false} {
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x16xf32>
    %acc = scf.for %arg3 = %c0_i32 to %c9_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x64xf32>) : i32 {
      %matrix_x = tt.load %x_ptrs_56, %mask_x_74, %cst_6 : tensor<128x16x!tt.ptr<f32>>
      %acc_89 = tt.dot %matrix_x, %matrix_w, %arg4 : tensor<128x16xf32> * tensor<16x64xf32> -> tensor<128x64xf32>
    }
  }
}
```

**变换后**（TTGIR，每个张量均带 `#blocked` 布局，module 带 GPU 属性，并插入大量 `ttg.convert_layout`）：

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_tem_fused_convolution_view_2(...) {
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked2>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked3>
    %acc = scf.for ... iter_args(%arg4 = %cst_2) -> (tensor<128x64xf32, #blocked2>) : i32 {
      %matrix_x = tt.load %x_ptrs_81, %mask_x_112, %cst_6 : tensor<128x16x!tt.ptr<f32>, #blocked3>
      %matrix_x_133 = ttg.convert_layout %matrix_x : tensor<128x16xf32, #blocked3> -> tensor<128x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked6}>>
      %matrix_w_134 = ttg.convert_layout %matrix_w : tensor<16x64xf32, #blocked2> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked6}>>
      %19 = ttg.convert_layout %arg4 : tensor<128x64xf32, #blocked2> -> tensor<128x64xf32, #blocked6>
      %acc_135 = tt.dot %matrix_x_133, %matrix_w_134, %19 : tensor<128x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked6}>> * tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked6}>> -> tensor<128x64xf32, #blocked6>
      %20 = ttg.convert_layout %acc_135 : tensor<128x64xf32, #blocked6> -> tensor<128x64xf32, #blocked2>
    }
  }
}
```

## 说明

该 Pass 完成了从"平台无关描述"到"GPU 线程感知描述"的关键转换。Pass 为全部 128 个 warp 线程（4 个 warp × 32 线程/warp）推断了多套 `#blocked` 布局：

- **`#blocked1`**：1D 张量（如 `tensor<128xi32>`）采用行优先线程分布，32 线程/warp、4 warp，用于索引计算。
- **`#blocked2`**：2D 张量（如累加器 `tensor<128x64xf32>`）采用 `[1,32]` 的列优先 warp 分布，这是为后续 `tt.dot` 累加 layout 设计的。
- **`#blocked3`**：输入激活矩阵 `tensor<128x16xf32>` 用 `[2,16]` 分布，匹配 K 维（16）的访问。
- **`#blocked6`**（`sizePerThread=[4,4]`）：点积计算结果的 layout，每线程计算 4×4 个输出元素，充分利用 Blackwell 的寄存器并行。

由于各操作 layout 不统一，Pass 插入了大量 `ttg.convert_layout` 以确保跨 layout 的操作数对齐，这些 layout 转换在后续 Pass（如 `TritonGPUCoalesce`、`TritonGPURemoveLayoutConversions`）中会被逐步优化消除。
