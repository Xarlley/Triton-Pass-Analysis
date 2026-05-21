# Pass 59：AllocateSharedMemoryNv

> kernel：卷积 (Convolution) ｜ CLI：`allocate-shared-memory-nv` ｜ 编译流水线第 59 个 Pass

## 这个 Pass 的作用

`AllocateSharedMemoryNv`（NVIDIA shared memory 分配器）对 IR 中所有 `ttg.local_alloc` 操作执行静态内存布局分析，计算每个缓冲区在 shared memory 中的偏移量（byte offset），并将结果以 `{allocation.offset = N : i32}` 属性写入每个 `ttg.local_alloc` 操作。同时在 module 属性中记录 shared memory 总需求量 `ttg.shared = 36864`。对于需要 shared memory 才能执行 layout 转换的 `ttg.convert_layout` 操作，也会标注其临时 shared memory 用途的偏移量。IR 行数保持 408 不变。

## IR 变化

**在 module 属性中记录 shared memory 总量**：

```mlir
// 变换前
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {

// 变换后（新增 ttg.shared = 36864）
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 36864 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
```

**为 `ttg.local_alloc` 标注内存偏移量**：

```mlir
// 变换前（无偏移量属性）
%matrix_x = ttg.local_alloc : () -> !ttg.memdesc<3x128x16xf32, #shared, #smem, mutable>
%matrix_w = ttg.local_alloc : () -> !ttg.memdesc<3x16x64xf32, #shared1, #smem, mutable>

// 变换后（激活缓冲从偏移 0 开始，权重缓冲从偏移 24576 开始）
%matrix_x = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x16xf32, #shared, #smem, mutable>
%matrix_w = ttg.local_alloc {allocation.offset = 24576 : i32} : () -> !ttg.memdesc<3x16x64xf32, #shared1, #smem, mutable>
```

**为 `ttg.convert_layout` 标注临时 shared memory 偏移量**：

```mlir
// 变换前
%13 = ttg.convert_layout %0 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked3>

// 变换后（借用偏移 0 的临时空间用于 layout 转换）
%13 = ttg.convert_layout %0 {allocation.offset = 0 : i32} : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked3>
```

## 说明

Shared memory 的静态分配计算如下：

- **激活矩阵 X 的三重缓冲**：`3 × 128 × 16 × sizeof(f32) = 3 × 128 × 16 × 4 = 24576` 字节，从偏移 0 开始
- **权重矩阵 W 的三重缓冲**：`3 × 16 × 64 × sizeof(f32) = 3 × 16 × 64 × 4 = 12288` 字节，从偏移 24576 开始
- **总计**：`24576 + 12288 = 36864` 字节（约 36KB）

`ttg.shared = 36864` 将在 PTX 生成时被用于 `.shared .align 16 .b8 _kernel_smem[36864]` 的声明，这是 Triton 为 kernel 静态预留 shared memory 的标准方式。

`ttg.convert_layout {allocation.offset = 0}` 表明输出 store 阶段将 `#blocked`（`sizePerThread=[4,4]`, 128×64 layout）转换到 `#blocked3`（`sizePerThread=[1,4]`）需要经过 shared memory 中转，借用了 `%matrix_x` 的起始偏移量 0 处的空间（此时三重缓冲已不再需要，可以安全复用）。这是 shared memory 生命周期分析（liveness analysis）的典型结果，实现了 in-place 内存复用。
