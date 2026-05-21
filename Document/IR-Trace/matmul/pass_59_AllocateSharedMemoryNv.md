# Pass 59：AllocateSharedMemoryNv

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`allocate-shared-memory` ｜ 编译流水线第 59 个 Pass

## 这个 Pass 的作用

AllocateSharedMemoryNv 负责为模块内所有 `ttg.local_alloc` 操作分配共享内存（Shared Memory）地址偏移量，并将模块总共享内存用量记录到 `module attributes` 的 `ttg.shared` 属性中。该 Pass 是将逻辑共享内存分配（`ttg.local_alloc` 无地址版本）转换为带有具体物理偏移量的版本的关键步骤，为后续 LLVM lowering（Pass 63）正确生成 `@global_smem` 全局变量和 `llvm.getelementptr` 寻址指令提供基础。

对于本 kernel，两个环形缓冲区的布局如下：
- **B 矩阵缓冲区**（`!ttg.memdesc<4x32x32xf32, #shared1, ...>`）：4 个槽 × 32×32 × 4 字节 = **16384 字节**，分配在偏移 **0**（共享内存起始）
- **A 矩阵缓冲区**（`!ttg.memdesc<4x16x32xf32, #shared, ...>`）：4 个槽 × 16×32 × 4 字节 = **8192 字节**，分配在偏移 **16384**（紧接 B 缓冲区之后）

总共享内存 = 16384 + 8192 = **24576 字节**（24 KiB）。IR 行数保持 276 行不变，有 3 处变化：模块属性新增 `ttg.shared`，两个 `ttg.local_alloc` 各增加 `{allocation.offset}` 属性。

## IR 变化

**变化 1：`module attributes` 新增 `ttg.shared = 24576 : i32`**

```mlir
// 变换前：
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32,
  ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32,
  "ttg.total-num-warps" = 2 : i32} {

// 变换后：
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32,
  ttg.shared = 24576 : i32,                     ← 新增：总共享内存 24576 字节
  ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32,
  "ttg.total-num-warps" = 2 : i32} {
```

**变化 2：两个 `ttg.local_alloc` 添加 `allocation.offset` 属性**

```mlir
// 变换前（无偏移量）：
%a_31 = ttg.local_alloc : () -> !ttg.memdesc<4x16x32xf32, #shared, #smem, mutable> loc(#loc73)
%b_32 = ttg.local_alloc : () -> !ttg.memdesc<4x32x32xf32, #shared1, #smem, mutable> loc(#loc74)

// 变换后（带物理偏移量）：
%a_31 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<4x16x32xf32, #shared, #smem, mutable> loc(#loc73)
%b_32 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x32x32xf32, #shared1, #smem, mutable> loc(#loc74)
```

## 说明

共享内存布局遵循"大 buffer 在前"的分配策略：B 矩阵 buffer（更大，16384 字节）分配在偏移 0，A 矩阵 buffer（8192 字节）分配在偏移 16384。这样做的好处是地址对齐更简单——偏移 0 天然按最大对齐要求对齐。

`ttg.shared = 24576` 属性将在 Pass 63（ConvertTritonGPUToLLVM）中被用于生成：
```mlir
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
```
（共享内存大小由 CUDA 运行时在内核启动时动态分配，MLIR 全局变量声明为 0 元素数组作为占位符。）

`allocation.offset` 属性将在 Pass 63 中被用于生成 `llvm.getelementptr @global_smem[offset]` 指令，以计算每个 buffer 槽的实际共享内存地址。两个布局别名 `#shared`（A 矩阵，`order=[1,0]`）和 `#shared1`（B 矩阵，`order=[0,1]`）的 swizzle 参数均为 `vec=1, perPhase=1, maxPhase=1`，对于 f32 类型在此配置下不进行 bank conflict 优化的 swizzle。
