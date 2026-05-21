# Pass 59：AllocateSharedMemoryNv

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`allocate-shared-memory-nv` ｜ 编译流水线第 59 个 Pass

## 这个 Pass 的作用

AllocateSharedMemoryNv 负责分析 kernel 中所有需要 shared memory 的操作（如 tl.dot 引起的 shared memory 缓冲、layout 转换中的 smem 中转等），计算所需的 shared memory 字节数，并将结果以 `ttg.shared` 属性写入 module。此属性将决定最终 PTX 中 `.shared` 段的大小。

## IR 变化

此 Pass 仅修改了 **module 级别属性**，函数体内容不变。IR 行数保持 134 行不变。

**变化前（before）**：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
```

**变化后（after）**：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
```

新增属性：`ttg.shared = 0 : i32`。

## 说明

`ttg.shared = 0` 表示本 kernel 不需要任何 shared memory。这对于本 MaxPool+BN+LIF fusion kernel 是完全符合预期的：

1. **无 matmul / dot 操作**：无需 shared memory 缓存矩阵块；
2. **无 layout 转换中的 smem 中转**：所有操作统一在 `#blocked<sizePerThread=[2]>` 布局下运行，Pass 04（RemoveLayoutConversions）已消除所有跨布局转换；
3. **纯 pointwise 操作**：4 次 global load → 3 级 max 比较 → 1 次 global store，数据流完全在寄存器中（每线程 2 个 float 的寄存器文件）；
4. **无归约（reduction）**：无需 warp shuffle 或 smem 归约。

`ttg.shared = 0` 会最终体现在 PTX 中的全局 shared memory 符号：
```mlir
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
```
即 `global_smem` 是一个 0 字节的空数组，不占用任何 L1/shared memory 容量。这意味着本 kernel 的 occupancy 完全由寄存器使用量决定，而非受 shared memory 限制。
