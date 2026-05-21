# Pass 60：TritonTensorMemoryAllocationPass

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`triton-tensor-memory-allocation` ｜ 编译流水线第 60 个 Pass

## 这个 Pass 的作用

TritonTensorMemoryAllocationPass 负责分析 kernel 中所有 Tensor Memory（TMem，Blackwell sm_120 专属片上存储）的使用量，并将分配的字节数以 `ttg.tensor_memory_size` 属性写入 module。TMem 主要用于 WGMMA（Warpgroup Matrix-Multiply-Accumulate）操作的累加器存储，是 Blackwell 架构特有的高带宽片上内存层次。

## IR 变化

此 Pass 仅修改了 **module 级别属性**，函数体内容不变。IR 行数保持 134 行不变。

**变化前（before）**：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
```

**变化后（after）**：
```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
```

新增属性：`ttg.tensor_memory_size = 0 : i32`。

## 说明

`ttg.tensor_memory_size = 0` 表示本 kernel 不使用任何 Tensor Memory。这对 MaxPool+BN+LIF pointwise kernel 同样完全符合预期：

1. **无 WGMMA 操作**：TMem 主要用于 Blackwell 的 warpgroup 矩阵乘法累加器，本 kernel 无矩阵乘法；
2. **无 TMem load/store**：所有计算均通过普通 `ld.global` 从全局内存读取，结果写入 `st.global`；
3. **TMem 分配为 0**：不会在 PTX/SASS 中生成任何 `tcgen05.alloc` 或 TMem 管理指令。

至此，module 的三个资源属性均已确定：
- `ttg.shared = 0`（Pass 59）：无 shared memory 使用；
- `ttg.tensor_memory_size = 0`（Pass 60）：无 Tensor Memory 使用；
- `ttg.total-num-warps = 8`（Pass 56）：8 个计算 warp。

这三个属性共同描述了本 kernel 的 GPU 资源占用轮廓：一个纯寄存器计算的轻量级 pointwise kernel，仅使用全局内存和寄存器文件，不依赖任何片上高带宽存储（smem 或 tmem）。此特征使其 SM occupancy 最大化，同时对 cache 压力相对较低。
