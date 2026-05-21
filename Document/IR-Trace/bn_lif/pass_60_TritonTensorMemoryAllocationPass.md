# Pass 60：TritonTensorMemoryAllocationPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`triton-tensor-memory-allocation` ｜ 编译流水线第 60 个 Pass

## 这个 Pass 的作用

TritonTensorMemoryAllocationPass 统计和分配 kernel 所需的 Tensor Memory（TMem）大小。TMem 是 Blackwell 架构（sm_120）上引入的一种新型片上内存，专用于 MMA（矩阵乘加）操作的 accumulator 存储，与共享内存（SMEM）独立。Pass 计算所有 `tt.tensor_memory` 类型操作所需的 TMem 大小，并将结果写入模块属性 `ttg.tensor_memory_size`。

## IR 变化

本次变换同样只修改模块属性，新增 `ttg.tensor_memory_size`：

**变换前：**

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32}
```

**变换后：**

```mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "cuda:120", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32}
```

新增属性：`ttg.tensor_memory_size = 0 : i32`

函数体内所有操作无变化，行数保持 233 行。

## 说明

`ttg.tensor_memory_size = 0` 表明本 kernel **不需要任何 Tensor Memory**，这与 BN+LIF 的 pointwise 性质完全吻合：

- TMem 专为 Blackwell 的 MMA 指令设计，只有矩阵乘法（`ttng.mma` 等）操作才会产生 TMem 分配需求。
- BN+LIF kernel 中只有逐元素算术（`arith.mulf`、`arith.cmpf`、`arith.addf` 等），没有任何矩阵运算，因此 TMem 大小为 0。
- 相比之下，在 VGG16-SNN 的卷积计算 kernel 中，若使用了 Blackwell 的 MMA 指令，`ttg.tensor_memory_size` 会是一个正整数（以 KB 为单位）。

`ttg.tensor_memory_size` 属性会被传递给 PTXAS 编译器，最终影响 PTX 中的 `.reqntid` 和内存资源声明，确保 GPU 驱动在调度 kernel 时预留足够的 TMem 配额（本 kernel 配额为 0，不影响 CTA 并发调度）。

这三个连续的元信息 Pass（56、59、60）分别记录了 total-num-warps=4、shared=4096、tensor_memory_size=0，共同构成了 kernel 的完整硬件资源声明。
