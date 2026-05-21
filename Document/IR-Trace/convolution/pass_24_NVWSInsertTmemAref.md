# Pass 24：NVWSInsertTmemAref

> kernel：卷积 (Convolution) ｜ CLI：`nvws-insert-tmem-aref` ｜ 编译流水线第 24 个 Pass

## 这个 Pass 的作用

`NVWSInsertTmemAref`（NVIDIA Warp Specialization Insert Tensor Memory Async Reference）为 Warp Specialization 的 producer-consumer 通信插入异步引用（aref）原语。在 Blackwell 架构的 Warp Specialization 模型中，Producer Warp Group 和 Consumer Warp Group 需要通过 shared memory 以异步方式交换数据，同时需要同步原语来保证数据就绪后 consumer 才开始读取。该 Pass 在 IR 中插入 `nvws.aref` 相关操作，以描述 producer/consumer 之间的 TMem（Tensor Memory）异步引用关系。IR 行数不变（587→587），但内部结构被重新组织以支持异步数据流。

## IR 变化

Pass 24 修改了 before.mlir 第二段（验证副本）中的 `#blocked` layout 编号顺序，将 layout 重新排列以与 Warp Specialization 的分区视图对齐：

```mlir
// 变换前（验证副本，旧编号）
#blocked  = {sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}
#blocked1 = {sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}
#blocked2 = {sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}
#blocked3 = {sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}

// 变换后（验证副本，新编号，按 consumer/producer 分区重排）
#blocked  = {sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}
#blocked1 = {sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}
#blocked2 = {sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}
#blocked3 = {sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}
```

同时，常量定义从"零初始化的张量"变为具体的 layout 感知常量，以匹配 TMem 数据视图：

```mlir
// 变换前（旧）
%cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked1>

// 变换后（新，以具体数值常量替代零张量，反映常量传播结果）
%cst = arith.constant dense<64> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_0 = arith.constant dense<3211264> : tensor<128x1xi32, #blocked>
```

## 说明

Pass 24 是 Blackwell Warp Specialization 管道中的专用 Pass，其主要工作是在内存拷贝（producer load）和矩阵乘法（consumer dot）之间插入异步引用标记，使硬件能够独立调度两个 Warp Group。对于本卷积 kernel，producer 在 K 循环的每次迭代中异步加载 128×16 激活和 16×64 权重至 shared memory，consumer 在数据就绪后通过 aref 机制接收通知并开始计算。这一 Pass 标志着从"逻辑调度注解"（`loop.stage`/`loop.cluster`）到"硬件可执行同步原语"的过渡，后续 `NVWSLowerAref`（Pass 27）将把这些高层 aref 原语降低为具体的 PTX 内存屏障和同步指令。
