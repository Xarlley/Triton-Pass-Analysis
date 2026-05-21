# Pass 24：NVWSInsertTmemAref

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`nvws-insert-tmem-aref` ｜ 编译流水线第 24 个 Pass

## 这个 Pass 的作用

NVWSInsertTmemAref（NVWS = NVIDIA Warp Specialization）是专为 Warp 专业化流水线中的 Tensor Memory（TMem）读写设计的 Pass。它为需要跨 warp group 传递 tensor memory 内容的操作插入 `nvws.aref` 抽象引用（abstract reference）节点，以便后续的 `NVWSLowerAref` Pass 将其转换为具体的同步原语。对于没有 TMem 相关操作的 kernel，Pass 基本为 no-op。

## IR 变化

本 Pass 对两份 IR（主 IR 和 VerifyWarpSpecializationPartitions 副本）分别处理，但均只发生了**布局别名顺序的重排**，功能 IR 不变：

```mlir
// 变换前（第二份 IR 中）
#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>

// 变换后（第二份 IR 中，名称互换）
#blocked  = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8],  warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

此外，常量池的排列顺序也相应改变（因为布局别名重命名导致常量的标注类型名变化），但数值内容和计算逻辑完全不变。

## 说明

布局别名重排是 MLIR Pass 基础设施的正常行为：当 Pass 遍历 IR 时，会重新为别名分配编号，按照实际使用顺序而非原有定义顺序排列。对本 BN+LIF kernel 而言，Pass 没有发现任何 TMem 相关操作（因为没有 warp 专业化，也没有 matmul tensor accumulator），故无实质变换。

两份 IR 的存在（469 行）是 Pass 20 遗留的，本 Pass 处理了这两份 IR 并都输出，行数维持 469 行不变。NVWSInsertTmemAref 真正发挥作用的场景是 Blackwell 架构的 GEMM kernel，其中 MMA accumulator 需要在 producer（数据加载 warp）和 consumer（计算 warp）之间通过 TMem 传递。
