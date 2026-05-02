# CoalesceUtils.cpp 代码分析

## 简要概述
`CoalesceUtils.cpp` 是 TritonGPU 内存合并优化过程中的实用工具（Utility）文件。它不直接定义独立的 MLIR Pass，而是提供了一个供多个 Coalescing Pass（如 `CoalescePass` 和 `CoalesceAsyncCopyPass`）复用的核心函数：`buildCoalescedEncoding`。该函数用于根据张量的内存连续性属性，计算出最有利于内存合并的 `BlockedEncodingAttr`。

## 详细分析

### 1. 核心功能与目的
此文件的核心在于将前端分析得到的内存访问特征（尤其是哪一个维度在内存中是连续存放的）转换为后端具体的布局配置（Encoding）。正确的 Encoding 会决定 GPU 上的线程到数据元素的映射关系，使得连续的线程（如同一个 Warp 内的 32 个线程）分配到连续的内存地址上。

### 2. 核心函数解析：`buildCoalescedEncoding`
- **参数**: 接收轴分析信息（`ModuleAxisInfoAnalysis`），具体的访存操作（`op`），Warp 数量，每个 Warp 的线程数，以及每个 CTA（Thread Block）的形状。
- **获取内存顺序 (Order)**: 通过调用 `getOrderFromContiguity`，将张量的最连续的维度排在 Order 的最前面（通常是最内层维度，即 Order[0]）。
- **同质指针查找 (Multi-root Slice)**:
  - 算法会向后/向前遍历与当前指针关联的切片（Slice），寻找具有相同形状和相同内存连续性顺序的其他指针操作（`memAccessesSameOrder`）。
  - 这确保了如果有多个相关的读写操作，它们能够共同协商出一个全局最优的单线程处理元素个数（`perThread`）。
- **计算每线程处理量 (`perThread`)**:
  - 理想情况下，每个线程处理的连续元素个数越多，可以使用的宽指令（如 128-bit 向量加载）就越高效。
  - 算法取所有相关访存操作中最大的 `perThread`。
  - 并通过总元素数除以总线程数来进行裁剪（确保不会有线程空闲）。
  - 对于 Store 操作（非 Load），严格限制每线程最多处理 128-bit，因为过大的块会导致写入时存在空隙（Gaps），对 Store 的性能影响极为致命（Load 可以通过 L1 Cache 缓解）。
- **生成 Encoding**: 最终利用计算出的 `sizePerThread` 和 `order` 生成并返回 `BlockedEncodingAttr`，完成了从物理特性向逻辑 Layout 的映射。

### 4. 关键代码段分析

```cpp
// CoalesceUtils.cpp - buildCoalescedEncoding
BlockedEncodingAttr
buildCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       triton::gpu::CGAEncodingAttr cgaLayout,
                       SmallVector<int64_t> shapePerCTA) {
  // ... (省略前序步骤)
  auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
  SmallVector<unsigned> order = getOrderFromContiguity(contiguity);

  unsigned perThread =
      getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA);

  for (Operation *opSameOrder : memAccessesSameOrder) {
    if (opSameOrder == op)
      continue;
    unsigned currPerThread = getNumElementsPerThread(
        opSameOrder, order, axisInfoAnalysis, shapePerCTA);
    perThread = std::max(perThread, currPerThread);
  }

  perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
  // 对于非Load(Store)，控制单个线程最大处理128bits，防止写空隙
  if (!dyn_cast<triton::LoadOp>(op)) {
    perThread = std::min<int>(
        perThread,
        getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA));
  }
  SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
  sizePerThread[order[0]] = perThread;
  return BlockedEncodingAttr::get(op->getContext(), refTensorType.getShape(),
                                  sizePerThread, order, numWarps,
                                  threadsPerWarp, cgaLayout);
}
```
* **代码功能说明**: 通过从静态分析（`axisInfoAnalysis`）中获取目标张量在内存中的连续性，推算出最高效的线程分配顺序 `order`。随后根据同级别其他内存操作的需求，取最大的单个线程处理元素数 `perThread`。并且为了防止全局内存写入（Store）时出现不必要的写空隙，特别限制其不要超过硬件标量宽度的上限（通常是 128 bit）。
* **原理解析**: Blocked 布局的核心在于 `order` 和 `sizePerThread`。将最连续的维度放在 `order[0]` 可以让同一个 warp 的线程沿着内存地址连续分布。而提升 `sizePerThread` 会使得线程生成宽字节加载（例如 `ld.global.v4.b32`）直接读取 128-bit，这是 GPU 获得峰值内存吞吐的最有效途径。
* **在整个 PASS 中起到的作用**: 它是 Coalesce Pass 产生“智能”的核心计算器。无论有多少种内存指针变体（Read, Write, Async Load），都需要通过这个辅助函数生成硬件真正能够进行内存合并的最优 Layout。
