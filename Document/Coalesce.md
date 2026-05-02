# Coalesce.cpp 代码分析

## 简要概述
`Coalesce.cpp` 实现了一个优化 Pass，其核心目标是优化 GPU 上的全局内存访问模式，最大化内存访问的合并（Memory Coalescing）。通过分析张量指针的连续性（Contiguity），该 Pass 重新规划线程与数据的映射关系（Layout Encoding），使得相邻的线程在物理上访问相邻的内存地址，从而大幅提高内存带宽利用率。

## 详细分析

### 1. 核心功能与目的
在 GPU 编程中，全局内存访问的性能极大依赖于合并（Coalescing）——即同一 Warp 内的 32 个线程应当尽量访问一段连续的内存块。如果使用不当的 Layout（如默认的 Blocked Layout），可能导致严重的内存访问碎片化。此 Pass 遍历所有的内存读写操作，并赋予它们有利于合并的自定义 Layout。

### 2. 主要组件与逻辑
- **`CoalescePass` 结构**: 继承自 `TritonGPUCoalesceBase`。在 `runOnOperation` 中，它首先运行 `ModuleAxisInfoAnalysis`（轴信息分析），以获取各个指针在各维度上的连续性和对齐情况。
- **轴信息分析 (Axis Info Analysis)**: 依靠前端计算出指针的 Contiguity（哪一个维度的内存是连续的）。例如，行主序矩阵在最内侧维度上连续。
- **Layout 重构 (`buildCoalescedEncoding`)**: 调用外部工具函数为特定的内存操作（如 Load/Store）创建一个新的 `BlockedEncodingAttr`。在这个新的 Encoding 中，连续的维度会被优先分配给连续的线程，确保访问被硬件合并。
- **描述符加载/存储 (`pickDescriptorLoadStoreLayout`)**: 针对使用 TMA（Tensor Memory Accelerator）或类似的描述符加载操作，该 Pass 也进行了布局调整。虽然 TMA 硬件负责合并，但目标共享内存的布局仍然会影响后续操作，因此允许对齐到最高 16 字节的向量化加载。

### 3. 转换流程
1. 为每个访存操作寻找最佳合并 Layout。
2. 将所有输入操作数通过 `ConvertLayoutOp` 转换为新的合并 Layout。
3. 执行访存操作（现在处于合并友好的 Layout 中）。
4. 将访存操作的结果再转换回原始所需的 Layout（如果后续计算有不同要求的话）。由于 Layout 转换大多发生在寄存器或共享内存之间，相对于全局内存访问的开销是可以接受甚至被消除的。

### 4. 关键代码段分析

```cpp
// Coalesce.cpp - CoalescePass::runOnOperation
void runOnOperation() override {
  ModuleOp moduleOp = getOperation();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  llvm::MapVector<Operation *, Attribute> layoutMap;
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(moduleOp);
  moduleOp.walk([&](Operation *curr) {
    Value ptr = getMemAccessPtr(curr);
    if (!ptr) return;
    // We only convert `tensor<tt.ptr<>>` load/store
    // ...
    int numWarps = lookupNumWarps(curr);
    auto tensorType = cast<RankedTensorType>(ptr.getType());
    CGAEncodingAttr cgaLayout = getCGALayout(tensorType.getEncoding());
    SmallVector<int64_t> shapePerCTA = getShapePerCTA(tensorType);
    auto layout =
        buildCoalescedEncoding(axisInfoAnalysis, curr, numWarps,
                               threadsPerWarp, cgaLayout, shapePerCTA);
    layoutMap[curr] = layout;
  });
  
  for (auto &kv : layoutMap) {
    convertDistributedOpEncoding(kv.second, kv.first);
  }
}
```
* **代码功能说明**: Pass的入口函数，首先执行了关于全局张量连续性特征的静态分析（`ModuleAxisInfoAnalysis`）。随后遍历所有内存访存操作，识别出需要执行优化的张量指针。并调用 `buildCoalescedEncoding` 获取理想的连续布局配置。最后一步对所有获取到理想布局的算子执行 `convertDistributedOpEncoding`。
* **原理解析**: 内存合并不只是修改单一加载/存储指令，而是需要在不破坏代码语义的前提下修改指针生成和操作数的布局。将操作数提前转换为`BlockedEncoding`布局可以确保到底层 PTX 的映射满足 128-bit 向量化访存的要求，从而不产生碎片化读取。
* **在整个 PASS 中起到的作用**: 这是整个 Coalescing 机制的调度器（Dispatcher）。它决定了“哪里需要合并访存”、“新的合并布局长什么样”以及“如何在原始的张量计算图中无缝注入这个布局变换”。
