# Prefetch.cpp 代码分析

## 简要概述
`Prefetch.cpp` 实现了一个经典的编译器优化：软件预取（Software Pipelining / Prefetching）。该 Pass 针对 `scf.for` 循环内部的矩阵乘法（`tt.dot`）操作，将下一次迭代所需的寄存器加载（`local_load`）提前到当前迭代中执行。这有助于最大化 Tensor Core 计算与共享内存读取之间的指令重叠（Instruction Overlap），隐藏内存延迟。

## 详细分析

### 1. 核心功能与使用场景
在标准的内循环中，执行流程通常是：`等待共享内存就绪 -> 将共享内存数据加载到寄存器 -> 执行 Dot -> Yield`。
由于寄存器加载和 Dot 计算存在先后依赖，容易形成计算管线气泡。
`Prefetch` 算法将第一轮循环的加载提取到循环外部（Prologue），并在循环内部的 Dot 计算之后，立即发射对“下一轮”数据的加载指令。

### 2. 算法实现细节

#### 2.1 候选循环与指令提取
- 遍历包含 `triton.dot`（目前仅限 Nvidia MMA v2 和 AMD MFMA 编码）的 `scf.for` 循环。
- 通过追溯（`getPrefetchSrc`）找出 Dot 操作数从 Shared Memory 加载到寄存器的完整链路（主要是 `ttg.local_load` 操作）。
- 检查其输入是否带有异步 Token（`dot2aToken`），以确保预取不会破坏异步内存加载的内存一致性。

#### 2.2 序言发射 (Prologue Emission)
- 在循环外部（进入循环前），调用 `emitPrologue`。
- 它模拟第一次循环的行为，提取矩阵 A 和 B 首个 K 维度块（大小由 `prefetchWidth` 决定）的 `local_load`，将其放入初始化 `iter_args` 中。

#### 2.3 循环体重构 (Loop Body Reconstruction)
- 创建一个新的 `scf.for`，将预取到的 A 和 B 数据块作为循环的迭代参数（`iter_args`）传入。
- **拆分 Dot**: 如果 K 维度较大，算法支持将单个大的 `tt.dot` 拆解（Split）为多个小的 `tt.dot`。
- **预取下一迭代数据**: 算法利用 `getNextTrackedValue` 推算出下一次循环迭代的源数据地址和 Token，并在当前循环迭代结束前（`yield` 之前），生成加载下一次循环头数据的 `local_load`。
- 将这些新的 Load 结果 `yield` 出去，形成闭环。

### 3. 注意事项
该 Pass 仅对 `MMA v2` 和 `MFMA` 起效，因为新一代的架构（如 Hopper MMAv3 引入了 TMA 和 WGMMA 指令）有自己专属的异步流水线管理模式（如在 `MMAv5PipelineUtility` 中处理），不再使用这种纯软件的 `local_load` 寄存器预取策略。

### 4. 关键代码段分析

```cpp
// Prefetch.cpp - generatePrefetch
Value Prefetcher::generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   Value token, std::optional<int64_t> offsetK,
                                   std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int32_t> offset(rank, 0);
  Type elementType = type.getElementType();

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  offset[kIdx] = isPrologue ? 0 : prefetchWidth;
  shape[kIdx] = isPrologue ? prefetchWidth : (shape[kIdx] - prefetchWidth);

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  Value newSmem = triton::gpu::MemDescSubsliceOp::create(
      builder, v.getLoc(),
      triton::gpu::MemDescType::get(
          shape, elementType, type.getEncoding(), type.getMemorySpace(),
          type.getMutableMemory(), type.getAllocShape()),
      v, offset);

  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, prefetchWidth / 8);
  Value prefetchSlice = triton::gpu::LocalLoadOp::create(
      builder, v.getLoc(),
      RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem, token);

  return prefetchSlice;
}
```
* **代码功能说明**: 这个辅助函数负责生成预取操作所需的一小块矩阵切片的加载指令（`local_load`）。在被调用时，它通过传入的标志（如 `isPrologue` 和 `offsetK`），计算出应该从共享内存中读取的目标子切片（Subslice）的偏移量（`offset`）和大小（`shape`），然后生成 `MemDescSubsliceOp` 获取子切片视图，最后发射一条 `LocalLoadOp` 加载出正确的数据。
* **原理解析**: 预取算法的核心并不意味着把一整个块的数据全部都预先从共享内存搬迁出来，那样寄存器会吃不消。而是仅切分（Split）出满足一条最小的 MMA 指令所需的步长（`prefetchWidth`）。这种对共享内存视图进行“切片和预取”的技术是隐藏 Load/Use 延迟的基础。
* **在整个 PASS 中起到的作用**: 无论是循环前的序言（Prologue）阶段发射第一次加载，还是循环结尾处发射下一次迭代所需的数据，都需要依靠这个灵活的函数在正确的位置精确开辟加载。
