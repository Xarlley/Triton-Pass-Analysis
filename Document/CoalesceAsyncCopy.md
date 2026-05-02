# CoalesceAsyncCopy.cpp 代码分析

## 简要概述
`CoalesceAsyncCopy.cpp` 是 TritonGPU 中的一个专门针对异步拷贝操作（`cp.async`，即将数据从全局内存直接拷贝至共享内存，绕过寄存器）的优化 Pass。它通过调整异步拷贝操作数的线程级块大小（`sizePerThread`），来确保底层生成的 PTX `cp.async` 指令能够产生最少量的内存事务，从而避免破坏合并访存（Coalescing）。

## 详细分析

### 1. 核心功能与目的
NVIDIA GPU 提供了 `cp.async` 指令以实现高效的异步全局内存到共享内存拷贝。然而，如果全局内存的 `BlockedEncoding` 在连续维度上的 `sizePerThread`（每个线程处理的大小）大于共享内存 `SwizzledSharedEncoding` 的向量大小（Vector Size），拷贝操作会被拆分成多条较小的 `cp.async` 指令，这会导致每个线程加载多个非连续的数据块，从而破坏了原本可以合并的全局内存访问。该 Pass 专门检测并修复这种情况。

### 2. 主要组件与逻辑
- **`ClipAsyncCopySizePerThread`**:
  - 该 Rewrite Pattern 寻找 `AsyncCopyGlobalToLocalOp`。
  - 它计算源张量（全局内存）的线性布局和目标张量（共享内存）的线性布局。
  - 推导出最大连续拷贝大小（`copyContigSize`）。
  - 如果发现源张量在连续维度上的 `sizePerThread` 大于 `copyContigSize`，则将源张量的 `sizePerThread` 裁剪（Clip）至 `copyContigSize`。
  - 通过重新分配 Layout，确保底层只会生成最大宽度的向量化拷贝指令，避免不必要的指令分裂。
- **`CoalesceCheapAsyncCopyGlobalToLocal`**:
  - 针对体积较小的（Cheap）的拷贝操作。
  - 当拷贝的数据量较小，以至于无法占满所有线程，或者数据位宽小于 32-bit 时，系统原本通常会根据消费者的计算 Layout 决定拷贝 Layout。
  - 该 Pattern 强制为这些拷贝分配基于 `AxisInfoAnalysis` 推导出的最佳内存合并 Layout，因为拷贝操作的 Layout 应该独立于消费者计算，以访存性能为第一优先级。

### 3. 工作原理
该 Pass 使用 `ModuleAxisInfoAnalysis` 来获取掩码（Mask）的对齐信息和张量的连续性，然后更新拷贝指令的 `contiguity` 属性。并在必要时在源数据、掩码和其他操作数之前插入 `ConvertLayoutOp`，将它们重定向至具有正确 `sizePerThread` 配置的新 Encoding。

### 4. 关键代码段分析

```cpp
// CoalesceAsyncCopy.cpp - ClipAsyncCopySizePerThread::matchAndRewrite
LogicalResult matchAndRewrite(AsyncCopyGlobalToLocalOp copyOp,
                              PatternRewriter &rewriter) const override {
  // ... (省略类型获取)
  // obtain max contiguous copy size
  LinearLayout regLayout = triton::gpu::toLinearLayout(srcTy);
  LinearLayout sharedLayout = triton::gpu::toLinearLayout(dstTy);
  auto copyContigSize =
      regLayout.invertAndCompose(sharedLayout).getNumConsecutiveInOut();

  // obtain block sizePerThread along contig dim
  auto contigPerThread = getContigPerThread(srcTy);
  auto blockContigSize = contigPerThread[blockedEnc.getOrder()[0]];

  if (blockContigSize <= copyContigSize)
    return rewriter.notifyMatchFailure(
        copyOp,
        "blocked sizePerThread along contiguous dim must be greater than the "
        "max contiguous copy size ");

  contigPerThread[blockedEnc.getOrder()[0]] = copyContigSize;

  // obtain new blockedEnc based on clipped sizePerThread
  auto newBlockEnc = BlockedEncodingAttr::get(
      copyOp.getContext(), srcTy.getShape(), contigPerThread,
      blockedEnc.getOrder(), numWarps, threadsPerWarp,
      blockedEnc.getCGALayout());

  retargetCopyOperandsToEncoding(copyOp, newBlockEnc, axisInfoAnalysis,
                                 rewriter);
  return success();
}
```
* **代码功能说明**: 当源数据在连续维度上的分块大小（`blockContigSize`）大于 `AsyncCopy` 操作实际支持的最大连续拷贝大小（`copyContigSize`）时，该函数会强制将前者的线程切块裁剪（Clip）至后者的大小，并生成新的布局替换原来的布局。
* **原理解析**: `async_cp` 等待执行的事务必须是对齐的向量化操作（最大 128 bit/16 Bytes）。如果由于源 `BlockedEncoding` 的 `sizePerThread` 设置不合理（过大），会导致本应产生一条 16 bytes 内存事务的逻辑被编译为两条 8 bytes 的离散拷贝指令。通过调整 `sizePerThread` 为刚好吻合的尺寸，编译器就能生成完美合并（Coalesced）的单条内存拷贝指令。
* **在整个 PASS 中起到的作用**: 专门针对将数据读入共享内存的场景做了针对性的内存拓扑补丁，防止由于上游 Layout 定义的向量过大而意外导致内存带宽利用率下降。
