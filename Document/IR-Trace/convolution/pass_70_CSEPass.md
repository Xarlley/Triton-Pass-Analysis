# Pass 70：CSEPass

> kernel：卷积 (Convolution) ｜ CLI：`cse` ｜ 编译流水线第 70 个 Pass

## 这个 Pass 的作用

`CSEPass`（公共子表达式消除，第四次执行）在 Pass 68（ConvertNVGPUToLLVM）之后清理新引入的重复读取。Pass 68 将 `ttg.warp_id` 展开为 `nvvm.read.ptx.sreg.tid.x + llvm.udiv`，从而在 warp_id 计算处新增了一条 `tid.x` 读取；但这个值与 Pass 65 CSE 之后保留的 `%nhw_1`（同一个 `nvvm.read.ptx.sreg.tid.x`）完全相同。本 Pass 将新增的 `tid.x` 读取替换为对已有 `%nhw_1` 的直接引用，并删除冗余的读取操作。IR 行数从 5753 降至 5752（减少 1 行）。

## IR 变化

**将 Pass 68 新增的 `nvvm.read.ptx.sreg.tid.x` 合并到已有定义**：

```mlir
// 变换前（Pass 68 引入的 warp_id 计算中新增了一条 tid.x 读取）
%nhw_1 = nvvm.read.ptx.sreg.tid.x : i32   // Pass 65 CSE 保留的唯一 tid.x 读取（已在上方）
// ...
%nhw_4 = nvvm.read.ptx.sreg.tid.x : i32   // Pass 68 新增的（用于 warp_id 计算）
%nhw_5 = llvm.udiv %nhw_4, %88 : i32      // warpId = nhw_4 / 32

// 变换后（删除新增的 tid.x 读取，直接复用 %nhw_1）
// %nhw_4 已删除
%nhw_4 = llvm.udiv %nhw_1, %88 : i32      // warpId = nhw_1 / 32（直接引用 %nhw_1）
```

## 说明

这是 Triton 编译流水线中频繁出现的"Pass 引入冗余，下一个 Pass 清理冗余"模式的典型体现。Pass 68 在将 `ttg.warp_id` 转换为等价的 `tid.x / 32` 时，不知道函数作用域中已经有一个 `tid.x` 读取（由 Pass 65 CSE 合并后保留的 `%nhw_1`），因此生成了新的读取。Pass 70 在这两条 `nvvm.read.ptx.sreg.tid.x` 之间再次做 CSE，发现它们语义等价（都是读取 threadIdx.x，且在 kernel 执行期间该值不变），将新增的 `%nhw_4 = tid.x` 替换为对已有 `%nhw_1` 的引用，净减少 1 行。

经过此 Pass，所有 TritonGPU 和 NVGPU 方言的操作均已完全转换为 LLVM + NVVM 方言，且 IR 经过 CSE 清理处于最简形式。最终 5752 行 LLVM IR 将由 LLVM 后端（`mlir-translate`）转换为 PTX，再由 PTXAS 编译为 GPU 可执行的 CUBIN，完成整个从 Python 到 GPU 机器码的编译流程。

本 kernel（VGG16 第一卷积层，3→64 通道，3×3，224×224）编译后的 CUBIN 将对每个 128-行输出分块执行：2 次异步预热 load（prologue）+ 9 次流水线化的 `cp.async + tt.dot` 迭代（steady-state）+ 1 次等待和输出写回（epilogue），利用 36KB shared memory 的三重缓冲实现全程内存延迟隐藏。
