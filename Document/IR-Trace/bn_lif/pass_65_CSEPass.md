# Pass 65：CSEPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`cse` ｜ 编译流水线第 65 个 Pass

## 这个 Pass 的作用

这是流水线中对 LLVM IR 执行的第一次 CSE（公共子表达式消除）。在 CanonicalizeLLVMIR（Pass 64）将 3125 行压缩至 870 行后，仍存在一些跨不同 BN/LIF 计算分支共享的常量读取和线程索引计算，CSE 识别并消除这些重复，进一步将行数从 870 行减少至 826 行（减少 44 行）。

## IR 变化

本次 CSE 消除的主要是**线程索引计算中的重复子表达式**，尤其是 y 方向（行索引）的复杂计算链。在 before.mlir 中，有多套几乎相同的 warp_id + lane_id 组合计算：

**变换前（含重复的 warp_id 计算）：**

```mlir
%yindex_17 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc70)
%yindex_18 = llvm.and %yindex_17, %25 : i32 loc(#loc70)
%yindex_19 = llvm.urem %yindex_18, %24 : i32 loc(#loc70)
%yindex_20 = ttg.warp_id {omitUniformHint} loc(#loc70)
%yindex_21 = llvm.shl %yindex_19, %23 : i32 loc(#loc70)
%yindex_22 = llvm.or %23, %yindex_21 : i32 loc(#loc70)
%yindex_23 = llvm.shl %yindex_20, %22 : i32 loc(#loc70)
%yindex_24 = llvm.or %yindex_22, %yindex_23 : i32 loc(#loc70)
%yindex_25 = llvm.and %yindex_24, %18 : i32 loc(#loc70)
%yindex_26 = llvm.shl %yindex_25, %17 : i32 loc(#loc70)
%yindex_27 = llvm.or disjoint %yindex_26, %23 : i32 loc(#loc70)
%yindex_28 = llvm.xor %23, %yindex_27 : i32 loc(#loc70)
%yindex_29 = llvm.xor %yindex_28, %23 : i32 loc(#loc70)
%yindex_30 = llvm.add %yindex_29, %26 : i32 loc(#loc70)
%yindex_31 = llvm.add %yoffset_1, %yindex_15 : i32 loc(#loc71)
%yindex_32 = llvm.add %yoffset_1, %yindex_16 : i32 loc(#loc71)
%yindex_33 = llvm.add %yoffset_1, %yindex_30 : i32 loc(#loc71)
```

**变换后（重复路径被消除，公共结果复用）：**

```mlir
%yindex_17 = llvm.and %yindex_8, %18 : i32 loc(#loc70)
%yindex_18 = llvm.shl %yindex_17, %17 : i32 loc(#loc70)
%yindex_19 = llvm.or disjoint %yindex_18, %23 : i32 loc(#loc70)
%yindex_20 = llvm.xor %23, %yindex_19 : i32 loc(#loc70)
%yindex_21 = llvm.xor %yindex_20, %23 : i32 loc(#loc70)
%yindex_22 = llvm.add %yindex_21, %26 : i32 loc(#loc70)
%yindex_23 = llvm.add %yoffset_1, %yindex_15 : i32 loc(#loc71)
%yindex_24 = llvm.add %yoffset_1, %yindex_16 : i32 loc(#loc71)
%yindex_25 = llvm.add %yoffset_1, %yindex_22 : i32 loc(#loc71)
```

多次重复计算的 `nvvm.read.ptx.sreg.tid.x` 读取、`ttg.warp_id` 调用和相同的 `llvm.and`/`llvm.urem` 链被合并为首次出现的结果，后续直接引用 `%yindex_8`（已在 Pass 64 after 中计算过的 y 方向 warp 编号组合值）。

## 说明

CSE 在这里消除的 44 行主要来自两类重复：

1. **线程索引重复**：BN+LIF kernel 的 y 方向索引（行号）在写入 `in_out_ptr0`（膜电位缓冲）和 `out_ptr0`（脉冲输出）时分别计算了一次，由于两者的 y 索引计算完全相同，CSE 将第二次计算删除并引用第一次结果。

2. **常量重复**：Pass 64 规范化后仍存在少量重复的常量定义（如多处出现的 `0 : i32`），CSE 合并为单一定义。

经过 Pass 65，LLVM IR 达到较稳定的 826 行形态。后续 Pass 66（ConvertWarpSpecializeToLLVM）和 67（ReconcileUnrealizedCasts）对本 kernel 为 no-op（无 warp 专业化、无需 reconcile），Pass 68（ConvertNVGPUToLLVM）则处理最后一个 NVGPU 方言降级，最终生成纯净的 LLVM IR，由 PTXAS 编译为 PTX。
