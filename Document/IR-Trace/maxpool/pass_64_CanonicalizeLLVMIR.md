# Pass 64：CanonicalizeLLVMIR

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`canonicalize-llvm-ir` ｜ 编译流水线第 64 个 Pass

## 这个 Pass 的作用

CanonicalizeLLVMIR 是 Triton 专有的 LLVM IR 规范化 Pass，在 ConvertTritonGPUToLLVM 产生的冗余 LLVM IR 基础上进行大规模化简：消除冗余的 `llvm.bitcast`、`llvm.insertvalue`/`llvm.extractvalue` 结构对、死代码、`llvm.mlir.undef` 占位等，将代码从机械展开的"结构体包装"形式压缩为直接操作标量的精简形式。

## IR 变化

IR 规模从 566 行**急剧压缩至 191 行**（缩减 2/3），是整个流水线中 IR 行数下降最多的一次。

**核心变化 1：消除 struct 包装**

Pass 63 将每个张量元素包装为 `!llvm.struct<(i32, i32)>` 然后立即 extract，CanonicalizeLLVMIR 识别这些 insert/extract 对为恒等操作并消除：

```mlir
// before（566 行中的典型模式）：
%3 = llvm.mlir.undef : !llvm.struct<(i32, i32)>
%4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i32, i32)>
%5 = llvm.insertvalue %2, %4[1] : !llvm.struct<(i32, i32)>
// ... 后续立即 extractvalue [0] 和 [1]

// after（191 行，直接使用标量）：
%12 = llvm.mlir.constant(14400 : i32) : i32
// 常量直接使用，无 struct 包装
```

**核心变化 2：redundant bitcast 消除**

```mlir
// before：
%2 = llvm.bitcast %1 : i32 to i32   // bitcast i32 -> i32 是恒等操作

// after：直接使用 %1，bitcast 消失
```

**核心变化 3：inline asm 保留，但上下文精简**

load 的 PTX inline asm 保留（不可化简），但其输入指针的计算链大幅精简：

```mlir
// after：直接的 getelementptr + inline asm
%tmp0_20 = llvm.getelementptr %in_ptr0[%tmp0_19] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
%tmp0_21 = llvm.inline_asm ... "ld.global.v2.b32 { $0, $1 }, [ $2 + 0 ];" ... %tmp0_20
%tmp0_22 = llvm.extractvalue %tmp0_21[0] : !llvm.struct<(i32, i32)>
%tmp0_23 = llvm.bitcast %tmp0_22 : i32 to vector<1xf32>
%tmp0_26 = llvm.extractelement %tmp0_23[%5 : i32] : vector<1xf32>
```

**核心变化 4：warp_id 展开**

```mlir
// before：
%xindex_4 = ttg.warp_id {omitUniformHint}

// after：
%xindex_4 = nvvm.read.ptx.sreg.tid.x : i32
%xindex_5 = llvm.udiv %xindex_4, %3 : i32   // tid.x / 32 = warp_id
```

`ttg.warp_id` 被展开为 `tid.x / 32` 的标量算术，所有 CUDA 专用 PTX 寄存器映射完成。

## 说明

Pass 63（ConvertTritonGPUToLLVM）产出的 LLVM IR 是机械降级的结果，其中包含大量"为了类型系统正确性而引入"的冗余操作（struct pack/unpack、bitcast i32→i32 等）。CanonicalizeLLVMIR 是 Triton 的 LLVM 层优化步骤，与 MLIR 的 `canonicalize` 不同，它专门针对 Triton 生成 LLVM IR 的模式进行优化。

压缩后的 191 行 LLVM IR 才是真正接近最终 PTX 的形态：每条 load 对应一条 `ld.global.v2.b32`，每对 float 的比较对应 `llvm.fcmp + llvm.select`，地址计算为直接的整数算术 + `getelementptr`。这与真实 PTX 的指令结构基本一一对应，体现了本 MaxPool+BN+LIF kernel 的高度精简性。
