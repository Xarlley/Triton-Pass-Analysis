# Pass 70：CSEPass

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`cse` ｜ 编译流水线第 70 个 Pass

## 这个 Pass 的作用

CSEPass（Common Subexpression Elimination）在 LLVM IR 层面消除公共子表达式。经过 ConvertNVGPUToLLVM（Pass 68）后，IR 中可能存在对同一 PTX 寄存器的重复读取（如 `nvvm.read.ptx.sreg.tid.x` 被读取两次），CSE 将其合并为一次读取，减少冗余的寄存器读取指令。

## IR 变化

IR 行数从 192 **降至 191 行**（减少 1 行），消除了一个重复的 `nvvm.read.ptx.sreg.tid.x` 读取操作。

**变化前（before，pass 68 后，`tid.x` 读取两次）**：
```mlir
%xindex_1 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc35)
%xindex_2 = llvm.and %xindex_1, %4 : i32 loc(#loc35)
%xindex_3 = llvm.urem %xindex_2, %3 : i32 loc(#loc35)
%xindex_4 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc35)   // 重复！
%xindex_5 = llvm.udiv %xindex_4, %3 : i32 loc(#loc35)
```

**变化后（after，`tid.x` 只读取一次）**：
```mlir
%xindex_1 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc35)
%xindex_2 = llvm.and %xindex_1, %4 : i32 loc(#loc35)
%xindex_3 = llvm.urem %xindex_2, %3 : i32 loc(#loc35)
%xindex_4 = llvm.udiv %xindex_1, %3 : i32 loc(#loc35)    // 复用 %xindex_1
```

`%xindex_4` 的操作数从原来的重复 `nvvm.read.ptx.sreg.tid.x`（即 before 中的 `%xindex_4` 的前一行）改为直接引用首次读取的 `%xindex_1`，节省一次 PTX 寄存器读取指令。

## 说明

这次 CSE 消除的是 Pass 68（ConvertNVGPUToLLVM）将 `ttg.warp_id` 展开为 `tid.x / 32` 时产生的重复读取：

- Pass 64（CanonicalizeLLVMIR）已将一次 `tid.x` 读取用于 lane ID 计算（`tid.x % 32`）；
- Pass 68 展开 warp_id 时新增了第二次 `tid.x` 读取（`tid.x / 32`）；
- Pass 70（CSE）识别两次 `nvvm.read.ptx.sreg.tid.x` 是相同的无副作用读操作，将第二次引用替换为第一次的结果 `%xindex_1`。

`nvvm.read.ptx.sreg.tid.x` 在 LLVM/NVVM 中是一个无副作用的纯读操作（读取 PTX `%tid.x` 寄存器），因此 CSE 可以安全地合并。合并后 PTX 只生成一条 `mov.u32 reg, %tid.x` 指令，节省了一条指令和对应的寄存器分配压力。

这是整个流水线的最后一次有意义的 IR 变化。经过此 Pass，IR 进入最终的 191 行 LLVM IR 形态，准备交由 LLVM NVPTX 后端编译为 PTX 汇编。最终 PTX 中可以看到 `ld.param.u64`、`mov.u32 %r, %tid.x`、`mul.lo.s32`、`ld.global.v2.b32` 等指令，与本 LLVM IR 直接对应。
