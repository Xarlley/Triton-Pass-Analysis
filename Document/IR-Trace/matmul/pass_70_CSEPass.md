# Pass 70：CSEPass（第四次）

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`cse` ｜ 编译流水线第 70 个 Pass

## 这个 Pass 的作用

这是 CSEPass（公共子表达式消除）的第四次执行，也是整个编译流水线的最后一次 CSE。本次 CSE 针对 Pass 68（ConvertNVGPUToLLVM）展开的 `ttg.warp_id → nvvm.read.ptx.sreg.tid.x + llvm.udiv` 序列，消除其中新引入的重复 `nvvm.read.ptx.sreg.tid.x` 调用。

Pass 68 为实现 warp ID 计算，引入了第二次 `nvvm.read.ptx.sreg.tid.x`（即 `%rm_7 = nvvm.read.ptx.sreg.tid.x`），而在它之前已经存在 `%rm_4 = nvvm.read.ptx.sreg.tid.x`。CSE 识别两次读取相同 PTX 寄存器的操作等价，将 `%rm_7` 替换为对已有 `%rm_4` 的复用，从而消除 1 行重复指令。

IR 行数从 2074 行降至 2073 行（减少 1 行），这是本流水线最小的单次变化。变量编号随之向前移位 1（所有 `%rm_8` 以后的编号回退为 `%rm_7` 以后）。

## IR 变化

**关键变化：消除重复的 `nvvm.read.ptx.sreg.tid.x`，warp_id 改用已有的 `%rm_4`**

```mlir
// 变换前（pass 68 after）：
%rm_4 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc54)    ← 第一次读取 tid.x（用于 lane_id）
%rm_5 = llvm.and %rm_4, %37 : i32 loc(#loc54)
%rm_6 = llvm.urem %rm_5, %48 : i32 loc(#loc54)
%rm_7 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc54)    ← 第二次读取（Pass 68 新增，被消除）
%rm_8 = llvm.udiv %rm_7, %48 : i32 loc(#loc54)        ← warp_id = %rm_7 / 32
%rm_9 = llvm.shl %rm_6, %50 : i32 loc(#loc54)
%rm_10 = llvm.or %50, %rm_9 : i32 loc(#loc54)
%rm_11 = llvm.shl %rm_8, %36 : i32 loc(#loc54)        ← 使用 warp_id（%rm_8）
...

// 变换后（pass 70 after）：
%rm_4 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc54)    ← 保留，唯一的 tid.x 读取
%rm_5 = llvm.and %rm_4, %37 : i32 loc(#loc54)
%rm_6 = llvm.urem %rm_5, %48 : i32 loc(#loc54)
// %rm_7（第二次 tid.x 读取）已消除
%rm_7 = llvm.udiv %rm_4, %48 : i32 loc(#loc54)        ← 直接复用 %rm_4，编号回退
%rm_8 = llvm.shl %rm_6, %50 : i32 loc(#loc54)         ← 原 %rm_9
%rm_9 = llvm.or %50, %rm_8 : i32 loc(#loc54)          ← 原 %rm_10
%rm_10 = llvm.shl %rm_7, %36 : i32 loc(#loc54)        ← 原 %rm_11，使用 %rm_7（新 warp_id）
...
```

下游所有依赖变量（`%rm_16` → `%rm_15` 用于 xor 运算、`%rn_23` → `%rn_22` 等）均向前移位 1，形成连锁重编号，但语义不变。

## 说明

**为什么 CSE 能消除 `nvvm.read.ptx.sreg.tid.x`**：MLIR 的 CSE Pass 对 `nvvm.read.ptx.sreg.tid.x` 有特殊处理——虽然 PTX 寄存器读取在硬件上是"有副作用"的操作（因为它依赖线程上下文），但 MLIR 将 `nvvm.read.ptx.sreg.tid.x` 建模为无副作用（`ReadNone`）的确定性操作：在同一 kernel 执行期间，同一线程的 `tid.x` 永远不会改变，因此多次读取完全等价，可以安全去重。

**最终 IR 状态**：经过 Pass 70，IR 变为完全合法的 NVVM LLVM IR：
- 不含任何 Triton/TTG/NVGPU 方言操作
- 所有共享内存访问通过 `@global_smem` + `llvm.getelementptr` 寻址
- 所有异步拷贝通过 `llvm.inline_asm` 的 `cp.async.cg.shared.global` PTX 指令完成
- warp ID 通过 `nvvm.read.ptx.sreg.tid.x` + `llvm.udiv 32` 计算
- 矩阵乘通过 `llvm.intr.fmuladd` 标量 FMA 链完成
- 全局存储通过 `llvm.inline_asm` 的 `st.global.v4.b32` PTX 指令完成

这是 Triton 编译流水线的终态 IR，将直接被 LLVM 后端翻译为 PTX 汇编，再由 NVCC 编译为 cubin 二进制，最终由 CUDA 运行时加载执行。
