# Pass 68：ConvertNVGPUToLLVM

> kernel：矩阵乘法 / 全连接 (addmm) ｜ CLI：`convert-nvgpu-to-llvm` ｜ 编译流水线第 68 个 Pass

## 这个 Pass 的作用

ConvertNVGPUToLLVM 将残留的 NVIDIA GPU 专有伪操作（`ttg.warp_id` 等）降低为标准 LLVM 方言 + NVVM intrinsic 的等价序列，使 IR 成为完全合法的 LLVM IR（不含任何 Triton/TTG 方言操作）。

对于本 kernel，唯一需要替换的操作是 `ttg.warp_id {omitUniformHint}`，该伪操作表示"读取当前线程的 warp ID"。在 CUDA PTX 中，没有直接获取 warp ID 的单一寄存器指令，warp ID 必须通过线程 ID（`tid.x`）除以每 warp 线程数（32）来计算：

```
warp_id = tid.x / 32
```

因此 `ttg.warp_id` → `nvvm.read.ptx.sreg.tid.x` + `llvm.udiv %tid, 32`（2 条指令替换 1 条，净增 1 行）。

IR 行数从 2073 行增加至 2074 行（净增 1 行：1 条伪操作被 2 条 LLVM 指令替换）。

## IR 变化

**关键变化：`ttg.warp_id` → `nvvm.read.ptx.sreg.tid.x` + `llvm.udiv`**

```mlir
// 变换前（pass 65 after / pass 68 before）：
%rm_4 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc54)    ← 已有 tid.x（用于 lane_id）
%rm_5 = llvm.and %rm_4, %37 : i32 loc(#loc54)         ← lane_id = tid.x & 63
%rm_6 = llvm.urem %rm_5, %48 : i32 loc(#loc54)        ← lane_in_warp = lane_id % 32
%rm_7 = ttg.warp_id {omitUniformHint} loc(#loc54)     ← warp ID 伪操作，待替换
%rm_8 = llvm.shl %rm_6, %50 : i32 loc(#loc54)
%rm_9 = llvm.or %50, %rm_8 : i32 loc(#loc54)
%rm_10 = llvm.shl %rm_7, %36 : i32 loc(#loc54)        ← 使用 warp_id
%rm_11 = llvm.or %rm_9, %rm_10 : i32 loc(#loc54)
...（后续计算依赖 %rm_7 = warp_id，变量编号依次后移）

// 变换后（pass 68 after）：
%rm_4 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc54)    ← 同上
%rm_5 = llvm.and %rm_4, %37 : i32 loc(#loc54)
%rm_6 = llvm.urem %rm_5, %48 : i32 loc(#loc54)
%rm_7 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc54)    ← 新增：再次读取 tid.x
%rm_8 = llvm.udiv %rm_7, %48 : i32 loc(#loc54)        ← 新增：warp_id = tid.x / 32（%48=32）
%rm_9 = llvm.shl %rm_6, %50 : i32 loc(#loc54)         ← 原 %rm_8
%rm_10 = llvm.or %50, %rm_9 : i32 loc(#loc54)         ← 原 %rm_9
%rm_11 = llvm.shl %rm_8, %36 : i32 loc(#loc54)        ← 原 %rm_10，使用 %rm_8（新 warp_id）
%rm_12 = llvm.or %rm_10, %rm_11 : i32 loc(#loc54)     ← 原 %rm_11
...（所有依赖后移一个编号）
```

注意下游所有引用 `%rm_15`（xor 中间值）的操作也随之更新，使用 `%rm_16` 代替，循环头块参数由 `%acc_171/%acc_172` 变为 `%acc_172/%acc_173`，均为编号向后移位 1 的连锁反应。

## 说明

**`ttg.warp_id` 的语义**：`ttg.warp_id {omitUniformHint}` 是 Triton 的高层伪操作，语义为"返回当前执行此指令的 warp 的 ID（从 0 开始，每 CTA 内唯一）"。`omitUniformHint` 表示不保证所有 warp 返回相同值（即该值在 warp 间不均匀），阻止编译器将其错误标记为 uniform（否则 LLVM 可能优化为只在一个 warp 上计算并广播）。

**降低策略**：PTX ISA 没有 `%warpid` 专用寄存器（仅 `%tid.x`，`%laneid` 等），因此 warp ID 通过 `%tid.x ÷ 32` 得到：
- `nvvm.read.ptx.sreg.tid.x`：读取 PTX `%tid.x` 寄存器（当前线程在线程块内的线性索引）。
- `llvm.udiv %tid_x, 32`：整数除法（无符号），得到 warp ID（0 或 1，对于本 kernel num_warps=2）。

**为什么出现第二次 `nvvm.read.ptx.sreg.tid.x`**：因为 `%rm_4`（第一次 tid.x 读取）已被用于计算 lane ID（`%rm_4 & 63`，然后 `% 32`），而 warp ID 计算需要不经过 AND-mask 的原始 tid.x，故重新读取一次（在 PTX 中两次读取同一寄存器不会产生额外硬件开销，只是多一行汇编）。在后续 Pass 70（CSEPass）中，这两次重复的 `nvvm.read.ptx.sreg.tid.x` 将被 CSE 合并消除。
