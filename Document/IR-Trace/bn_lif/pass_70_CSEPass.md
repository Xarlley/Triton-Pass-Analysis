# Pass 70：CSEPass

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`cse` ｜ 编译流水线第 70 个 Pass

## 这个 Pass 的作用

这是流水线中最后一次 CSE（公共子表达式消除），专门清理 Pass 68（ConvertNVGPUToLLVM）引入的一个新的重复计算：`ttg.warp_id` 展开后产生的额外 `nvvm.read.ptx.sreg.tid.x` 读取，与已有的 `tid.x` 读取重复，CSE 消除这个重复，将行数从 827 行恢复至 826 行。

## IR 变化

本次 CSE 消除的是 y 方向索引计算中新产生的重复 `nvvm.read.ptx.sreg.tid.x` 读取：

**变换前（Pass 68 后，827 行，含重复的 tid.x 读取）：**

```mlir
%yindex = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc70)   // 第一次读取（Pass 64 保留）
%yindex_2 = llvm.and %yindex, %25 : i32 loc(#loc70)
%yindex_3 = llvm.urem %yindex_2, %24 : i32 loc(#loc70)
%yindex_4 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc70)   // Pass 68 新增的重复读取
%yindex_5 = llvm.udiv %yindex_4, %24 : i32 loc(#loc70)
%yindex_6 = llvm.shl %yindex_3, %23 : i32 loc(#loc70)
%yindex_7 = llvm.or %23, %yindex_6 : i32 loc(#loc70)
%yindex_8 = llvm.shl %yindex_5, %22 : i32 loc(#loc70)
%yindex_9 = llvm.or %yindex_7, %yindex_8 : i32 loc(#loc70)
```

**变换后（826 行，第二次 tid.x 读取替换为第一次的引用）：**

```mlir
%yindex = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc70)   // 唯一的 tid.x 读取
%yindex_2 = llvm.and %yindex, %25 : i32 loc(#loc70)
%yindex_3 = llvm.urem %yindex_2, %24 : i32 loc(#loc70)
%yindex_4 = llvm.udiv %yindex, %24 : i32 loc(#loc70)   // 引用 %yindex，而非重复读取
%yindex_5 = llvm.shl %yindex_3, %23 : i32 loc(#loc70)
%yindex_6 = llvm.or %23, %yindex_5 : i32 loc(#loc70)
%yindex_7 = llvm.shl %yindex_4, %22 : i32 loc(#loc70)
%yindex_8 = llvm.or %yindex_6, %yindex_7 : i32 loc(#loc70)
```

`%yindex_4 = nvvm.read.ptx.sreg.tid.x` 被消除，对应的 `udiv` 操作改为引用已有的 `%yindex`（即 `llvm.udiv %yindex, %24`），后续编号向前移动一位（`%yindex_5` 变为 `%yindex_4`，等等）。

## 说明

这是整个编译流水线中最后一次改变 BN+LIF kernel IR 的 Pass。826 行的纯 LLVM IR 是最终提交给 LLVM NVPTX 后端的形态。

`nvvm.read.ptx.sreg.tid.x` 对应 PTX 指令 `mov.u32 %r, %tid.x`，这是一个特殊寄存器读取操作，在每次调用时都返回相同的值（线程 ID 在整个 kernel 执行期间不变），因此 CSE 将其视为纯函数（pure function）并消除重复读取是安全的。

最终 826 行 LLVM IR 的结构：
- 前 ~40 行：模块属性和 SMEM 全局声明
- 约 10 行：标量常量定义（浮点和整数）
- 约 50 行：线程/warp 索引计算（y 和 x 方向）
- 约 400 行：4 次 load + BN+LIF 计算（4 个时间步，每步约 100 行）
- 约 100 行：`ttg.convert_layout` 展开的 SMEM write/sync/read 序列
- 约 50 行：2 次 store（写回膜电位和脉冲输出）

后续 Pass 71（SymbolDCEPass）和 Pass 72（ConvertNVVMToLLVMPass）对本 kernel 均为 no-op，826 行 LLVM IR 直接送入 PTXAS 生成 PTX 汇编（`real.ptx`，约 826 行 PTX 指令）。
