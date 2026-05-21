# Pass 68：ConvertNVGPUToLLVM

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`convert-nv-gpu-to-llvm` ｜ 编译流水线第 68 个 Pass

## 这个 Pass 的作用

ConvertNVGPUToLLVM 将 LLVM IR 中残留的少量非标准 Triton 方言操作（如 `ttg.warp_id`）转换为标准的 NVVM 内建函数（如 `nvvm.read.ptx.sreg.tid.x` + 整数除法）。这是 GPU 方言完全降级为纯 NVVM/LLVM 方言的最后一步，之后的 IR 可以直接由 LLVM NVPTX 后端编译为 PTX。

## IR 变化

行数从 826 行增至 827 行（+1 行），唯一的变化是 `ttg.warp_id` 被展开为标准 NVVM 指令序列：

**变换前（使用 Triton 专有 `ttg.warp_id`）：**

```mlir
%yindex_4 = ttg.warp_id {omitUniformHint} loc(#loc70)
%yindex_5 = llvm.shl %yindex_3, %23 : i32 loc(#loc70)
%yindex_6 = llvm.or %23, %yindex_5 : i32 loc(#loc70)
%yindex_7 = llvm.shl %yindex_4, %22 : i32 loc(#loc70)
%yindex_8 = llvm.or %yindex_6, %yindex_7 : i32 loc(#loc70)
```

**变换后（展开为 NVVM 标准指令）：**

```mlir
%yindex_4 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc70)
%yindex_5 = llvm.udiv %yindex_4, %24 : i32 loc(#loc70)    // warp_id = tid.x / 32
%yindex_6 = llvm.shl %yindex_3, %23 : i32 loc(#loc70)
%yindex_7 = llvm.or %23, %yindex_6 : i32 loc(#loc70)
%yindex_8 = llvm.shl %yindex_5, %22 : i32 loc(#loc70)
%yindex_9 = llvm.or %yindex_7, %yindex_8 : i32 loc(#loc70)
```

展开规则：`warp_id = tid.x / threads_per_warp = tid.x / 32`，即 `ttg.warp_id {omitUniformHint}` 被替换为一次 `nvvm.read.ptx.sreg.tid.x` 读取加一次整数除法 `llvm.udiv`。`{omitUniformHint}` 说明 Triton 知道这个值在 warp 内不是 uniform 的（每个线程的 warp_id 相同，但 Triton 选择不做 SGPR 优化）。

行数 +1 的原因是：原来的 `ttg.warp_id` 是 1 行，展开后是 2 行（`nvvm.read.ptx.sreg.tid.x` + `llvm.udiv`）。

## 说明

`ttg.warp_id` 是 Triton GPU 方言中对 warp 编号的高层抽象，在不同硬件上可能对应不同的计算方式。对于 NVIDIA GPU（PTX 模型）：

```
warp_id = threadIdx.x / 32
```

这就是 Pass 68 展开后的计算：`nvvm.read.ptx.sreg.tid.x`（读取 `threadIdx.x`）→ `llvm.udiv %tid, 32`（整除 32 得 warp 编号）。

在本 BN+LIF kernel 中，warp_id 用于计算 y 方向的行索引偏移：每个 warp 负责特定的若干行（由布局 `warpsPerCTA=[4,1]` 决定），warp_id 决定了这个 warp 处理哪些行。具体地：

- warp 0：处理 y 方向第 0 块的行
- warp 1：处理 y 方向第 1 块的行
- warp 2：处理 y 方向第 2 块的行
- warp 3：处理 y 方向第 3 块的行

展开后的 `udiv` 在最终 PTX 中会生成 `div.u32` 指令，或由 PTXAS 优化为 `shr.u32` + 移位（因为 32 是 2 的幂次）。Pass 70 的 CSE 将进一步优化掉这次新增的重复 `tid.x` 读取。
