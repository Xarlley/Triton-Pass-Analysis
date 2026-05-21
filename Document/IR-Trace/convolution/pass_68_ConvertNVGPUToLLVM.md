# Pass 68：ConvertNVGPUToLLVM

> kernel：卷积 (Convolution) ｜ CLI：`convert-nv-gpu-to-llvm` ｜ 编译流水线第 68 个 Pass

## 这个 Pass 的作用

`ConvertNVGPUToLLVM`（NVGPU 方言到 LLVM 方言转换）将残留的 Triton/NVGPU 混合方言操作降低为纯 LLVM 方言。主要工作是将 `ttg.warp_id` 操作替换为等价的 `nvvm.read.ptx.sreg.tid.x + llvm.udiv`（即通过 threadIdx.x 除以 warp 大小 32 来计算 warp ID）。IR 行数从 5752 增至 5753（增加 1 行，因为 `ttg.warp_id` 的每次替换需要额外一条 `llvm.udiv` 指令，但经过 CSE 后 Pass 65 已将多次重复的 warp_id 读取合并为一处，所以只有 1 行净增加）。

## IR 变化

**将 `ttg.warp_id` 替换为 `tid.x / 32`**：

```mlir
// 变换前（Triton 方言残留的 warp_id 操作）
%nhw_4 = ttg.warp_id {omitUniformHint} : i32

// 变换后（等价的 LLVM 序列：先读 tid.x，再整除 warp size = 32 = 0x20）
%nhw_4 = nvvm.read.ptx.sreg.tid.x : i32    // threadIdx.x
%nhw_5 = llvm.udiv %nhw_4, %88 : i32       // warpId = threadIdx.x / 32（%88 = 常量 32）
```

由于 Pass 65（CSE）已将所有重复的 `ttg.warp_id` 调用合并为单次调用，此处只有一处替换，净增加 1 行（`nvvm.read.ptx.sreg.tid.x` 替换 `ttg.warp_id`，但需要额外的 `llvm.udiv`）。

## 说明

`ttg.warp_id` 是 Triton 方言中的高层抽象，用于表示当前线程所在的 warp 编号（0~3 对于 4-warp kernel）。在 PTX/CUDA 编程模型中，没有直接读取 warp ID 的指令；标准做法是读取 `%tid.x`（threadIdx.x，0~127），然后整除 warp 大小（32）得到 warp ID（0~3）。

`{omitUniformHint}` 属性表示编译器已知此操作对于同一 warp 内所有线程返回相同的值（warp 内的 uniform value），可以用单次读取代替 lane 广播，但此提示在 LLVM IR 层面通过常规的 `udiv` 表达。

替换后的计算链 `tid.x → udiv 32 → warpId` 在 PTX 后端会被优化为 `mov.u32 %r, %tid.x; shr.u32 %r, %r, 5`（右移 5 位等价于整除 32），最终 PTX 汇编中表现为一条 `shr` 指令，开销极小。

`{addr_space = 3 : i32}` 的 `@global_smem` 全局符号在此 Pass 中也会被 ConvertNVGPUToLLVM 确认（shared memory 的地址空间在 NVVM 中固定为 3），但由于本 kernel 只有 `ttg.warp_id` 需要此 Pass 处理，所以行数净增仅 1 行。
