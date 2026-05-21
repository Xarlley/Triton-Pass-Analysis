# Pass 68：ConvertNVGPUToLLVM

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`convert-nv-gpu-to-llvm` ｜ 编译流水线第 68 个 Pass

## 这个 Pass 的作用

ConvertNVGPUToLLVM 将残留的高级 NVIDIA GPU 方言操作（`nvgpu.*`、`ttg.warp_id` 等）降级为标准 LLVM IR 或 PTX intrinsic。最典型的转换是将 `ttg.warp_id` 操作（Triton 自定义的 warp ID 获取）替换为 `nvvm.read.ptx.sreg.tid.x` 加整数除法（`tid.x / 32`）的组合，完成向纯 LLVM/NVVM 操作的最后映射。

## IR 变化

IR 行数从 191 增至 192（仅 1 行变化），是本流水线中最小的一次有效变化。

核心变化是 `ttg.warp_id` 被展开为 `tid.x / 32`：

**变化前（before）**：
```mlir
%xindex_4 = ttg.warp_id {omitUniformHint} loc(#loc35)
%xindex_5 = llvm.shl %xindex_3, %2 : i32 loc(#loc35)
%xindex_6 = llvm.or %2, %xindex_5 : i32 loc(#loc35)
%xindex_7 = llvm.shl %xindex_4, %xindex : i32 loc(#loc35)
%xindex_8 = llvm.or %xindex_6, %xindex_7 : i32 loc(#loc35)
```

**变化后（after）**：
```mlir
%xindex_4 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc35)
%xindex_5 = llvm.udiv %xindex_4, %3 : i32 loc(#loc35)    // warp_id = tid.x / 32
%xindex_6 = llvm.shl %xindex_3, %2 : i32 loc(#loc35)
%xindex_7 = llvm.or %2, %xindex_6 : i32 loc(#loc35)
%xindex_8 = llvm.shl %xindex_5, %xindex : i32 loc(#loc35)
%xindex_9 = llvm.or %xindex_7, %xindex_8 : i32 loc(#loc35)
```

`ttg.warp_id {omitUniformHint}` → 新增 `nvvm.read.ptx.sreg.tid.x` + `llvm.udiv ... %3`（`%3 = 32`），行数从 191 增至 192，编号整体后移 1。

## 说明

`ttg.warp_id` 是 Triton GPU 方言中的抽象操作，语义为"当前线程所属 warp 的编号"。在 CUDA 中，warp ID = `threadIdx.x / 32`（对于 1D block）。`{omitUniformHint}` 属性表明此操作不具有 warp uniform 性质（每个 warp 的值不同，因此 LLVM 不应对其做 uniform 假设进行特殊优化）。

展开后：
- `nvvm.read.ptx.sreg.tid.x` → PTX `%tid.x` 寄存器（线程在 block 内的 x 维度 ID）；
- `llvm.udiv tid_x, 32` → 计算 warp ID，这是 PTX 层面计算 warp ID 的标准方式（CUDA 硬件没有直接的 warp ID 寄存器，sm_80+ 可以通过 `%warpid` 读取，但 Triton 选择使用 `tid.x / 32` 以保证更广泛的兼容性）。

经过此 Pass，所有 Triton 专用操作已完全消除，IR 中仅剩标准 `llvm.*` 和 `nvvm.*` 操作，可以直接进入 LLVM NVPTX 后端生成 PTX。
