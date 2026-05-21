# Pass 63：ConvertTritonGPUToLLVM

> kernel：MaxPool + BN + LIF 融合 ｜ CLI：`convert-triton-gpu-to-llvm` ｜ 编译流水线第 63 个 Pass

## 这个 Pass 的作用

ConvertTritonGPUToLLVM 是编译流水线中最关键的降级 Pass，将 Triton GPU IR（`tt.*`、`ttg.*` 方言）全面转换为 LLVM IR（`llvm.*`、`nvvm.*` 操作）。这一转换将所有高级抽象（张量操作、blocked layout、program_id）展开为具体的线程级标量操作：每个 tensor 操作变为每线程处理 sizePerThread 个元素的标量指令序列，内存访问变为 PTX inline assembly。

## IR 变化

IR 规模发生了巨大扩张，从 134 行增至 566 行（约 4 倍），函数从 `tt.func` 变为 `llvm.func`，所有张量操作展开为标量操作链。

**before（Triton GPU IR，张量操作）**：
```mlir
tt.func public @triton_poi_fused_...(%in_ptr0: !tt.ptr<f32> ..., %out_ptr0: !tt.ptr<f32> ...) attributes {noinline = false} {
    %xoffset = tt.get_program_id x : i32
    %xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %tmp0_13 = tt.load %tmp0_12 : tensor<512x!tt.ptr<f32>, #blocked>
```

**after（LLVM IR，展开为标量 + PTX inline asm）**：
```mlir
llvm.func @triton_poi_fused_...(%in_ptr0: !llvm.ptr<1> ..., %out_ptr0: !llvm.ptr<1> ..., %xnumel: i32, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 256>} {
    %xoffset = nvvm.read.ptx.sreg.ctaid.x : i32
    %xindex_3 = nvvm.read.ptx.sreg.tid.x : i32
    %xindex_4 = ttg.warp_id {omitUniformHint}
    ...
    %tmp0_133 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = []
        "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09ld.global.v2.b32 { $0, $1 }, [ $2 + 0 ];"
        "=r,=r,l" %tmp0_131 : (!llvm.ptr<1>) -> !llvm.struct<(i32, i32)>
```

关键转换要点：
1. **张量布局展开**：`sizePerThread=[2]` 布局 → 每个张量变量变为 2 元素的 `!llvm.struct<(T, T)>`；
2. **线程索引计算**：`tt.get_program_id x` → `nvvm.read.ptx.sreg.ctaid.x`；`tt.make_range` → `nvvm.read.ptx.sreg.tid.x` + warp_id 计算；
3. **内存访问**：`tt.load` → PTX inline asm `ld.global.v2.b32`（128-bit 宽加载）；
4. **全局符号**：新增 `llvm.mlir.global external @global_smem() : !llvm.array<0 x i8>`（0 字节 shared memory）；
5. **kernel 属性**：`nvvm.kernel = 1 : ui1`（标记为 CUDA kernel）、`nvvm.reqntid = array<i32: 256>`（强制 256 线程/block）。

## 说明

这一转换的核心在于将 Triton 的"向量化编程模型"（每个程序实例操作 BLOCK_SIZE 个元素）映射到 CUDA 的"单线程编程模型"（每个线程操作 sizePerThread=2 个元素）。

具体到本 MaxPool+BN+LIF kernel，线程 `tid` 在 warp `wid` 中，其负责的两个元素全局索引为：
```
elem_0 = blockIdx.x * 512 + (wid * 32 + tid_in_warp) * 2
elem_1 = elem_0 + 1
```

这两个元素用一条 `ld.global.v2.b32` 指令（128-bit 宽加载）同时从全局内存读取，充分利用了 NVIDIA GPU 的宽内存事务能力（L2 cache line 为 128 字节）。

4 次 load（对应 MaxPool 2×2 的 4 个候选位置）各自产生一个 `!llvm.struct<(i32, i32)>`，经 bitcast 为 `vector<1xf32>` 后提取为标量 f32，然后进行三级 `llvm.fcmp une + llvm.select` 比较（对应原 IR 的 `arith.cmpf une/ogt + arith.ori + arith.select`，即 NaN 安全的 max 操作）。
