# Pass 63：ConvertTritonGPUToLLVM

> kernel：BatchNorm + LIF 脉冲神经元 ｜ CLI：`convert-triton-gpu-to-llvm` ｜ 编译流水线第 63 个 Pass

## 这个 Pass 的作用

ConvertTritonGPUToLLVM 是编译流水线中最重要的降级 Pass，它将 Triton GPU IR（TTGIR，包含 `tt.*`、`ttg.*` 等高层方言）完整转换为 LLVM IR 方言（`llvm.*`、`nvvm.*`），同时实现以下核心工作：
1. 将 `tt.func` 转换为 `llvm.func`，添加 `nvvm.kernel` 和 `nvvm.reqntid` 属性；
2. 将张量（tensor）类型展开为 struct（每个线程持有的标量元素组）；
3. 将 `tt.load`/`tt.store` 转换为带地址计算的 LLVM 内存操作；
4. 将线程索引（由 `tt.get_program_id`、`tt.make_range` 表示的抽象程序 ID）转换为 PTX 的 `ctaid.*`、`tid.x` 寄存器读取；
5. 将 `ttg.convert_layout`（需要 SMEM 的那条）转换为 SMEM load/store 序列。

## IR 变化

行数从 233 行爆炸式增长至 3125 行（增加约 13×），这是最大幅度的变换。

**函数签名变化：**

```mlir
// 变换前（TTGIR）
tt.func public @triton_poi_fused_...(%in_out_ptr0: !tt.ptr<f32>, %in_ptr0: !tt.ptr<f32>, %out_ptr0: !tt.ptr<f32>, %ynumel: i32, %xnumel: i32)

// 变换后（LLVM IR）
llvm.func @triton_poi_fused_...(%in_out_ptr0: !llvm.ptr<1>, %in_ptr0: !llvm.ptr<1>, %out_ptr0: !llvm.ptr<1>, %ynumel: i32, %xnumel: i32, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>)
    attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>}
```

新增了 `%arg5`（smem 基地址）和 `%arg6`（全局基地址），以及 `nvvm.kernel` 标记和 `nvvm.reqntid = 128`（每线程块 128 线程）。

**全局 SMEM 声明：**

```mlir
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
```

**张量常量展开为 struct（以浮点常量 0.5 为例）：**

```mlir
// 变换前（TTGIR，1 行）
%cst = arith.constant dense<5.000000e-01> : tensor<16x64xf32, #blocked>

// 变换后（LLVM IR，每线程 8 个元素的 struct，10 行）
%0 = llvm.mlir.constant(5.000000e-01 : f32) : f32
%1 = llvm.bitcast %0 : f32 to f32
%2 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
%3 = llvm.insertvalue %1, %2[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
%4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
...（共 8 个 insertvalue）...
%10 = llvm.insertvalue %1, %9[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
```

每线程 8 个 f32，对应布局 `{sizePerThread=[1,4], threadsPerWarp=[2,16], warpsPerCTA=[4,1]}`：每线程处理 1×4=4 个元素，但因为有两个 warp 方向的展开，实际展开为 8 个元素（y 方向 2 个 × x 方向 4 个）。

**线程索引计算（`ctaid.y`、`tid.x` 展开）：**

```mlir
%yoffset = nvvm.read.ptx.sreg.ctaid.y : i32        // tt.get_program_id y
%yoffset_3 = llvm.mul %yoffset, %34 : i32           // × 16
%yindex = llvm.mlir.constant(0 : index) : i32
%yindex_4 = nvvm.read.ptx.sreg.tid.x : i32          // thread ID
%yindex_5 = llvm.mlir.constant(127 : i32) : i32
%yindex_6 = llvm.and %yindex_4, %yindex_5 : i32     // tid & 127
%yindex_8 = llvm.urem %yindex_6, %yindex_7 : i32    // tid % 32 = lane ID
%yindex_9 = ttg.warp_id {omitUniformHint}            // warp ID
```

**内存偏移量展开（BN 时间步偏移 3211264 等）：**

```mlir
%64 = llvm.mlir.constant(3211264 : i32) : i32
%65 = llvm.bitcast %64 : i32 to i32
%66 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
%67 = llvm.insertvalue %65, %66[0] : !llvm.struct<(i32, i32, i32, i32)>
...（共 4 个 insertvalue，对应每线程 4 个 x 方向元素的偏移）...
```

## 说明

这次降级是从"what"到"how"的完全展开：

1. **每线程 8 个元素**：布局 `{sizePerThread=[1,4], threadsPerWarp=[2,16], warpsPerCTA=[4,1]}` 在展开后，y 方向每 warp 的两个行组（来自 warpsPerCTA=[4,1] 中 y=1 的贡献）和 x 方向 4 个元素，合计每线程负责 2×4=8 个 f32，因此常量 struct 为 8 元素。

2. **BN+LIF 计算展开**：kernel 中的每个 `arith.mulf`、`arith.cmpf`、`arith.addf` 等操作在 LLVM IR 中都会对 struct 的每个字段分别执行，形成 8 路并行标量操作链，这在最终 PTX 中会被 PTXAS 向量化为 128-bit load（`ld.global.v4.f32`）等宽操作。

3. **SMEM 使用**：唯一的 `ttg.convert_layout`（从 `#blocked` 转为 `#blocked1`，偏移量为 0）被展开为完整的 SMEM write（每线程写 8 个 f32 到 `global_smem[...]`）然后 SMEM read（以新布局读取），中间插入 `__syncthreads()`（`nvvm.barrier0`）。

4. **函数扩展**：增加的两个参数 `%arg5`、`%arg6` 是 Triton runtime 传入的辅助指针，用于访问 global smem 基地址等 runtime 信息，在 PTX 中对应 `__launch_bounds__` 相关的 kernel 参数。
