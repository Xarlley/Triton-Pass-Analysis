# Pass 63：ConvertTritonGPUToLLVM

> kernel：卷积 (Convolution) ｜ CLI：`convert-triton-gpu-to-llvm` ｜ 编译流水线第 63 个 Pass

## 这个 Pass 的作用

`ConvertTritonGPUToLLVM` 是整个编译流水线的核心下沉步骤，将所有 TritonGPU 方言（`tt.*`、`ttg.*`、`ttng.*`）操作完整转换为 LLVM 方言 + NVVM 方言的低级 IR。IR 行数从 408 行爆增至 22043 行（约 54 倍展开），因为每一个高层抽象操作都被展开为针对每个线程、每个寄存器变量的具体 LLVM 指令序列。此 Pass 是将 GPU 感知的 Triton 中间表示转化为可被 LLVM/PTX 后端直接处理的形式的关键节点。

## IR 变化

**函数签名从 `tt.func` 降低为 `llvm.func`**：

```mlir
// 变换前（Triton 函数）
tt.func public @triton_tem_fused_convolution_view_2(%arg_X: !tt.ptr<f32>, %arg_W: !tt.ptr<f32>, %out_ptr0: !tt.ptr<f32>) {

// 变换后（LLVM 函数，带 NVVM kernel 属性和线程数标注）
llvm.func @triton_tem_fused_convolution_view_2(%arg_X: !llvm.ptr<1>, %arg_W: !llvm.ptr<1>, %out_ptr0: !llvm.ptr<1>, ...) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
```

**`tensor<128x64xf32, #blocked3>` 累加器展开为 64 个标量寄存器**：

```mlir
// 变换后：128×64 的输出张量被展开为每个线程持有 64 个 f32（基于 sizePerThread=[4,4]，每线程负责 4×4=16 个元素 × ... = 64 个）
%2 = llvm.mlir.undef : !llvm.struct<(f32 × 64)>
// 逐个初始化为 0.0：
%3 = llvm.insertvalue %1, %2[0] : !llvm.struct<(f32 × 64)>
// ... (共 64 次 insertvalue，对应 #blocked3 中每线程的 4×4×4 = 64 个输出元素)
```

**`tt.get_program_id` 降低为 `nvvm.read.ptx.sreg.ctaid`**：

```mlir
// 变换后
%nhw = nvvm.read.ptx.sreg.ctaid.x : i32   // blockIdx.x
%idx_y_c = nvvm.read.ptx.sreg.ctaid.y : i32  // blockIdx.y
%nhw_5 = nvvm.read.ptx.sreg.tid.x : i32    // threadIdx.x
```

**`ttg.async_copy_global_to_local` 降低为内联 PTX `cp.async` + `nvvm.cp.async.commit.group`**：

```mlir
// 变换后（每个元素一条内联 PTX，16 个线程 × 16 列 = 128 列，但每线程只处理部分）
%matrix_x_3141 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = []
    "cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4, $2;", "r,l,r"
    %matrix_x_3135, %matrix_x_3136, %matrix_x_3140 : (!llvm.ptr<3>, !llvm.ptr<1>, i32) -> !llvm.void
// ...（共 16 条 cp.async 对应激活矩阵 128×16 中每线程负责的元素）
nvvm.cp.async.commit.group
```

**`ttg.async_wait {num=4}` 降低为 `nvvm.cp.async.wait.group`**：

```mlir
nvvm.cp.async.wait.group 4   // 等待直到 in-flight async group 数量 <= 4
```

**`tt.dot` 降低为 1024 个 `llvm.intr.fmuladd`**：

```mlir
// 变换后（tt.dot 128×16 × 16×64 → 每个线程 1024/128 = 8 次累加 × 循环体 = 1024 总计）
%acc_11696 = llvm.intr.fmuladd(%acc_11376, %acc_11632, %acc_11312) : (f32, f32, f32) -> f32
%acc_11697 = llvm.intr.fmuladd(%acc_11377, %acc_11636, %acc_11696) : (f32, f32, f32) -> f32
// ...（共 1024 个 fmuladd，对应单次迭代中 128×16 × 16×64 矩阵乘法的线程级计算）
```

**shared memory 全局符号声明**：

```mlir
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// 地址空间 3 = CUDA shared memory
// 实际大小由 PTX 后端根据 ttg.shared = 36864 填充
```

## 说明

此 Pass 是整个 Triton 编译流水线行数爆增的原因。54 倍的行数增长来自多个维度的展开：

1. **线程级展开**：每个 `tensor<128x...>` 操作按 `sizePerThread` 展开。`#blocked3`（`sizePerThread=[4,4]`, `warpsPerCTA=[4,1]`）表示每线程处理 4×4=16 个元素，4 个 warp × 32 线程/warp = 128 线程共同处理 128 行；但 `#blocked`（`sizePerThread=[4,4]`）在 128×64 的累加器中给每线程 64 个 f32，需要 64 次 `insertvalue` 初始化。

2. **`tt.dot` 展开**：`128×16 × 16×64 = 128×64` 的矩阵乘在每线程拥有 64 个输出元素、16 个内积元素的情况下，变成 64 × 16 = 1024 个标量乘加运算（`llvm.intr.fmuladd`），将在 PTX 后端映射为 `fma.rn.f32` 指令。

3. **`cp.async` 展开**：`ttg.async_copy_global_to_local` 针对 128×16 的激活矩阵，128 个线程每线程复制 16 个 f32，每次复制 1 个 4 字节元素（`0x4` = 4 bytes），因此每次 prologue load 展开为 16 条内联 PTX `cp.async` 指令 + 1 条 `commit.group`。

4. **软件流水线展开**：prologue（2 次）+ 稳态循环体 + epilogue 三段式结构中，每段的 cp.async 都被完全展开，贡献了多处的 `nvvm.cp.async.commit.group`（共出现 8 次）。

`nvvm.kernel = 1` 标注使 LLVM 后端将此函数生成为 PTX kernel（`.entry` 指令），`nvvm.reqntid = array<i32: 128>` 对应 PTX 的 `.reqntid 128, 1, 1`（请求恰好 128 个线程 per block，匹配 `num_warps=4 × 32 = 128`）。
