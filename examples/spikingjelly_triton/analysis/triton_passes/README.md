# Triton 编译器内部 Pass 对照分析

> 所有文件均从 `/tmp/torchinductor_charlley/triton/0/WQZWHKDXGTUIK36K53HKSMUGNOKBSNYG4D7VHO6MHHNRK6PAKXGQ/` 真实提取。
> 对应的 Kernel 函数名：`triton_poi_fused__native_batch_norm_legit_functional_add_div_mul_native_batch_norm_backward_pow_reciprocal_rsub_sub_3`
> 该 Kernel 融合了 LIF 神经元的 ATan 替代梯度 + BatchNorm2d 反向传播的 Elementwise 部分。

## Triton 编译器 Pipeline（来自 `triton/python/triton/compiler/compiler.py`）

Triton 的 `compile()` 函数（第 226 行）注册并依次执行以下 Stage：

```python
# compiler.py 第 290-351 行的核心循环:
stages = dict()
backend.add_stages(stages, options, src.language)  # NVIDIA 后端注册: ttir→ttgir→llir→ptx→cubin
for ext, compile_ir in list(stages.items())[first_stage:]:
    next_module = compile_ir(module, metadata)      # 每个 stage 执行一次变换
    fn_cache_manager.put(next_module, f"{file_name}.{ext}")  # 每个阶段的输出写入缓存
```

这也解释了为什么缓存目录中包含所有中间文件。

---

## Pass 1: Triton IR (TTIR) → [`01_unoptimized.ttir`](./01_unoptimized.ttir)

**生成方式**：`ASTSource.make_ir()` → `code_generator.ast_to_ttir()`

这是从 `@triton.jit` Python DSL 代码生成的第一个 MLIR 表示。

**关键特征**：
- 无 GPU 硬件绑定：张量类型为 `tensor<256xf32>`（没有线程布局）
- 内存操作：`tt.load` / `tt.store`，地址通过 `tt.addptr` 计算
- 常量折叠已完成：`3.141592653589793` → `dense<3.14159274>`（f32 截断）

**Python Triton DSL → TTIR 映射示例**：

| Python (`03_triton_kernel.py`) | TTIR (`01_unoptimized.ttir`) |
|-------------------------------|------------------------------|
| `tl.program_id(0)` | `%xoffset = tt.get_program_id x : i32` |
| `tl.arange(0, XBLOCK)[:]` | `%xindex = tt.make_range {end=256, start=0}` |
| `tl.load(in_ptr0 + x3, mask)` | `%tmp0_9 = tt.load %tmp0_8, %xmask_5` |
| `tmp8 * tmp8` | `%tmp9 = arith.mulf %tmp8_28, %tmp8_28` |
| `1.0 / tmp10` | `%tmp12 = arith.divf %cst_0, %tmp10` |
| `tl.store(ptr, val, mask)` | `tt.store %ptr, %val, %mask` |

**ATan 替代梯度在 TTIR 中的表示**（行 61-81）：

```mlir
%tmp3  = arith.subf %cst_0, %tmp1_11  ; (1.0 - spike)
%tmp4  = arith.mulf %tmp0_9, %tmp3    ; grad * (1-spike)
%tmp6  = arith.subf %tmp5_13, %cst_0  ; (v - 1.0)
%tmp8  = arith.mulf %tmp6, %tmp8      ; (v-1) * π
%tmp9  = arith.mulf %tmp8_28, %tmp8_28 ; ((v-1)*π)^2
%tmp10 = arith.addf %tmp9, %cst_0     ; 1 + ((v-1)*π)^2
%tmp12 = arith.divf %cst_0, %tmp10    ; 1 / (1 + ...)   ← reciprocal
%tmp15 = arith.mulf %tmp12, %tmp14_15 ; * chain_grad
%tmp16 = arith.addf %tmp4, %tmp15     ; 两路梯度汇合
%tmp18 = arith.mulf %tmp16, %tmp18    ; * 0.5
```

---

## Pass 2: Triton GPU IR (TTGIR) → [`02_optimized_gpu.ttgir`](./02_optimized_gpu.ttgir)

**对应 Pass 名称**：`convert-triton-to-tritongpu`（NVIDIA 后端注册）

这是 Triton 最核心的优化 Pass：将抽象张量操作具体化为 GPU 线程层次结构。

**核心变化：新增块布局属性**

```mlir
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

**布局数学**：
- 每个 CTA 的线程数 = `threadsPerWarp × warpsPerCTA = 32 × 4 = 128`（`.reqntid 128` in PTX）
- 每个 CTA 处理的元素数 = `sizePerThread × threadsPerWarp × warpsPerCTA = 2 × 32 × 4 = 256 = XBLOCK`

**所有张量的类型标注从**：
```mlir
tensor<256xf32>      (TTIR)
```
**变为**：
```mlir
tensor<256xf32, #blocked>  (TTGIR)
```

**内存合并访问的保证**：`order = [0]` 表示最内层维度（即连续内存方向）是索引 0。同一 Warp 中的连续线程 ID 会访问连续的内存地址，GPU 内存控制器可以将 32 个 4 字节访问合并为 1 次 128 字节事务。

**注意**：TTGIR 中的数学逻辑与 TTIR 完全相同，只是所有张量类型增加了 `#blocked` 布局注解。

---

## Pass 3: LLVM IR → [`03_llvm.llir`](./03_llvm.llir)

**对应 Pass 名称**：`convert-triton-gpu-to-llvm`，由 Triton 的 LLVM lowering 层完成

**关键变化**：MLIR 高层语义降级为 LLVM 指令，并插入 NVIDIA GPU 专用内联汇编。

**向量化加载的实现**（对应 `sizePerThread=2`，每线程处理 2 个 f32）：

```llvm
; 一条 v2.b32 指令同时加载 2 个 float (8 字节)
%25 = tail call { i32, i32 } asm sideeffect
    "mov.u32 $0, 0x0;\09mov.u32 $1, 0x0;\09@$3 ld.global.v2.b32 { $0, $1 }, [ $2 + 0 ];",
    "=r,=r,l,b"(ptr addrspace(1) %24, i1 %20) #3
%26 = extractvalue { i32, i32 } %25, 0   ; 第 1 个 float (低 32 位)
%27 = extractvalue { i32, i32 } %25, 1   ; 第 2 个 float (高 32 位)
```

**L2 缓存驱逐策略实现**（对应 `eviction_policy='evict_last'`）：

```llvm
; 创建 L2 "最后驱逐" 缓存策略令牌
%56 = tail call i64 asm sideeffect
    "mov.u64 $0, 0x0;\09createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", "=l"()

; 带缓存策略的加载（per-channel 统计量，保留在 L2）
%57 = tail call i32 asm sideeffect
    "mov.u32 $0, 0x0;\09@$3 ld.global.L1::evict_last.L2::cache_hint.b32 { $0 }, [ $1 + 0 ], $2;",
    "=r,l,l,b"(ptr addrspace(1) %55, i64 %56, i1 %20)
```

**ATen `reciprocal` → LLVM 内置函数**：

```llvm
; aten.reciprocal 对应:
%102 = tail call float @llvm.nvvm.div.full(float 1.000000e+00, float %100)
```

---

## Pass 4: PTX Assembly → [`04_assembly.ptx`](./04_assembly.ptx)

**由 LLVM NVPTX 后端生成**，目标架构：`.target sm_120a`（NVIDIA Blackwell）

**关键 PTX 指令语义**：

| PTX 指令 | 语义 | 对应操作 |
|---------|------|---------|
| `mov.u32 %r23, %ctaid.x` | 读 CTA X 维度 ID | `tl.program_id(0)` |
| `shl.b32 %r24, %r23, 8` | `r24 = r23 << 8 = r23 * 256` | `xoffset = pid * XBLOCK` |
| `setp.lt.s32 %p1, %r28, 50176` | 设置谓词寄存器 | `xmask = xindex < xnumel` |
| `mul.hi.s32 %r29, %r28, 1402438301` | 乘以魔法数 (快速整数除法) | `xindex // 784` |
| `mul.f32 %r42, %r40, 0f40490FDB` | 乘以 π (IEEE 754 hex) | `(v-1) * pi` |
| `fma.rn.f32 %r44, %r42, %r42, 0f3F800000` | FMA: `r42*r42 + 1.0` | `(v-1)^2*π^2 + 1` |
| `div.full.f32 %r46, %r37, %r44` | 全精度浮点除法 | `1 / (1 + ...)` |
| `@%p1 ld.global.v2.b32 { %r1, %r2 }, [%rd1 + 0]` | 谓词向量化加载 | 带掩码的 `tl.load` |
| `@%p1 st.global.v2.b32 [%rd2 + 0], {%r21, %r22}` | 谓词向量化存储 | 带掩码的 `tl.store` |

**`mul.hi.s32` 快速整数除法**（第 58-67 行）：

编译器将 `xindex // 784` 和 `xindex % 16` 转换为乘法+移位序列（魔法数算法），避免了昂贵的 `div` 指令：
```ptx
mul.hi.s32 %r29, %r28, 1402438301  ; 1402438301 ≈ 2^39 / 784 (魔法数)
shr.u32    %r30, %r29, 31           ; 符号修正
shr.s32    %r31, %r29, 8            ; 移位完成除法
add.s32    %r32, %r31, %r30         ; 得到 quotient = xindex // 784
```

**FMA 优化**：`tmp8 * tmp8 + 1.0`（TTIR 中是两条独立指令）在 PTX 中被合并为：
```ptx
fma.rn.f32 %r44, %r42, %r42, 0f3F800000  ; x*x + 1.0f 单条指令
```

---

## 各层关键差异速览

| 特征 | TTIR | TTGIR | LLVM IR | PTX |
|-----|------|-------|---------|-----|
| 张量布局 | 逻辑形状 | `#blocked` 布局 | 标量 + 向量 | 寄存器 |
| 硬件线程意识 | 无 | 有 (Warp/CTA) | 有 (NVVM intrinsic) | 有 (ctaid, tid) |
| 内存操作 | `tt.load` | `tt.load + #blocked` | PTX ASM | `ld.global.v2.b32` |
| 除法 | `arith.divf` | `arith.divf` | `@llvm.nvvm.div.full` | `div.full.f32` |
| 整数除法 | `arith.divsi` | `arith.divsi` | `sdiv` | `mul.hi.s32` (魔法数) |
| FMA | 两条独立 `mul+add` | 两条独立 `mul+add` | 单 FMA intrinsic | `fma.rn.f32` |
