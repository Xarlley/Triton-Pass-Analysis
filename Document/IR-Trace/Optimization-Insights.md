# 优化洞察：VGG16-SNN 在 Triton 中的关键事实核查

> 本文针对 SNN 编译优化（`dev-log/dev-plan.md` §2.1 / §2.2）涉及的关键说法，基于
> `Document/IR-Trace/` 下捕获的**真实 VGG16-SNN 推理 IR** 逐一核查取证。每条结论都
> 配有指向真实 IR 文件的链接（捕获方法与等价性见 [README.md](./README.md)）。
>
> 涉及的 4 个代表 kernel 及其逐 Pass 跟踪 / 各阶段 IR：
> [卷积](./convolution/00_index.md)｜[BN+LIF](./bn_lif/00_index.md)｜
> [MaxPool](./maxpool/00_index.md)｜[矩阵乘法](./matmul/00_index.md)。
> 每个 kernel 目录下：`stage_0_entry.ttir` → `stage_1_final.ttgir` → `stage_2.llir`
> → `stage_3.ptx` → `stage_4.sass`。

---

# 第一部分：时间步结构 —— 「时间拆分」还可行吗？

## 1.0 摘要

| 问题 | 结论 |
|---|---|
| **时间步循环还存在吗？何时消失？** | **不存在。** 任何 Triton IR（TTIR→LLVM IR）里都没有时间步循环。它不是被某个 Triton Pass 删除的——在**进入 Triton 之前**就已不是循环：无状态层的时间维被 spikingjelly `seq_to_ann_forward` 折叠进 batch 维；LIF 的时间循环被 **TorchDynamo 在图捕获时静态展开**。 |
| **还能精确界定每个时间步吗？** | **早期勉强、且只是隐式；中后期不能。** TTIR/早期 TTGIR 里时间步是「batch 维固定区段」或「内联展开的 4 份重复代码」，可凭索引算术/模式识别勉强辨认；经 CSE、规范化、指令重排、`ConvertTritonGPUToLLVM` 后，4 个时间步被合并/交错/标量化，边界消失。 |
| **「时间拆分 + 分块编译优化」可行吗？** | **作为 Triton GPU Pass：基本不可行**——到了 Triton 已无「时间」这个可操作结构。该思路要成立必须上移到 Inductor / 图层。 |

## 1.1 时间步在这个 SNN 里是怎么来的

`examples/vgg16_snn/vgg16_test.py` 用 spikingjelly 多步模式（`step_mode='m'`，`T=4`），
输入 `x : [T, N, C, H, W] = [4, 1, 3, 224, 224]`。多步模式下：

- **无状态层**（Conv / BN / MaxPool / Linear）：经 `seq_to_ann_forward` 把 `[T,N,...]`
  **展平成 `[T*N,...]`**，做一次前向再 reshape 回去。
- **LIF 脉冲神经元**：`multi_step_forward` 里 `for t in range(T)` 逐步对膜电位充电、
  发放、复位，膜电位 `v` 跨时间步携带——**这才是真正的「时间步循环」**。

## 1.2 证据：逐 Pass 核查全部 IR 记录

### 1.2.1 整条流水线没有任何与时间相关的循环

扫描 4 个 kernel 从 TTIR 到 TTGIR 的 `scf.for`：

| kernel | 入口 TTIR | 最终 TTGIR | 唯一的循环是什么 |
|---|:--:|:--:|---|
| [BN+LIF](./bn_lif/stage_0_entry.ttir) | 0 | [0](./bn_lif/stage_1_final.ttgir) | 无任何循环 |
| [MaxPool](./maxpool/stage_0_entry.ttir) | 0 | [0](./maxpool/stage_1_final.ttgir) | 无任何循环 |
| [卷积](./convolution/stage_0_entry.ttir) | 1 | [1](./convolution/stage_1_final.ttgir) | [`scf.for %ijk = 0 to 9`](./convolution/stage_0_entry.ttir#L52)——卷积 **K 维归约**，与 T 无关 |
| [矩阵乘法](./matmul/stage_0_entry.ttir) | 1 | [1](./matmul/stage_1_final.ttgir) | `scf.for %k_idx = 0 to 128`——矩阵乘 **K 维归约**，与 T 无关 |

- 两个逐元素 kernel 自始至终**一个循环都没有**。
- 两个模板 kernel 各有且仅有 **1 个循环**，是卷积 / 矩阵乘的 **K 维归约**（迭代 9、
  128 次，均 ≠ T=4）。
- TTIR 的 `scf.for` 数 == TTGIR 的 `scf.for` 数 —— 对照各 kernel 的 73-Pass 流水线表
  （[卷积索引](./convolution/00_index.md)、[BN+LIF 索引](./bn_lif/00_index.md) 等），
  **没有任何一个 Pass 引入或删除过循环**。不存在「某个 Pass 把时间步循环优化掉了」。

### 1.2.2 LIF 的时间步循环：进入 Triton 前已被展开成内联直线代码

Inductor 生成 BN+LIF kernel 时附带的 ATen 图片段（来自 `output_code`）：

```
# Source Nodes: [..., v, ge, spike, ..., v_1, ..., v_2, ge_1, spike_1, ...,
#                v_3, ..., v_4, ge_2, spike_2, ..., v_5, ..., v_6, ge_3, spike_3, ..., v_7]
```

`v, v_1, …, v_7` 是膜电位在各时间步的连续版本，`spike … spike_3` 是 4 个时间步的脉冲，
`ge … ge_3` 是 4 次 Heaviside 发放比较。**`for t in range(4)` 已被 TorchDynamo 完全
展开**——4 个时间步变成一串内联直线节点，全部融合进**同一个 kernel**。

入口 TTIR 印证 ——
[`bn_lif/stage_0_entry.ttir`](./bn_lif/stage_0_entry.ttir) 第
[57](./bn_lif/stage_0_entry.ttir#L57)、[66](./bn_lif/stage_0_entry.ttir#L66)、
[75](./bn_lif/stage_0_entry.ttir#L75)、[84](./bn_lif/stage_0_entry.ttir#L84) 行：

```mlir
%tmp4  = arith.cmpf oge, %tmp2,  %cst_1 : tensor<16x64xf32>   // 时间步 0 发放
%tmp16 = arith.cmpf oge, %tmp15, %cst_1 : tensor<16x64xf32>   // 时间步 1 发放
%tmp27 = arith.cmpf oge, %tmp26, %cst_1 : tensor<16x64xf32>   // 时间步 2 发放
%tmp38 = arith.cmpf oge, %tmp37, %cst_1 : tensor<16x64xf32>   // 时间步 3 发放
```

整整 4 次发放阈值比较（`v ≥ v_threshold`），等间距内联排布；kernel 本身在二维 grid
上逐元素执行、**没有循环**——4 个时间步是 kernel 体内**展开的直线代码**。

### 1.2.3 无状态层：时间维被折叠进 batch 维

卷积 kernel 的 ATen 片段（`output_code`）：

```
%view : f32[4, 3, 224, 224]  = reshape(%arg0_1 : f32[4, 1, 3, 224, 224], [4, 3, 224, 224])
%convolution : f32[4, 64, 224, 224] = aten.convolution(%view, ...)
```

`seq_to_ann_forward` 把 `[T=4,N=1,C,H,W]` 展平为 `[T*N=4,C,H,W]`。此后「时间」只是
张量第 0 维（大小 4），与普通 batch 维无异。
[`convolution/stage_0_entry.ttir`](./convolution/stage_0_entry.ttir) 里唯一的循环
[`scf.for 0..9`](./convolution/stage_0_entry.ttir#L52) 是 K 维归约——没有一处是时间循环。

## 1.3 问题一：时间步循环还存在吗？从什么时候消失的？

**不存在于任何 Triton IR 中，也不是被某个 Triton Pass 删除的。** 消失发生在两个上游环节：

1. **spikingjelly 框架层**：无状态层的时间维从一开始就不是循环——`seq_to_ann_forward`
   把 `[T,N,...]` 展平进 batch。
2. **TorchDynamo 图捕获阶段**：LIF 的 `for t in range(T)` 因 `T=4` 是 Python 常量被
   **静态展开**，循环在这一步消失。

到达 Triton 入口（[各 kernel 的 `stage_0_entry.ttir`](./bn_lif/stage_0_entry.ttir)）时，
时间步已是「无状态层的一个 batch 维 + LIF 内联展开的 4 份直线代码」。**73 个 Pass 的
流水线里没有时间步循环，自然也没有哪个 Pass 让它消失。**

## 1.4 问题二：经过这些 Pass，还能精确界定每个时间步吗？

- **无状态层**：时间步 = batch 维固定区段，可凭索引算术界定，但 **T 与 N 已被合并**
  （`seq_to_ann_forward` 展平的是 `T*N`），IR 里只有一个大小为 4 的维度、无任何标注，
  界定是**隐式**的、要靠 IR 之外的知识。
- **LIF**：
  - 入口 [TTIR](./bn_lif/stage_0_entry.ttir) 与最终
    [TTGIR](./bn_lif/stage_1_final.ttgir) 里都保持 **4 次 `cmpf`**，凭重复模式尚能辨认。
  - 但 [`CSEPass`](./bn_lif/pass_26_CSEPass.md)、`CanonicalizerPass`、
    `TritonGPUReorderInstructions` 会合并/重排 4 份展开代码间的公共子表达式，时间步的
    外围代码开始交融。
  - 经 [`ConvertTritonGPUToLLVM`](./bn_lif/pass_63_ConvertTritonGPUToLLVM.md) 之后，
    张量操作被标量化：[`bn_lif/stage_2.llir`](./bn_lif/stage_2.llir) 里发放比较从
    TTGIR 的 4 次 `arith.cmpf` 变成 **25 次 `fcmp`**（混入 NaN 检查等），4 个时间步的
    指令被打散、交错。**到这一步时间步边界已不可精确界定。**

**结论**：TTIR/早期 TTGIR 阶段勉强可隐式界定；经 CSE、重排、LLVM 下降后实际不可界定。

## 1.5 问题三：「时间拆分 + 分块编译优化」的可能性 / 必要性 / 可行性

| 维度 | 结论 |
|---|---|
| **可行性（作为 Triton GPU Pass）** | **基本不可行**。到 Triton GPU Pass 这层已无「时间」这个可操作结构：无状态层时间是 batch 维一段下标、kernel 是覆盖整个 `[4,...]` 的单一编译产物；LIF 时间是内联展开且被 CSE 部分抹平的直线代码。「对不同时间块单独应用编译策略」要求时间块是**各自独立的 kernel**，而 kernel 边界由 **Inductor** 决定。 |
| **正确层级** | **TorchInductor / 图层**——那里时间步仍是独立节点（`v`、`v_1`…，每步一个 `select`），可在调度阶段阻断跨时间块融合、为各时间块 kernel 指定独立配置。 |
| **必要性** | 弱。稀疏度是运行期属性、编译期看不到；纯时间拆分（切 kernel）增益有限（Inductor 已按 kernel autotune）。曾设想「利用稀疏度做零值检测」更值得，但**第四部分**用真实训练 SNN 的实测同样否定了这条路。 |

## 1.6 在 Triton kernel 源码层面：时间步循环已展开为直线代码

§1.2–§1.5 看的是 IR（从 TTIR 起）。再往上一层——Inductor 交给 Triton 的**第一手产物，
即 `@triton.jit` kernel 源码**——同样**没有时间步循环**。

以 BN+LIF kernel（[`All-Kernels.md` buffer #4](./All-Kernels.md#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4)）
为例，源码里没有任何 `for`，时间步表现为 **4 条硬编码偏移的 `tl.load` + 重复 4 遍的
LIF 计算**：

```python
tmp0  = tl.load(in_ptr0 + (x1 + 64*y0), ...)            # 时间步 0
tmp11 = tl.load(in_ptr0 + (3211264 + x1 + 64*y0), ...)  # 时间步 1
tmp22 = tl.load(in_ptr0 + (6422528 + x1 + 64*y0), ...)  # 时间步 2
tmp33 = tl.load(in_ptr0 + (9633792 + x1 + 64*y0), ...)  # 时间步 3
```

偏移步长 `3211264 = 64×224×224` 恰是一个时间步的元素数。这份源码 1:1 翻译进 TTIR
（[`bn_lif/stage_0_entry.ttir`](./bn_lif/stage_0_entry.ttir)）后仍是 4 条 `tt.load` +
4 次发放比较，右列链接指向真实 IR 的具体行号：

| `@triton.jit` 源码（kernel #4） | TTIR（`bn_lif/stage_0_entry.ttir`） | 时间步 |
|---|---|---|
| `tmp0 = tl.load(in_ptr0 + (x1+64*y0), ...)` | [`%tmp0_19 = tt.load`（L40）](./bn_lif/stage_0_entry.ttir#L40) | 步 0 取数 |
| `tmp4 = tmp2 >= tmp3` | [`%tmp4 = arith.cmpf oge`（L57）](./bn_lif/stage_0_entry.ttir#L57) | 步 0 发放 |
| `tmp11 = tl.load(in_ptr0 + (3211264+...), ...)` | [`%tmp11_24 = tt.load`（L45）](./bn_lif/stage_0_entry.ttir#L45) | 步 1 取数 |
| `tmp16 = tmp15 >= tmp3` | [`%tmp16 = arith.cmpf oge`（L66）](./bn_lif/stage_0_entry.ttir#L66) | 步 1 发放 |
| `tmp22 = tl.load(in_ptr0 + (6422528+...), ...)` | [`%tmp22_29 = tt.load`（L50）](./bn_lif/stage_0_entry.ttir#L50) | 步 2 取数 |
| `tmp27 = tmp26 >= tmp3` | [`%tmp27 = arith.cmpf oge`（L75）](./bn_lif/stage_0_entry.ttir#L75) | 步 2 发放 |
| `tmp33 = tl.load(in_ptr0 + (9633792+...), ...)` | [`%tmp33_34 = tt.load`（L55）](./bn_lif/stage_0_entry.ttir#L55) | 步 3 取数 |
| `tmp38 = tmp37 >= tmp3` | [`%tmp38 = arith.cmpf oge`（L84）](./bn_lif/stage_0_entry.ttir#L84) | 步 3 发放 |

**结论**：在 Triton 收到的最上层产物（`@triton.jit` 源码）里，时间步循环就已是展开形态
——4 份带常量偏移的直线代码，没有任何 `for`。卷积 / 矩阵乘 kernel 连这 4 份重复都看不到：
时间被折叠进 batch 维（§1.2.3）。`@triton.jit` 源码与 TTIR 的完整逐行对应见
[`All-Kernels.md` 第 4 节](./All-Kernels.md)。

---

# 第二部分：寄存器与膜电位 —— 「膜电位必然导致寄存器溢出」吗？

## 2.1 待核查的说法

> *「不再存在时间步的情况下，代表神经元膜电位的中间变量必然涉及到寄存器溢出，
> 因为寄存器数量不可能容纳整个神经网络。」*

## 2.2 摘要

**说法不成立。** 真正持有膜电位的 kernel（BN+LIF、MaxPool）寄存器溢出为零；唯一发生
溢出的卷积 kernel 根本不含膜电位。

## 2.3 证据：真实 cubin 资源占用 + SASS 溢出指令

寄存器分配与溢出由 **ptxas** 在 PTX→SASS→cubin 时决定。`STL`（Store Local）/
`LDL`（Load Local）是 SASS 中的实际溢出指令，其数量直接等于溢出强度。

| kernel | 是否含膜电位 | 寄存器 (cubin REG) | spill 栈帧 (STACK) | SASS 溢出指令 | 是否溢出 |
|---|---|:--:|:--:|---|:--:|
| [卷积](./convolution/stage_4.sass) | ❌ 无 | **255**（撞 sm_120 硬件上限）| 32 B | [STL=8, LDL=8](./convolution/stage_4.sass) | ✅ 溢出 |
| [**BN+LIF**](./bn_lif/stage_4.sass)（含 `v,v₁…v₇`）| ✅ 有 | **56** | 0 | [STL=0, LDL=0](./bn_lif/stage_4.sass) | ❌ 不溢出 |
| [**MaxPool**](./maxpool/stage_4.sass)（含膜电位）| ✅ 有 | **40** | 0 | [STL=0, LDL=0](./maxpool/stage_4.sass) | ❌ 不溢出 |
| [矩阵乘法](./matmul/stage_4.sass) | ❌ 无 | 54 | 0 | [STL=0, LDL=0](./matmul/stage_4.sass) | ❌ 不溢出 |

（寄存器/栈帧数取自 `cuobjdump --dump-resource-usage`；STL/LDL 取自上面链接的真实
`stage_4.sass`。）

真正持有膜电位的 [BN+LIF](./bn_lif/stage_4.sass) 与 [MaxPool](./maxpool/stage_4.sass)
kernel **溢出为零**，仅用 56 / 40 个寄存器、远低于 255 上限。唯一溢出的
[卷积 kernel](./convolution/stage_4.sass) **不含任何膜电位**。

## 2.4 说法为何不成立

1. **「寄存器容纳不下整个网络」是真的，但不相关。** 没有任何一个 kernel 试图容纳整个
   网络。VGG16-SNN 被 Inductor 拆成 ~48 个**逐层 kernel**，每个只处理一个 **tile**。
   [BN+LIF kernel](./bn_lif/stage_0_entry.ttir) 只持有一个 `16×64` tile、4 个时间步的
   膜电位——一共 56 个寄存器就够。

2. **时间步展开 ≠ 所有时间步膜电位同时存活。** LIF 是递推链 `v_t = f(v_{t-1}, x_t)`：
   `v` 在 `v₁` 算出后即死亡。编译器活跃性分析任一时刻只保留 ~1–2 个膜电位。
   [BN+LIF kernel](./bn_lif/stage_0_entry.ttir) 把 4 个时间步全展开（见 §1.2.2 的 4 次
   `cmpf`），仍只用 56 寄存器——若 8 份 `v` 真要同时存活早就溢出了，事实是没有。

## 2.5 真实发生的那次溢出（卷积 kernel）：在哪一步、什么机制

虽然说法对膜电位不成立，但卷积 kernel **确实溢出了**。给出这一例的「在哪一步、什么
机制」：

- **在哪一步**：溢出由 **ptxas** 在 **`make_cubin` 阶段**完成——编译的**最后一步**，
  在本目录记录的 73 个 Pass **之后**（见 [卷积 00_index.md](./convolution/00_index.md)：
  流水线止于 `ConvertNVVMToLLVMPass`，其后才是 LLVM→PTX→ptxas）。
  证据：[`convolution/stage_3.ptx`](./convolution/stage_3.ptx) 中 `.local / st.local /
  ld.local` 计数**全为 0**（PTX 用无限虚拟寄存器 `%r0,%r1,…`）；溢出指令 `STL/LDL`
  **只出现在 [`stage_4.sass`](./convolution/stage_4.sass) 里**。PTX→SASS 由 ptxas 做，
  所以**物理寄存器分配与溢出纯粹是 ptxas 的决定**。

- **溢出机制**：ptxas 做物理寄存器分配，当单线程寄存器压力超过预算时，把放不下的值
  **溢出到 local memory**（`.local` 地址空间——每线程私有、实为设备 DRAM、经 L1/L2
  缓存）。被换出时发 `STL`（存 local），需要时发 `LDL`（取 local）。卷积 kernel：每
  线程 32 字节栈帧（8 个 4 字节槽）、8 次 `STL` + 8 次 `LDL`。

- **为什么是卷积**：它是矩阵乘模板 kernel——[`128×64` 的 f32 累加器
  tile](./convolution/stage_0_entry.ttir#L52)（8192 元素 / 128 线程 = 光累加器就
  64 寄存器/线程）+ `num_stages=4` 软件流水的预取缓冲 + 指针/索引/K 循环状态，一起把
  它顶到 255 寄存器上限 → 轻度溢出。这是矩阵乘经典的**寄存器压力 / occupancy 权衡**，
  与膜电位、与「整个网络」都无关。

---

# 第三部分：对 dev-plan §2.1 的综合启示

两次核查指向同一个根本结论：

1. **§2.1「时间拆分」不应实现为 TritonGPU C++ Pass。** 进入 Triton 后时间步既无循环、
   也难以精确界定（第一部分）。如确需按时间块分别编译，应实现为 **TorchInductor 自定义
   图层 Pass**：在调度阶段阻断跨时间块融合、为每个时间块 kernel 指定独立编译配置。

2. **§2.1「空间拆分」（避免寄存器溢出的最大分块尺寸）同样够不着。** 真实溢出确实存在
   （[卷积 kernel](./convolution/stage_4.sass)），但：(a) tile 尺寸由 Inductor 模板
   自动调优决定；(b) 寄存器分配与溢出发生在 **ptxas**（§2.5），在所有 73 个 Pass
   之后。TritonGPU Pass 这一层既改不了 tile、也管不到 ptxas 的寄存器分配。

3. **膜电位本身不是寄存器问题。** 它被 Inductor 的逐层拆分 + LIF 递推的短活跃区间
   自然控制在几十个寄存器内（第二部分）。§2.1 不必为「膜电位溢出」做专门设计。

4. **曾把「利用脉冲稀疏度」（§2.2 基于块的零值检测）寄望为收益来源**——但**第四部分**
   对真实训练 SNN 的实测否定了这条路：脉冲张量虽 80% 为零，可整块跳过的工作量却只有
   1.6%，在稠密 GPU 上是负优化。

> 一句话：现有 `MyNoOpPass`（TritonGPU C++ Pass）路线对 §2.1 的时间/空间拆分处在
> **错误的层级**（应上移到 Inductor 图层）；而 §2.2 的零值检测经实测在 GPU 上不可行
> （第四部分）。

---

# 第四部分：脉冲稀疏度实测与块级零值检测（dev-plan §2.2）

第三部分一度把「利用脉冲稀疏度」寄望为真正的收益来源。但稀疏度能否转化为 GPU 加速，
必须用**真实数据**检验，而非假设。为此训练了一个真实的 ImageNet SNN 并实测。

## 4.1 测量对象：一个真实训练的 T=4 VGG16-SNN

[`examples/vgg16_snn/finetune_snn.py`](../../examples/vgg16_snn/finetune_snn.py)：
以 torchvision ImageNet 预训练 VGG16-BN 权重初始化，逐层数据驱动地校准 LIF 阈值
（消除 ANN→SNN 的全层静默坍缩），再用替代梯度 BPTT + TET 损失微调，得到
**top-1 ≈ 46.6%** 的 T=4 脉冲网络。
[`measure_spike_sparsity.py`](../../examples/vgg16_snn/measure_spike_sparsity.py)
在 1024 张 ImageNet val 图上用 forward hook 统计全部 15 个 LIF 层的脉冲张量。

## 4.2 实测：逐层脉冲稀疏度，以及零值是否「结构化」

逐元素的高稀疏度要在 GPU 上转化为加速，前提是零值**成块出现**：GPU 以 32 线程的
warp 锁步执行、又靠 Tensor Core 做固定尺寸的稠密 MMA——只有「一整段归约块全为零」
才可能整块跳过而不引入 warp 发散；逐元素散布的零值无法跳过。

故在测发放率的同时，沿通道（卷积/矩阵乘的归约维）切成连续 32 元素一组，统计「整块
全零」的比例，并与「零值独立同分布」时的理论值 `(1-发放率)^32` 对比：

| 层 | 发放率 | 稀疏度 | 全零 32 块率（实测） | 实测/i.i.d. |
|---|---|---|---|---|
| LIF0  | 21.6% | 78.4% | 0.003% | 0.1× |
| LIF1  | 22.1% | 77.9% | 0.065% | 1.9× |
| LIF2  | 23.9% | 76.1% | 0.033% | 2.0× |
| LIF3  |  9.5% | 90.5% | 5.64%  | 1.4× |
| LIF4  | 25.0% | 75.0% | 0.006% | 0.6× |
| LIF5  | 25.1% | 74.9% | 0.007% | 0.7× |
| LIF6  | 21.0% | 79.0% | 0.34%  | 6.4× |
| LIF7  | 13.8% | 86.2% | 3.53%  | 4.1× |
| LIF8  | 16.3% | 83.7% | 1.79%  | 5.2× |
| LIF9  |  8.9% | 91.1% | 13.65% | 2.7× |
| LIF10 | 17.1% | 82.9% | 0.59%  | 2.4× |
| LIF11 | 12.9% | 87.1% | 6.86%  | 5.7× |
| LIF12 |  8.8% | 91.2% | 37.42% | 7.1× |
| LIF13（分类器） |  9.0% | 91.0% | 19.83% | 4.1× |
| LIF14（分类器） |  5.7% | 94.3% | 16.47% | 1.1× |
| **整网** | **19.95%** | **80.05%** | **1.61%** | — |

三个关键事实：

1. **逐元素稀疏度确实很高**：整网平均发放率 19.95%，即 **80% 的神经元-时间步为零**。
2. **但可整块跳过的工作量极少**：即便用最宽松的 32 元素粒度，整网仅 **1.6%** 的归约块
   为全零——这就是块级零值检测能跳过的工作量**上限**（粒度取 64 只会更低）。
3. **零值轻度成簇、但远不足以利用**：实测/i.i.d. = 1~7×，零值比纯随机分布更集中，
   说明有一点结构；可结构最明显的是末端小层（LIF12 37%、LIF13 20%），而**承担绝大
   部分卷积 FLOPs 的前中段大特征图层（LIF0–6）几乎没有可跳过的块**（0.003%–0.34%）。
   能跳的地方没多少计算量，有计算量的地方跳不动。

## 4.3 结论：块级零值检测在稠密 GPU 上是「易致负优化」的做法

| 维度 | 结论 |
|---|---|
| **必要性** | **表面有、实则无**。80% 的乘加确为「乘以 0」，看似浪费；但 GPU 的 FMA 乘 0 与乘任何数同价、同延迟，不整块跳过就一点省不下来。 |
| **可行性** | **差**。可整块跳过的工作量上限仅 1.6%，且集中在 FLOPs 占比很小的末端层；FLOP 大户的前中段层近乎 0。而零值检测的代价（每个归约块一次「是否全零」判定 + 数据依赖分支，并破坏 Tensor Core 稠密 MMA 的规整调度）要在 **100%** 的块上付出。代价覆盖全部、收益不足 1.6%。 |
| **好思路 / 易致负优化** | 在稠密 SIMT GPU + 本稀疏度分布（80%、非结构化）下，块级零值检测**几乎必然是负优化**——典型的「看上去省了 80% FLOPs、实际越优化越慢」。 |

零值检测要真正成立，需满足下列之一，本场景**均不满足**：

- 稀疏度**结构化**（整 tile / 整通道成块为零，跳过不引发散）——实测可跳块仅 1.6%；
- 稀疏度**极高**（>99%，非结构化稀疏 GEMM 才可能胜过稠密）——实测仅 80%；
- 目标是**事件驱动 / 神经形态硬件**（天然只处理脉冲事件）——本项目目标是 GPU + Triton。

> 一句话：脉冲张量 80% 为零是真的，但这 80% 在 GPU 上**取不出来**。dev-plan §2.2 的
> 「基于块的零值检测」对 GPU 后端不应实现——它把成本摊到每个块、收益却不足 1.6%。
> 至此，dev-plan 所列的时间拆分、空间拆分、（膜电位的）符号重物质化、块级零值检测
> 四条优化思路，经源码级与实测级核查**均已判定不值得或不可行**。
