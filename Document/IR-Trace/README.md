# VGG16-SNN 在 Triton 中的逐 Pass IR 变换跟踪

本目录完整记录**真实 VGG16 脉冲神经网络（SNN）推理**的 Triton kernel 在编译期
经过的**每一个 Pass**，以及每一次 IR 变换的**变换前后中间表示（IR）**，一直跟踪到
LLVM IR / PTX。

## 1. 这是什么

`examples/vgg16_snn/vgg16_test.py` 跑一次推理时，`torch.compile` 会把 VGG16-SNN
拆成约 48 个 Triton kernel，每个 kernel 都走完全相同的 ~73 个编译 Pass
（TTIR → TTGIR → LLVM IR）。本目录挑选 **4 个代表性 kernel**，逐 Pass 记录其 IR 演变：

| 子目录 | kernel 类型 | 说明 |
|---|---|---|
| [`convolution/`](./convolution/00_index.md) | 卷积 | VGG16 首个卷积层，Triton 卷积模板（含 K 维归约循环）|
| [`bn_lif/`](./bn_lif/00_index.md) | BatchNorm + LIF | BN 归一化 + LIF 脉冲神经元（逐元素）|
| [`maxpool/`](./maxpool/00_index.md) | MaxPool + BN + LIF | 下采样 + BN + LIF 融合（逐元素）|
| [`matmul/`](./matmul/00_index.md) | 矩阵乘法 / 全连接 | 分类器 FC 层，Triton 矩阵乘法模板（含 K 维归约循环）|

每个子目录下：`00_index.md` 是该 kernel 的完整 Pass 流水线索引；`pass_NN_*.md` 是
每一次真实 IR 变换的详细说明；`stage_*.{ttir,ttgir,llir,ptx,sass}` 是各阶段完整 IR
产物。

**[`Optimization-Insights.md`](./Optimization-Insights.md)** —— 基于本跟踪记录与真实
训练 SNN 的关键事实核查：时间步循环是否还存在、能否精确界定时间步、膜电位是否导致
寄存器溢出、80% 的脉冲稀疏度能否被 GPU 利用，并据此给出对 `dev-log/dev-plan.md`
§2.1 / §2.2 各优化思路的实测结论。

**[`All-Kernels.md`](./All-Kernels.md)** —— 一次真实推理生成的**全部 48 个 Triton
kernel 的完整代码**，解释为何只有少数几种「操作」的网络会编译出 48 个 kernel。

**[`Inductor-Tile-Register-Strategy.md`](./Inductor-Tile-Register-Strategy.md)** ——
结合 TorchInductor 源码（`pytorch/` submodule）讲解：tile 尺寸在哪一步决定、为何逐元素
kernel 的寄存器不被占满、conv 的寄存器压力 / occupancy 权衡发生在哪。

## 2. 方法与等价性保证

直接对整个 VGG16 运行开 `MLIR_ENABLE_DUMP` 会因 `max_autotune` 的自动调优候选
kernel 爆炸到数十 GB。本跟踪采用**确定性重放**，保证所记录的 IR 与真实推理**逐字节
等价**：

1. **抓真实 IR**：用 `TRITON_KERNEL_DUMP` 跑一次真实 VGG16-SNN 推理，导出每个 kernel
   真实的 `ttir / ttgir / llir / ptx`。
2. **逐 Pass 重放**：把代表 kernel 的真实 `.ttir` 配上真实编译选项
   （`num_warps` / `num_stages` / `num_ctas`，从真实 `.ttgir` 与 Inductor 模板元数据
   读出），单独重新编译并开 `MLIR_ENABLE_DUMP`，捕获逐 Pass IR。
3. **对账验证**：重放产出的 TTGIR 与 PTX 与第 1 步的真实产物**逐字节比对一致**。

由于 Triton 编译器是确定性的——相同输入 IR + 相同选项 ⇒ 相同输出——逐字节一致即
证明：本目录记录的每一步 IR 变换，**就是**真实 SNN 推理时发生的变换。

4 个 kernel 的 TTGIR、PTX 均已通过该对账（见各 `00_index.md`）。

## 3. 编译流水线总览

跟踪覆盖从 **Triton IR (TTIR)** 进入优化流水线，到生成 **LLVM IR** 的全部 ~73 个 Pass：

```
@triton.jit kernel
   │  (前端 ast_to_ttir + make_ttir，已是各 stage_0 的起点)
   ▼
TTIR  ──[ ConvertTritonToTritonGPU ]──►  TTGIR
   │      绑定 GPU 线程层次（#blocked 布局、warp/CTA）
   ▼
TTGIR ──[ Coalesce / RemoveLayoutConversions / Pipeline / ... ]──►  优化后 TTGIR
   │      访存合并、布局传播、软件流水、warp 专精 ...
   ▼
优化 TTGIR ──[ ConvertTritonGPUToLLVM / ConvertNVGPUToLLVM / ... ]──►  LLVM IR
   │
   ▼
LLVM IR ──[ LLVM NVPTX 后端 ]──►  PTX ──[ ptxas ]──►  cubin（二进制）
```

每个 kernel 的 `00_index.md` 列出全部 73 个 Pass；其中真正改变了该 kernel IR 的
Pass 各有一篇 `pass_NN_*.md` 详述，未改变 IR 的 Pass 在该 kernel 上为 no-op。

## 4. 复现

```bash
# 1. 抓真实 IR（TRITON_KERNEL_DUMP）
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=<dir> python examples/vgg16_snn/vgg16_test.py
# 2. 逐 Pass 重放某 kernel（从真实 .ttir + 真实选项编译，开 MLIR_ENABLE_DUMP）
#    并与真实 .ttgir / .ptx 对账，详见各 00_index.md 的「来源」说明。
```
