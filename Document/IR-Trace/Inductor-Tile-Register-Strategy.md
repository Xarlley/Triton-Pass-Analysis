# TorchInductor 的 tile 尺寸与寄存器分配策略（源码级）

> 本文回答三个问题：(1) tile 尺寸在**哪一步**决定？(2) 为什么 bn_lif / maxpool 的实际
> 寄存器占用（56 / 40）**远低于上限 255**，分配策略为何不把寄存器占满？(3) conv 的
> 「寄存器压力 / occupancy 权衡」**发生在哪里**？
>
> 全部结论结合 **TorchInductor 源码**讲解。PyTorch 已作为 submodule 纳入本仓库
> （`pytorch/`），**精确固定在已安装运行的 commit `70d99e9`**（torch 2.11.0+cu130）。
> 已逐字节校验：submodule 中的 `torch/_inductor/runtime/triton_heuristics.py`、
> `kernel/conv.py`、`template_heuristics/triton.py`、`select_algorithm.py`、`choices.py`
> 与 conda 环境里**实际运行的源码完全一致**。下文所有 `pytorch/...#Lxx` 链接都指向真实
> 运行的源码。

## 0. 总览：tile 与寄存器分别由谁决定

```
        TorchInductor (PyTorch)                       Triton + ptxas
   ───────────────────────────────────         ───────────────────────────
   ① 生成 tile 候选配置                          ④ Triton 编译 → PTX
      逐元素：triton_heuristics.pointwise            （PTX 用无限虚拟寄存器）
      卷积  ：template_heuristics.conv_configs
                  │                              ⑤ ptxas：PTX → SASS
   ② 自动调优实测每个候选耗时                         物理寄存器分配 + 溢出
   ③ 取 min(耗时) 的配置 → 即最终 tile
```

**两条关键事实：**
- **tile 尺寸**（`XBLOCK/YBLOCK` 或 `BLOCK_M/N/K`）由 **TorchInductor** 决定——先生成
  一组候选配置，再用**自动调优实测、取最快的那个**。
- **寄存器数量**由 **ptxas** 决定——它是「tile 大小 + kernel 代码」的**后果**，不是任何
  策略追逐的目标。没有任何环节会去「占满寄存器」。

## 1. 问题一：tile 尺寸在哪一步决定？

### 1.1 逐元素 kernel（bn_lif / maxpool）

三步，全部在 `pytorch/torch/_inductor/runtime/triton_heuristics.py`：

**① 生成候选 tile 配置** —— `pointwise()`
[（triton_heuristics.py:2818）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L2818)。
对二维逐元素 kernel（bn_lif 即此类），它产出 6 个候选 `(XBLOCK, YBLOCK)`
[（L2935–2943）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L2935)：

```python
configs = [
    triton_config_with_settings(size_hints, 32, 32),
    triton_config_with_settings(size_hints, 64, 64),
    triton_config_with_settings(size_hints, 256, 16),
    triton_config_with_settings(size_hints, 16, 256),
    triton_config_with_settings(size_hints, bs, 1),
    triton_config_with_settings(size_hints, 1, bs),
]
```

**② 每个候选经 `triton_config()` 收敛**
[（triton_heuristics.py:2395）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L2395)：
把种子 `(x, y)` 按 `size_hints`（张量各维元素数）裁剪 / 放大，并计算 `num_warps`。核心
常量是 `num_elements_per_warp=256`
[（L2401）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L2401)——「一个 warp
约处理 256 个元素」；`num_warps` 由此推出
[（L2463–2466）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L2463)：

```python
num_warps = _num_warps(conditional_product(x, y, z) // num_elements_per_warp, min_num_warps=1)
```

以本仓库捕获的 **bn_lif kernel #4**（`size_hints={'y':65536,'x':64}`）为例：种子 `(bs,1)`
（`bs=1024`）经 `triton_config` 收敛——`x` 被 `size_hints['x']=64` 截到 64、`y` 放大到 16
（使 `x*y` 接近 `target`）——得到 **XBLOCK=64, YBLOCK=16**；`num_warps = _num_warps(1024//256=4)
= 4`。这与 [`bn_lif/stage_0_entry.ttir`](./bn_lif/stage_0_entry.ttir) 里的
`tt.make_range {end = 64}` / `{end = 16}` 完全一致。

**③ 运行期自动调优选最快** —— `CachingAutotuner`
[（triton_heuristics.py:313）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L313)。
首次启动该 kernel 时，`autotune_to_one_config()`
[（L1173）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L1173) 实测全部候选、
**取耗时最小者**：

```python
timings = self.benchmark_all_configs(*args, **kwargs)
self.launchers = [builtins.min(timings, key=timings.get)]      # ← 按「实测耗时」取最快
```
[（triton_heuristics.py:1176 / 1178）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L1178)

所以**逐元素 kernel 的最终 tile = 6 个候选里实测最快的那个**。

### 1.2 卷积 / 矩阵乘 kernel（模板 kernel）

同样是「候选配置 + 自动调优」，但候选来自模板启发式、调优发生在**编译期**。

**① 候选配置列表** —— `conv_configs`
[（template_heuristics/triton.py:633–654）](../../pytorch/torch/_inductor/template_heuristics/triton.py#L633)，
共 17 个 `ConvConfig(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)`：

```python
self.conv_configs: list[BaseConfig] = [
    ConvConfig(64, 256, 16, 2, 4),
    ConvConfig(256, 64, 16, 2, 4),
    ConvConfig(1024, 16, 16, 1, 8),
    ConvConfig(128, 128, 32, 2, 8),
    ...
    ConvConfig(128, 64, 64, 4, 4),      # ← conv #2 最终选中的就是这一个（L646）
    ...
    ConvConfig(128, 256, 128, 2, 8),
]
```

字段定义见 `BaseConfig`
[（L53–63）](../../pytorch/torch/_inductor/template_heuristics/triton.py#L53)。

**② lowering 把每个配置变成一个模板「choice」** —— `convolution()`
[（kernel/conv.py:592–644）](../../pytorch/torch/_inductor/kernel/conv.py#L592)：对
`V.choices.get_conv_configs()`（[choices.py:116](../../pytorch/torch/_inductor/choices.py#L116)
→ [template_heuristics/triton.py:1017](../../pytorch/torch/_inductor/template_heuristics/triton.py#L1017)）
返回的每个配置，调用 `conv2d_template.maybe_append_choice(...)` 生成一个候选。

**③ 编译期自动调优选最快** —— `select_algorithm.py` 的 `do_autotuning()`
[（L3003）](../../pytorch/torch/_inductor/select_algorithm.py#L3003)：把所有 choice 实测
耗时、取最快者。与逐元素一样是「实测 → 取 min」，只是发生在 `torch.compile` 编译期。

本仓库的 **conv kernel #2**：与捕获到的 `num_warps=4, num_stages=4`、累加器 `128×64`
唯一吻合的候选是 **`ConvConfig(128, 64, 64, 4, 4)`**
[（template_heuristics/triton.py:646）](../../pytorch/torch/_inductor/template_heuristics/triton.py#L646)；
其 `BLOCK_K=64` 经 `preprocess_mm_configs` 按实际归约维（首个卷积 `in_channels=3`）收缩为
16，于是 [`convolution/stage_0_entry.ttir`](./convolution/stage_0_entry.ttir) 里是
`tt.dot [128×16]×[16×64]→[128×64]`。

## 2. 问题二：为什么 bn_lif / maxpool 寄存器没占满？

**因为「占满寄存器」从来不是目标——寄存器数是后果，不是被分配策略追逐的指标。** 三层
原因，逐层看源码：

**(a) 启发式刻意把每线程工作量压小。** `triton_config` 的 `num_elements_per_warp=256`
[（triton_heuristics.py:2401）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L2401)
规定「一个 warp ≈ 256 元素」。bn_lif 最终 tile 64×16=1024 元素、num_warps=4 → 每 warp
256 元素 → **每线程仅 8 个元素**。每个元素的活跃中间量是个位数，ptxas 因此只需约 56 个
寄存器（maxpool 同理，约 40 个）。工作集小 → 寄存器需求自然小。

**(b) ptxas 分配的是「够用的最少」，不是「占满」。** PTX 用无限虚拟寄存器；ptxas 在
PTX→SASS 时做物理分配，目标是用**尽量少**的寄存器——寄存器用得越少，一个 SM 上能同时
驻留的 warp 越多（occupancy 越高）。对 bn_lif / maxpool 这种**访存受限**的逐元素 kernel，
高 occupancy 才能掩盖访存延迟。**把寄存器占满会压低 occupancy、反而更慢**，没有任何动机
这样做。

**(c) 自动调优按「实测耗时」选，不按寄存器占用选。** `autotune_to_one_config` 用
`min(timings, key=timings.get)`
[（triton_heuristics.py:1178）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L1178)
选配置。值得注意：紧接着的 `log.debug`
[（L1185 起）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L1185) 会打印选中
配置的 `n_regs` / `n_spills`——每个候选 launcher 都带有这两个字段，说明调优器**知道**
寄存器数，但**选择时完全不用它**，只用实测时间。一个「占满寄存器」的配置若更慢，就会
被淘汰。

唯一一处与寄存器相关的启发式是 `_num_warps` 的 `register_intensive` 参数
[（triton_heuristics.py:2352, 2359–2360）](../../pytorch/torch/_inductor/runtime/triton_heuristics.py#L2352)：
对寄存器密集的 persistent reduction，它把 `max_num_warps` **减半**。注意方向——是
**降低**寄存器压力，而非占满。

> 结论：bn_lif 的 56、maxpool 的 40 个寄存器，是「小工作集 + ptxas 取最少 + 调优按速度
> 选」三者的自然结果。56 / 40 远低于 255 **不是缺陷**，正是**高 occupancy 的体现**，对
> 访存受限的逐元素 kernel 而言是最优的。

## 3. 问题三：conv 的「寄存器压力 / occupancy 权衡」发生在哪？

这个权衡**不在某一个步骤**，而是分布在四处：

**① 模板源码——寄存器压力的物理来源。** conv 模板把累加器整块放在寄存器里：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```
[（kernel/conv.py:142）](../../pytorch/torch/_inductor/kernel/conv.py#L142)

`BLOCK_M × BLOCK_N` 越大，累加器占的寄存器越多。conv #2 的 `128×64` f32 累加器 = 8192
元素 / 128 线程（num_warps=4）= **每线程光累加器就 64 个寄存器**，再加 `num_stages=4`
的软件流水预取缓冲、指针 / 索引 / K 循环状态。

**② 候选配置列表——权衡点的枚举。** `conv_configs`
[（template_heuristics/triton.py:633–654）](../../pytorch/torch/_inductor/template_heuristics/triton.py#L633)
本身就是这条权衡曲线上的 17 个采样点：从 `ConvConfig(64, 64, 32, 2, 4)`（小累加器、低
寄存器压力、高 occupancy）到 `ConvConfig(256, 128, 128, 2, 8)`（大累加器、高寄存器压力、
强数据复用）。**「权衡」就编码在这张表里。**

**③ select_algorithm 自动调优——经验式地解出权衡。** Inductor **不用公式**算「寄存器 vs
occupancy」的最优点，而是把全部配置**逐个实测、取最快**（`do_autotuning`，
[select_algorithm.py:3003](../../pytorch/torch/_inductor/select_algorithm.py#L3003)）。哪个
配置在寄存器压力与 occupancy / 数据复用之间取得最好平衡，由**实测耗时**说话。

**④ ptxas——物理实现这个权衡。** 选中 `ConvConfig(128, 64, 64, 4, 4)` 后，Triton 编出
PTX，ptxas 做物理寄存器分配；该配置的寄存器需求顶到硬件上限 **255**，放不下的值溢出到
local memory（详见 [Optimization-Insights.md §2.5](./Optimization-Insights.md)：conv #2
每线程栈帧 32 B、SASS 中 8 条 `STL` + 8 条 `LDL`）。

> 一句话：conv 的权衡 = 模板（`conv.py:142` 决定寄存器怎么用）+ 候选表（`conv_configs`
> 枚举权衡点）+ 自动调优（实测选最优）+ ptxas（物理分配并在顶到 255 时溢出）。没有单一
> 的「权衡决策步」，它是这条链路的涌现结果。

## 4. 与 IR-Trace 实测的对应

| kernel | Inductor 选定的 tile | num_warps / stages | 实测寄存器 | 溢出 |
|---|---|---|---|---|
| [bn_lif](./bn_lif/stage_0_entry.ttir) | XBLOCK=64, YBLOCK=16 | 4 / – | 56 | 无 |
| [maxpool](./maxpool/stage_0_entry.ttir) | 1D pointwise | 8 / – | 40 | 无 |
| [convolution](./convolution/stage_0_entry.ttir) | BLOCK 128×64×16 | 4 / 4 | 255（占满）| 32 B / 8×STL+8×LDL |

寄存器实测数据与溢出证据见 [Optimization-Insights.md 第二部分](./Optimization-Insights.md)；
48 个 kernel 的完整代码见 [All-Kernels.md](./All-Kernels.md)。

## 5. 源码索引（均在 `pytorch/` submodule，commit `70d99e9`）

| 作用 | 文件 : 函数 / 行 |
|---|---|
| 逐元素候选 tile 生成 | `torch/_inductor/runtime/triton_heuristics.py` · `pointwise()` L2818 |
| 逐元素 tile 收敛 + num_warps | 同上 · `triton_config()` L2395；`num_elements_per_warp=256` L2401 |
| num_warps 钳制（`register_intensive`）| 同上 · `_num_warps()` L2352 |
| 逐元素运行期自动调优选最快 | 同上 · `CachingAutotuner.autotune_to_one_config()` L1173；`min(timings)` L1178 |
| conv 候选配置列表 | `torch/_inductor/template_heuristics/triton.py` · `conv_configs` L633–654 |
| conv 配置字段定义 | 同上 · `BaseConfig` L53–63 |
| conv lowering → 模板 choice | `torch/_inductor/kernel/conv.py` · `convolution()` L592–644 |
| conv 模板累加器（寄存器压力源）| 同上 · `acc = tl.zeros((BLOCK_M, BLOCK_N))` L142 |
| 模板编译期自动调优 | `torch/_inductor/select_algorithm.py` · `do_autotuning()` L3003 |
| 寄存器物理分配 + 溢出 | ptxas（`make_cubin` 阶段，不在 IR-Trace 的 73 个 Pass 内）|
