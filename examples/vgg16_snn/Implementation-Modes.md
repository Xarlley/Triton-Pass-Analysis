# VGG16-SNN 三种实现路径：代码走读与推理延迟测量复现指南

本目录下同一个 VGG16-SNN 已经有三种可跑通的推理实现，对应不同的"算子后端组合"：

| 路径 | LIF 后端 | Conv / Pool / GEMM 后端 | 是否 `torch.compile` | 入口脚本 |
|---|---|---|---|---|
| **A: Eager** | SpikingJelly 手写 Triton kernel | cuDNN / cuBLAS / ATen native | 否 | `vgg16_test.py` / `benchmark_inference.py`（不带 `COMPILE=1`）|
| **B: torch.compile + 全 Triton** | SpikingJelly 手写 Triton kernel | Inductor 自动生成的 Triton kernel（强制 `max_autotune_conv/gemm_backends=TRITON`）| 是 | `vgg16_test.py` / `benchmark_inference.py COMPILE=1` |
| **C: NIR roundtrip + Eager** | SpikingJelly 手写 Triton kernel | cuDNN / cuBLAS / ATen native（同 A）| 否 | `vgg16_via_nir.py` / `benchmark_compare.py MODE=B` |

注意三件事：
1. **LIF 后端在三种路径里都是同一份代码** —— SpikingJelly 自带的 `multistep_lif` Triton kernel（参见
   [SpikingJelly-Triton-Patch.md](SpikingJelly-Triton-Patch.md) 的 §3 修补；本仓库的 Triton fork 上必须有该
   patch，否则编译期就崩在 `convert_and_store`）。所谓"A / B / C 三种模式"，区别只在 **stateless 层**
   的 backend 与是否走 `torch.compile`。
2. **C 不只是 A 的子集** —— C 把 13 个 BN 通过 fold-BN 吸收进 Conv，并把 5 个 MaxPool 替换成 AvgPool
   （NIR 协议没有 MaxPool 原语）。所以 C 在数学上与 A、B 不等价；只用作"另一种实现该结构 SNN"的对照。
3. **本文档只讲推理（eval, no_grad）**；训练路径走的代码分支不一样。

---

## 1. 实现路径 A：Eager（cuDNN 卷积 + SJ Triton LIF）

### 1.1 代码组成

模型构造（[vgg16_test.py:61-87](vgg16_test.py#L61-L87)）：

```python
class VGG16SNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        ...
        self.features = self._make_features(VGG16_CFG)
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(512 * 7 * 7, 4096),
            neuron.LIFNode(),
            layer.Linear(4096, 4096),
            neuron.LIFNode(),
            layer.Linear(4096, num_classes),
        )

    @staticmethod
    def _make_features(cfg):
        layers, in_ch = [], 3
        for v in cfg:
            if v == 'M':
                layers.append(layer.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(layer.Conv2d(in_ch, v, kernel_size=3, padding=1))
                layers.append(layer.BatchNorm2d(v))
                layers.append(neuron.LIFNode())   # 替代 VGG 的 ReLU
                in_ch = v
        return nn.Sequential(*layers)
```

切到多步 + eval（[vgg16_test.py:109-115](vgg16_test.py#L109-L115)）：

```python
model = VGG16SNN(NUM_CLASSES)
functional.set_step_mode(model, 'm')   # 多步（multi-step）模式
model.eval()                           # 推理模式：BN 使用 running stats
return model.to(device)
```

### 1.2 单层 forward 在多步模式下做了什么

`set_step_mode(model, 'm')` 之后，每个 `layer.*` 模块的 `forward` 走 step_mode=='m' 分支，把
`[T, B, ...]` 在时间维和 batch 维上 flatten 成 `[T·B, ...]` 再调底下的 PyTorch 原生层。以
`layer.Conv2d` 为例（[layer/stateless_wrapper.py:176-190](../../spikingjelly/spikingjelly/activation_based/layer/stateless_wrapper.py#L176-L190)）：

```python
def forward(self, x: Tensor):
    if self.step_mode == "s":
        x = super().forward(x)
    elif self.step_mode == "m":
        if x.dim() != 5:
            raise ValueError(...)
        y_shape = [x.shape[0], x.shape[1]]
        y = super().forward(x.flatten(0, 1))      # super = nn.Conv2d
        y_shape.extend(y.shape[1:])
        x = y.view(y_shape)
    return x
```

`super().forward(...)` 就是 `nn.Conv2d.forward(...)`。这条调用栈最终进入 cuDNN —— 详见 §6。

`layer.AvgPool2d` / `layer.MaxPool2d` 同 pattern（[stateless_wrapper.py:882](../../spikingjelly/spikingjelly/activation_based/layer/stateless_wrapper.py#L882),
[L697](../../spikingjelly/spikingjelly/activation_based/layer/stateless_wrapper.py#L697)）；
`layer.Linear` 在该 SJ 版本里也是 `nn.Linear` 子类，多步同样 `flatten(0,1) → super().forward → view`。

**例外**：`layer.BatchNorm2d` 在多步模式下走的是 `functional.seq_to_ann_forward(x, self.super_forward)`
（[layer/bn.py:88](../../spikingjelly/spikingjelly/activation_based/layer/bn.py#L88) / [L215](../../spikingjelly/spikingjelly/activation_based/layer/bn.py#L215)）—— 这是后面 §2 中 `patch_spikingjelly_for_full_graph` 要打补丁的那条分支。

### 1.3 LIF 在 eval 模式下走哪条路

`neuron.LIFNode.multi_step_forward` 的 eval 分支（[neuron/lif.py:562-587](../../spikingjelly/spikingjelly/activation_based/neuron/lif.py#L562-L587)）：

```python
else:   # eval mode
    self.v_float_to_tensor(x_seq[0])
    if x_seq.is_cuda and getattr(self.surrogate_function, 'spiking', True):
        try:
            spike_seq, v_seq = triton_kernel.multistep_lif(
                x_seq, self.v, self.decay_input, self.tau,
                self.v_threshold, self.v_reset,
                self.detach_reset, self.surrogate_function,
            )
            ...
            return spike_seq
        except (NotImplementedError, AttributeError, TypeError, KeyError) as e:
            logging.debug("Falling back from Triton LIF kernel in eval: %s", e)
        ...
    return super().multi_step_forward(x_seq)   # Python loop fallback
```

`backend` 属性在 eval 路径里**没有被检查**（只在 training 路径用，
[lif.py:451](../../spikingjelly/spikingjelly/activation_based/neuron/lif.py#L451)）；只要 `eval + CUDA + spiking surrogate`
三条都满足，就直接调 `triton_kernel.multistep_lif`。

该函数是 `@register_op("sj::multistep_lif_forward")`（[triton_kernel/neuron_kernel/lif.py:281](../../spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py#L281)）注册的
`torch.library.custom_op`，内部 wrap 一个 `@triton.jit` kernel `_multistep_lif_forward_kernel`，
**把所有 T 个时间步在 GPU kernel 里展开**（fused-T-loop），中间膜电位 `v` 全程驻留寄存器/SRAM。

### 1.4 启动

```bash
cd examples/vgg16_snn

# 单次推理 + 黄金输出比对（vgg16_test.py 默认就走 eager；它带 torch.compile，
# 想拿 pure-eager 数据请用 benchmark_inference.py 不带 COMPILE=1）
python benchmark_inference.py 10000 50          # 10000 张样本，BATCH=50
```

注：`vgg16_test.py` 本身**总是开 torch.compile**（[L227-231](vgg16_test.py#L227-L231)），所以它不能用来跑"路径 A"。
真要测路径 A 的延迟，用 `benchmark_inference.py` 不带 `COMPILE=1`。

---

## 2. 实现路径 B：torch.compile + 全 Triton

### 2.1 关键差异（相对 A）

整张前向图（13 Conv + 13 BN + 15 LIF + 5 MaxPool + 3 FC）被 `torch.compile` 抓进 dynamo →
Inductor → Triton。每个 conv / gemm 被强制由 Inductor 用 `max_autotune` 现场生成 Triton kernel
（**不走 cuDNN 的 extern fallback**）。LIF 由于是 `torch.library.custom_op`（黑盒），dynamo 不
trace 进去，按 op 调用——还是路径 A 用的那个 SJ 手写 Triton kernel。

### 2.2 代码组成

两个必备的配置函数都在 vgg16_test.py 里：

**`configure_full_triton_compilation()`**（[vgg16_test.py:158-184](vgg16_test.py#L158-L184)）：

```python
def configure_full_triton_compilation():
    import torch._dynamo
    import torch._inductor.config as inductor_cfg

    torch._dynamo.config.recompile_limit = 256        # 不让 SJ 多步层把 frame cache 撑爆
    torch._dynamo.config.cache_size_limit = 256
    inductor_cfg.max_autotune = True                   # 启用 conv/gemm 自动调优
    inductor_cfg.max_autotune_gemm_backends = "TRITON" # GEMM 走 Triton，不走 cuBLAS
    inductor_cfg.max_autotune_conv_backends = "TRITON" # CONV 走 Triton，不走 cuDNN
    inductor_cfg.force_disable_caches = True           # 让自定义 SNN Pass 每次都重新作用
```

**`patch_spikingjelly_for_full_graph()`**（[vgg16_test.py:187-206](vgg16_test.py#L187-L206)）：

```python
def patch_spikingjelly_for_full_graph():
    from spikingjelly.activation_based import functional as sjf

    def _seq_to_ann_forward_single_graph(x_seq, stateless_module):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)
        y = stateless_module(y)        # 多步层均传入单个 Callable
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)

    sjf.seq_to_ann_forward = _seq_to_ann_forward_single_graph
```

为什么要 patch：SJ 原版 `seq_to_ann_forward`（[functional/forward.py:217-271](../../spikingjelly/spikingjelly/activation_based/functional/forward.py#L217-L271)）里
有一条 `isinstance(stateless_module, (list, tuple, nn.Sequential))` 分支判定 ——
本网络里只有 `layer.BatchNorm2d` 会调它（[layer/bn.py:88](../../spikingjelly/spikingjelly/activation_based/layer/bn.py#L88)，
传入的是绑定方法 `self.super_forward`），dynamo 静态识别不了那个 isinstance，在每一个 BN 处图中断。
打成上面那个不含 isinstance 的等价实现，整网才能编进单一计算图（`graph_break=0`）。

### 2.3 启动

```bash
cd examples/vgg16_snn

# 选项 1: 单次推理 + 黄金输出比对（含编译耗时统计）
python vgg16_test.py

# 选项 2: 在 ImageNet val 上跑 10000 张，统计单张延迟 + top-1
COMPILE=1 python benchmark_inference.py 10000 50

# 选项 3: 与 NIR 版做 100-iter 对比（A 单边）
MODE=A BATCH=50 python benchmark_compare.py
```

首次启动会触发 `max_autotune` 对每个 conv 现场试 17–18 个 cfg，**冷启动 ~50–120 s**。
之后命中 Triton 自身的磁盘缓存 `~/.triton/cache`，编译时间会显著缩短。
要每次启动都强制重 codegen（开发 SNN Pass 时）加 `TRITON_ALWAYS_COMPILE=1`：

```bash
TRITON_ALWAYS_COMPILE=1 COMPILE=1 python benchmark_inference.py 10000 50
```

### 2.4 怎么验证"真的在走全 Triton 路径"

参考 [SpikingJelly-Triton-Patch.md §7.5](SpikingJelly-Triton-Patch.md)，三道关：
- 终端打印里有 `dynamo 图中断数: 0`（必要不充分）
- `TORCH_LOGS="output_code" python vgg16_test.py 2>&1 | grep -cE "^extern_kernels\.(convolution|addmm|mm|bmm)\("` 应为 `0`（充分）
- 日志里出现 `Name=_multistep_lif_forward_kernel` + `SingleProcess AUTOTUNE benchmarking ... for 4 choices` —— LIF Triton kernel 也在跑

---

## 3. 实现路径 C：NIR roundtrip + Eager

### 3.1 关键差异（相对 A）

- 13 个 BN 在 SJ 端 fold 进 Conv（数学等价，eval 下逐位一致），fold 后无 BN 节点；
- 5 个 MaxPool 替换为 AvgPool（**NIR 协议没有 MaxPool 原语**，被迫换；数学不再与 A 等价）；
- 经 `export_to_nir` 写出 NIR 图 → `import_from_nir(device='cuda', step_mode='m')` 读回
  一个 `fx.GraphModule`；
- 推理走 eager（无 `torch.compile`）；stateless 层落到 PyTorch 默认后端 = cuDNN/cuBLAS/ATen
  native；LIF 还是路径 A、B 用的那个 SJ 手写 Triton kernel。

### 3.2 代码组成

构造 SJ 端的 BN+AvgPool 单步网络（[vgg16_via_nir.py:46-79](vgg16_via_nir.py#L46-L79)）：

```python
def build_vgg16_snn(num_classes=NUM_CLASSES):
    feats, in_ch = [], 3
    for v in VGG16_CFG:
        if v == "P":
            feats.append(nn.AvgPool2d(kernel_size=2, stride=2))           # 不用 layer.AvgPool2d
        else:
            feats.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))   # 不用 layer.Conv2d
            feats.append(nn.BatchNorm2d(v))
            feats.append(neuron.LIFNode(step_mode="s"))                   # 必须 SJ neuron
            in_ch = v
    features = nn.Sequential(*feats)
    classifier = nn.Sequential(
        nn.Flatten(), nn.Linear(512 * 7 * 7, 4096),
        neuron.LIFNode(step_mode="s"),
        nn.Linear(4096, 4096),
        neuron.LIFNode(step_mode="s"),
        nn.Linear(4096, num_classes),
    )
    return nn.Sequential(features, classifier)
```

**为什么所有无状态层都用原生 `nn.*` 而非 `layer.*`**：fold-BN 用的 `_EvalFusionTracer`
（在 `fuse_conv_bn_eval_modules` 内部）会穿透 SJ 的 `layer.AvgPool2d.forward`，把它内联成
`torch._C._nn.avg_pool2d` 这样的 `call_function` 节点；而 NIR 端 tracer 只接受
`call_module` + `operator.add`，遇到内联函数会抛 `ValueError: The only supported function is addition`。
原生 `nn.*` 在 fx 默认是 leaf module，不会被内联。

fold + 导出 + 导入（[vgg16_via_nir.py:127-148](vgg16_via_nir.py#L127-L148)）：

```python
model = build_vgg16_snn().eval()
folded = fuse_conv_bn_eval_modules(model)                                 # BN 吸收进 Conv
example_input = torch.rand(1, 3, 224, 224)
graph = nir_exchange.export_to_nir(folded, example_input=example_input, dt=1e-4)
gm = nir_exchange.import_from_nir(graph, dt=1e-4, device='cuda', step_mode='m')
```

`gm` 是 `torch.fx.GraphModule`，其内部子模块由 `nirtorch.nir_to_torch` 按 NIR 节点类型重建：
NIR 的 `Conv2d` → `layer.Conv2d`，`AvgPool2d` → `layer.AvgPool2d`，`Linear`/`Affine` → `layer.Linear`，
`LIF` → `neuron.LIFNode`（详见 [nir_exchange/from_nir.py:32-146](../../spikingjelly/spikingjelly/activation_based/nir_exchange/from_nir.py#L32-L146)）。
也就是说 **NIR 端是 `layer.*` 多步包装**，与路径 A 的多步包装是同一份代码。

### 3.3 启动

```bash
cd examples/vgg16_snn

# 单次推理（10 iter 均值）
python vgg16_via_nir.py

# 与 A 做 100-iter 对比（C 单边；大 BATCH 必须 MODE=B 与 A 分进程跑，否则 OOM）
MODE=B BATCH=40 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python benchmark_compare.py
```

注意：NIR 版（eager fx.GraphModule，无 Inductor 的 buffer reuse）显存峰值远高于路径 B，
**BATCH=40 是 RTX 5070 Ti (16 GiB) 上的实测上限**。BATCH=48 起会在 SJ multistep_lif kernel
的 `@triton.autotune` pre-hook clone 处 OOM。

---

## 4. 在三种路径上跑 10000 次推理

### 4.1 路径 A / B：用 `benchmark_inference.py`

`benchmark_inference.py` 加载已训练 SNN 权重 + 在 ImageNet val 上跑 N 个样本，
适合"10000 次真实推理"场景。同一脚本通过环境变量 `COMPILE` 切 A/B：

```bash
cd examples/vgg16_snn

# 路径 A: eager fp32
python benchmark_inference.py 10000 50

# 路径 B: 全 Triton
COMPILE=1 python benchmark_inference.py 10000 50

# 第一个参数: N_SAMPLES   - 跑多少张样本（默认 10000）
# 第二个参数: BATCH       - 每个 forward 的 batch size（默认 50）
```

脚本会同时报告：
- 编译 + 首次前向（一次性开销，B 路径 ~50–120s）
- 纯 GPU 前向耗时（计入计时，CUDA 同步前后 perf_counter）
- 总墙钟耗时（含 JPEG 解码 + dataloader）
- 单张推理 ms = 纯 GPU 耗时 / 总样本数
- top-1（健全性检查，应 ~49% 与 eager 一致）

参考结果（[SpikingJelly-Triton-Patch.md §7](SpikingJelly-Triton-Patch.md)）：

| BATCH | 路径 A (eager) | 路径 B (全 Triton) |
|---|---|---|
| 1 | (没测) | 12.40 ms / 张 |
| 50 | 7.41 ms / 张 | 9.41 ms / 张 |
| 56 | (没测) | 9.39 ms / 张 (cap) |

### 4.2 路径 C：用 `benchmark_compare.py MODE=B`

NIR 版没有"跑 ImageNet val"的入口（权重是随机的），但可用 `benchmark_compare.py` 跑
100 次 forward 求均值。100 次 × BATCH = 总样本数：

```bash
cd examples/vgg16_snn

# 100 iter × BATCH=40 = 4000 张样本（NIR 上限）
MODE=B BATCH=40 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python benchmark_compare.py

# 想攒到 10000 张，再加大 MEASURE。下面把 MEASURE 改成 250，得 250×40=10000 张：
# 改的是脚本顶部 MEASURE 常量，或临时把它读 env var（见 §5.2）
```

`benchmark_compare.py` 也能跑路径 B（`MODE=A`），与 §4.1 的 `benchmark_inference.py COMPILE=1` 区别：
- `benchmark_compare.py`：随机输入、随机权重、纯 GPU 计时（不含 dataloader）、100 次 forward 平均；
- `benchmark_inference.py`：真实 ImageNet val、训练权重、含 dataloader 的墙钟 + 纯 GPU 双指标。

### 4.3 跑哪个？

- 要**真实推理延迟 + 验证 top-1**：用 `benchmark_inference.py`（路径 A / B）。
- 要**对照三路径同一输入下的纯 GPU 性能**：用 `benchmark_compare.py`。
- 要**重测黄金输出的逐位可复现**：用 `vgg16_test.py`（默认路径 B，单次）。

---

## 5. 修改 BATCH 进行多轮测量

### 5.1 `benchmark_inference.py`（路径 A / B）

BATCH 直接由命令行第 2 个参数控制（[benchmark_inference.py:30](benchmark_inference.py#L30)）：

```bash
for B in 1 8 16 32 50 56; do
    echo "==== BATCH=$B ===="
    COMPILE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python benchmark_inference.py 10000 $B 2>&1 | tail -8
done
```

注意每次更换 BATCH，`max_autotune` 会重新对该 BATCH 下的 conv 形状跑一遍自动调优（首次前向慢 ~50–120 s）。
后续如果再开同 BATCH，Triton 缓存命中会跳过。

### 5.2 `benchmark_compare.py`（三模式）

BATCH 由环境变量控制（[benchmark_compare.py:62](benchmark_compare.py#L62)）：

```bash
for B in 1 16 32 40; do
    echo "==== BATCH=$B  路径 A ===="
    MODE=A BATCH=$B python benchmark_compare.py 2>&1 | tail -8
    echo "==== BATCH=$B  路径 C ===="
    MODE=B BATCH=$B PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python benchmark_compare.py 2>&1 | tail -8
done
```

辅助选项：
- `MEASURE=N`：要让脚本支持环境变量改 `MEASURE`，目前需要小改动 —— [benchmark_compare.py:58](benchmark_compare.py#L58)
  里 `MEASURE = 100` 是常量，改成
  `MEASURE = int(os.environ.get("MEASURE", 100))` 即可命令行控制总样本量。
- `MODE=both`：A 和 C 在同一进程里前后跑（小 BATCH 才行；BATCH ≥ 16 时显存装不下两个模型 + autotune buffer，
  会在 [benchmark_compare.py:283](benchmark_compare.py#L283) 之后那个 `gc.collect() + empty_cache()` 都
  救不回来，必须分两次进程）。

### 5.3 BATCH 上限速查（RTX 5070 Ti，16 GiB）

| 路径 | 实测 BATCH 上限 | 卡在哪一步 |
|---|---|---|
| A (eager) | (没测，应远超 50) | — |
| B (torch.compile + 全 Triton) | **56** | LIF kernel 输出 buffer alloc |
| C (NIR + eager) | **40** | LIF kernel `@triton.autotune` pre-hook `restore_copies.clone()` |

C 的上限更低，是因为 eager fx.GraphModule **没有 Inductor 的 buffer reuse**，每层激活独立分配。

---

## 6. NIR 实现的 cuDNN 具体调用路径

这是路径 C 一个 forward 调用从 Python 到 cuDNN 库的完整链路。以 13 个 Conv2d 中的第一个为例
（在 BATCH=40, T=4 时，输入 `[4, 40, 3, 224, 224]`）。

### 6.1 第 1 层：`fx.GraphModule.forward` 派发

`import_from_nir` 返回的 `gm` 是 `torch.fx.GraphModule`，其 `forward` 是 nirtorch 自动生成的
Python 源码（[nirtorch/from_nir.py](../../nirtorch/nirtorch/from_nir.py) / `nir_interpreter.py`），大致：

```python
def forward(self, input, state = {...}):
    input_1 = input
    _0 = self._0(input_1);  input_1 = None   # NIR Conv2d#0
    _1 = self._1(_0);  _0 = None             # NIR LIF#0
    _2 = self._2(_1);  _1 = None             # NIR AvgPool2d#0
    ...
    return _37, state
```

每个 `self._N` 是按 NIR 节点类型映射出的 SJ 模块。`self._0` 是 `layer.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, step_mode='m')`。

### 6.2 第 2 层：`layer.Conv2d.forward`

[stateless_wrapper.py:176-190](../../spikingjelly/spikingjelly/activation_based/layer/stateless_wrapper.py#L176-L190)：

```python
def forward(self, x: Tensor):
    if self.step_mode == "s":
        x = super().forward(x)
    elif self.step_mode == "m":
        if x.dim() != 5:
            raise ValueError(...)
        y_shape = [x.shape[0], x.shape[1]]
        y = super().forward(x.flatten(0, 1))      # [4·40, 3, 224, 224]
        y_shape.extend(y.shape[1:])
        x = y.view(y_shape)                       # [4, 40, 64, 224, 224]
    return x
```

`super().forward(...)` 调到 `nn.Conv2d.forward`，传入 reshape 后的 `[T·B, 3, 224, 224]`。

### 6.3 第 3 层：`nn.Conv2d.forward → _conv_forward`

PyTorch 源码（[pytorch/torch/nn/modules/conv.py](../../pytorch/torch/nn/modules/conv.py)）：

```python
class Conv2d(_ConvNd):
    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, ..., mode=self.padding_mode),
                            weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)
```

继续转到 `torch.nn.functional.conv2d`。

### 6.4 第 4 层：`F.conv2d → torch._C._VariableFunctions.conv2d`

`torch/nn/functional.py` 里 `conv2d` 是 builtin —— 直接是 `torch._C._VariableFunctions.conv2d`，进入 ATen。

### 6.5 第 5 层：ATen `at::native::convolution → _convolution → select_conv_backend`

[pytorch/aten/src/ATen/native/Convolution.cpp:1190-1230](../../pytorch/aten/src/ATen/native/Convolution.cpp#L1190-L1230)
里的 `_select_conv_backend` 在多组条件后给出后端选择。对于 fp32 CUDA + 3x3 conv +
`torch.backends.cudnn.enabled=True`（默认）：

```cpp
static ConvBackend _select_conv_backend(...) {
    ...
    if (input.is_cuda() && cudnn_available && needs_cudnn_format) {
        return ConvBackend::Cudnn;
    }
    ...
}
```

然后在 [Convolution.cpp:1507-1530](../../pytorch/aten/src/ATen/native/Convolution.cpp#L1507-L1530)：

```cpp
ConvBackend backend = _select_conv_backend(input, weight, bias, ...);
switch (backend) {
    ...
    case ConvBackend::Cudnn:
        output = at::cudnn_convolution(
            input.contiguous(memory_format), weight, ...);
        break;
    ...
}
```

### 6.6 第 6 层：`at::cudnn_convolution → raw_cudnn_convolution_forward_out → cuDNN library`

[pytorch/aten/src/ATen/native/cudnn/Conv_v8.cpp:1214-1245](../../pytorch/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1214-L1245)：

```cpp
void raw_cudnn_convolution_forward_out(...) {
    ...
    raw_cudnn_convolution_forward_out_v7(input, output, weight, ...);   // 走 cuDNN v7 API
    // 或新版本走 v8 frontend ExecutionPlan
}
```

最终调进 NVIDIA `libcudnn.so.9` 里的 `cudnnConvolutionForward` / `cudnnBackendExecute`。

### 6.7 整链总结（路径 C 一次 Conv2d 前向）

```
fx.GraphModule.forward       (nirtorch 生成的 python source)
  └─ layer.Conv2d.forward    (SJ stateless_wrapper.py:176; flatten [T,B] → [T·B])
      └─ nn.Conv2d.forward   (torch/nn/modules/conv.py; _conv_forward)
          └─ F.conv2d        (torch/nn/functional.py 转 builtin)
              └─ torch._C._VariableFunctions.conv2d (进 ATen)
                  └─ at::native::convolution (Convolution.cpp:1190 selects backend)
                      └─ ConvBackend::Cudnn 命中
                          └─ at::cudnn_convolution
                              └─ raw_cudnn_convolution_forward_out (Conv_v8.cpp:1214)
                                  └─ cudnnBackendExecute / cudnnConvolutionForward
                                      └─ libcudnn.so.9 → GPU
```

### 6.8 路径 C 中其他 stateless 层走的库

| SJ 子模块 | super().forward → 最终库 |
|---|---|
| `layer.Conv2d` | `nn.Conv2d` → ATen `convolution` → **cuDNN** (`cudnnConvolutionForward`) |
| `layer.AvgPool2d` | `nn.AvgPool2d` → ATen `avg_pool2d_out_cuda` → **ATen native CUDA kernel**（**非 cuDNN**）|
| `layer.Linear` | `nn.Linear` → ATen `linear` → `addmm` → **cuBLAS** (`cublasSgemm` / Lt) |
| `layer.Flatten` / `nn.Flatten` | `tensor.view` / `reshape` → 无 GPU kernel，只改 stride/shape |
| `neuron.LIFNode` | `triton_kernel.multistep_lif` → **SJ 手写 Triton kernel**（与路径 A/B 同）|

### 6.9 如何验证 cuDNN 确实被调到

```bash
# 推理 + 用 nsys 抓 cuDNN/cuBLAS 调用
nsys profile -t cuda,cudnn,cublas --output=/tmp/vgg16_c_nir \
    python examples/vgg16_snn/vgg16_via_nir.py
nsys stats /tmp/vgg16_c_nir.nsys-rep | head -40

# 或直接看 cuDNN log（NVIDIA 提供的环境变量）
CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=stderr \
    python examples/vgg16_snn/vgg16_via_nir.py 2>&1 | grep -c "cudnnConvolutionForward"
# 期望: 远大于 0（13 个 Conv × 100 iter × T·B forward call 的数量）
```

---

## 7. 三种路径速查对比表

| 维度 | A: Eager | B: torch.compile + 全 Triton | C: NIR + Eager |
|---|---|---|---|
| 入口脚本 | `benchmark_inference.py` | `vgg16_test.py` / `benchmark_inference.py COMPILE=1` | `vgg16_via_nir.py` / `benchmark_compare.py MODE=B` |
| 网络结构 | 13 Conv + 13 BN + 5 MaxPool + 3 FC + 15 LIF | 同 A | 13 Conv (fold-BN) + 5 AvgPool + 3 FC + 15 LIF |
| Conv 后端 | cuDNN (ATen → libcudnn) | Inductor Triton (max_autotune) | cuDNN (同 A) |
| GEMM 后端 | cuBLAS (ATen → addmm) | Inductor Triton | cuBLAS (同 A) |
| Pool 后端 | ATen native CUDA | Inductor Triton | ATen native CUDA |
| LIF 后端 | SJ 手写 Triton kernel | SJ 手写 Triton kernel | SJ 手写 Triton kernel |
| 编译开销 | 0 | 50–120 s 首次 | 0 |
| BATCH 上限 (16 GiB) | 远大于 50 | 56 | 40 |
| 数学与黄金输出 | 等价 | 等价 | **不等价**（AvgPool 替 MaxPool）|
| BATCH=50 单张延迟 | 7.41 ms | 9.41 ms | (BATCH=40) 4.90 ms |

### 测量延迟时的固定 boilerplate

不论哪条路径，eval 模式下前向之前都要：

```python
functional.reset_net(model)               # 复位 LIF 膜电位
with torch.no_grad():
    out = model(x)                        # x: [T, B, C, H, W]
torch.cuda.synchronize()                  # CUDA 异步执行，计时前必须同步
```

其中 `reset_net` 是被 `vgg16_test.py:229`、`benchmark_inference.py:67`、`benchmark_compare.py` 三处都
调用的关键步骤——LIF 的膜电位状态 `self.v` 在前一次前向结束后还残留，不复位会让第二次推理拿
到错误的初始电位。
