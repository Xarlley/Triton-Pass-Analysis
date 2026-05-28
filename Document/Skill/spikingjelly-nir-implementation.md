# SpikingJelly 中 NIR 的实现机制与到 GPU 的完整 IR 下降链

> 本文回答四件事：
> 1. 在 SpikingJelly 里**怎么用** NIR 编程；
> 2. 用户写完 [`examples/vgg16_snn/vgg16_via_nir.py`](../../examples/vgg16_snn/vgg16_via_nir.py)
>    这样的高级代码后，它**怎么映射**成一个 `nir.NIRGraph` —— 源码级穿透 SJ 的
>    `nir_exchange` 与 `nirtorch` 两层；
> 3. 本示例的网络**真实**生成的 NIRGraph 长什么样（捕获自一次真机运行）；
> 4. 这个 NIRGraph 之后**怎么下沉**到 PyTorch eager → ATen → cuDNN/cuBLAS（无状态层）
>    与 Triton TTIR → TTGIR → LLIR → PTX（LIF 层），全部基于一次真机运行的中间产物捕获。
>
> 配套的真实捕获产物在 [`Document/IR-Trace/nir_lif_kernel/`](../IR-Trace/nir_lif_kernel/)。
> 验证环境：torch 2.11.0+cu130、spikingjelly 0.0.0.0.15 (Xarlley fork,
> `triton-fork-compat`)、triton 3.7.0+gitef02d646、nir 1.0.8、nirtorch 2.7、
> RTX 5070 Ti (sm_120, Blackwell)。

---

## 1. SpikingJelly 的 NIR 编程模型

### 1.1 NIR 是什么 / 不是什么

**[NIR (Neuromorphic Intermediate Representation)](https://neuroir.org/)** 是一种跨框架/跨硬件的
**模型描述协议**，类比 ONNX 之于传统神经网络。它由两部分组成：

- 一组**原语节点类型**（`nir.Conv2d`、`nir.AvgPool2d`、`nir.Linear`、`nir.Affine`、`nir.Flatten`、
  `nir.IF`、`nir.LIF`、`nir.CubaLIF`、`nir.Delay`、`nir.Input`、`nir.Output` 等，
  完整列表在 [`nir/nir/ir/__init__.py`](../../nir/nir/ir/__init__.py)）；
- 一个**有向图**容器 `nir.NIRGraph(nodes, edges)`（[`nir/nir/ir/graph.py:21`](../../nir/nir/ir/graph.py#L21)）。

NIR **不是**：
- 不是一个执行引擎（它没有 forward()）；
- 不是一个编译器（它没有 lowering 规则）；
- 不是一个完备张量计算 IR（无 `BatchNorm` / `MaxPool` / `Dropout` / `LayerNorm` / 注意力 / Conv3d）。

NIR 的作用是**让两个不同神经形态框架/芯片能交换 SNN 模型描述**，把"模型表达"从"模型执行"中
解耦出来。

### 1.2 SJ 的 NIR API

SpikingJelly 在 [`spikingjelly/activation_based/nir_exchange/`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/)
提供双向两个函数：

```python
from spikingjelly.activation_based import nir_exchange

# 出口：SJ Module → NIRGraph
graph = nir_exchange.export_to_nir(net, example_input, save_path=None, dt=1e-4)

# 入口：NIRGraph → SJ Module（封成 fx.GraphModule）
gm = nir_exchange.import_from_nir(graph, dt=1e-4, device="cuda", step_mode="m")
```

两者都是薄包装层 —— 真正干活的是 `nirtorch.torch_to_nir` 与 `nirtorch.nir_to_torch`。
SJ 这一层只贡献：
- 一份 `_ModuleMapper.map_dict`（[`to_nir.py:62-74`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/to_nir.py#L62-L74)）
  把每种 PyTorch / SJ 模块类映射成对应的 NIR 节点构造逻辑；
- 一份 `_NodeMapper.map_dict`（[`from_nir.py:32-40`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/from_nir.py#L32-L40)）
  反向把每种 NIR 节点类型映射成对应的 SJ `layer.*` / `neuron.*` 模块。

### 1.3 单步 vs 多步：NIR 协议本身不含 T/B

NIR 节点的 `input_type` / `output_type` 字段**只描述一个样本一个时间步**的形状（如
`[3, 224, 224]`），不含 T、B 维度（教程明确说明）。SJ 这样处理：

- 导出时：用户传入的 `example_input` 可以带任意 T/B，SJ 内部经 `ShapeProp` 推 shape 并剥掉
  T/B 维度后写进 NIR 节点（[`to_nir.py:34-58`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/to_nir.py#L34-L58)）。
- 导入时：`import_from_nir(step_mode='m')` 返回的 `fx.GraphModule` **内部**用 SJ
  `layer.*` 多步包装，因此能直接吃 `[T, B, C, H, W]` 输入。

### 1.4 协议级强约束（不是 SJ 自家限制）

| 算子 | 在 NIR 里有吗 | 后果 |
|---|---|---|
| `BatchNorm2d` | **无** | 导出前必须 fold（eval 模式下数学等价折进前面 Conv 的 weight/bias），否则 `_ModuleMapper.map_dict` 查不到、SJ 端报 KeyError。 |
| `MaxPool2d` | **无**（只有 `AvgPool2d`、`SumPool2d`）| 模型必须改用 AvgPool；强行映射会破坏黄金输出。 |
| 软复位 (`v_reset=None`) | LIF 节点不区分软/硬复位 | SJ 强制写 `v_reset_=0.0`，并在 docstring 警告（[`to_nir.py:128-153`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/to_nir.py#L128-L153)）。 |
| `ParametricLIFNode` | 无独立节点 | 在 fix-tau 后退化为普通 LIF，导出为 `nir.LIF`。 |
| 多重输入 / 残差 add | NIR 支持有限的 `operator.add` | 其他 call_function 一律 `ValueError: The only supported function is addition`（[`torch_tracer.py:218`](../../nirtorch/nirtorch/torch_tracer.py#L218)）。 |

---

## 2. 高级代码 → NIRGraph 的源码级穿透

下面以 [`vgg16_via_nir.py`](../../examples/vgg16_snn/vgg16_via_nir.py) 的一行调用为锚点：

```python
graph = nir_exchange.export_to_nir(folded, example_input=example_input, dt=1e-4)
```

走进去要经过 SJ 与 nirtorch 两层。逐层拆。

### 2.1 SJ 这层：薄桥 + 映射字典

[`spikingjelly/activation_based/nir_exchange/to_nir.py:293-299`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/to_nir.py#L293-L299) 全部代码：

```python
def export_to_nir(net, example_input, save_path=None, dt=1e-4):
    mapper = _ModuleMapper(net, example_input, dt=dt)
    graph = nirtorch.torch_to_nir(net, mapper.map_dict, type_check=True)
    if save_path is not None:
        nir.write(save_path, graph)
    return graph
```

`_ModuleMapper`（[同文件 22 行起](../../spikingjelly/spikingjelly/activation_based/nir_exchange/to_nir.py#L22)）做两件事：

1. **预先扫一遍**用户网络，记下每个子模块的 input/output shape（用 PyTorch 自带的
   `torch.fx.passes.shape_prop.ShapeProp`）。
2. 在 `map_dict` 里给每种支持的模块类提供一个**实例 → NIR 节点**的 lambda：

```python
@property
def map_dict(self):
    return {
        nn.Linear:        self.map_linear,
        layer.Linear:     self.map_linear,
        nn.Conv2d:        self.map_conv2d,
        layer.Conv2d:     self.map_conv2d,
        nn.AvgPool2d:     self.map_avgpool2d,
        layer.AvgPool2d:  self.map_avgpool2d,
        nn.Flatten:       self.map_flatten,
        layer.Flatten:    self.map_flatten,
        neuron.IFNode:    self.map_if,
        neuron.LIFNode:   self.map_lif,
        neuron.ParametricLIFNode: self.map_plif,
    }
```

比如 `map_conv2d`（[L85-101](../../spikingjelly/spikingjelly/activation_based/nir_exchange/to_nir.py#L85-L101)）：

```python
def map_conv2d(self, module):
    if module.bias is None: bias = np.zeros((module.weight.shape[0]))
    else: bias = _to_numpy(module.bias)
    H, W = self.module_io_shape[module]["input_shape"][-2:]
    return nir.Conv2d(
        input_shape=(H, W),
        weight=_to_numpy(module.weight),
        stride=module.stride, padding=module.padding,
        dilation=module.dilation, groups=module.groups,
        bias=bias,
    )
```

每条 lambda 都把 PyTorch 模块的**权重与超参一次性拷贝**进 NIR 节点的 numpy 字段。NIR 节点
**自身不持有 PyTorch 张量**，全部存 numpy —— 这是 NIR 「跨硬件可移植」的基础。

注意：SJ 这层**完全不知道**用户网络拓扑、不做 fx 追踪、不构图。它只是给 nirtorch 提交了一份
「我支持这些类，每类怎么变 NIR」的小词典。真正干图遍历的活都丢给了 nirtorch。

### 2.2 nirtorch 这层：fx tracer + 节点遍历

[`nirtorch.torch_to_nir`](../../nirtorch/nirtorch/torch_tracer.py#L74) 的实现要点（精简）：

```python
def torch_to_nir(module, module_map, default_dict=DEFAULT_MAP, type_check=True,
                 stateful_modules=None, concrete_args=None):
    module_map = module_map | default_dict
    # 1) 边界情形：单个 leaf 模块直接映射
    if module.__class__ in module_map:
        return module_map[module.__class__](module)
    # 2) fx 追踪整个网络
    tracer = NIRTorchTracer(module_map.keys())
    traced = tracer.trace(module, concrete_args=concrete_args)
    graph_module = torch.fx.GraphModule(tracer.root, traced)
    # 3) 遍历 fx 节点，按 op 分发
    nodes, edges = {}, []
    for node in traced.nodes:
        if node.op == "placeholder": ...        # → nir.Input
        elif node.op == "output":    ...        # → nir.Output
        elif node.op == "call_module":          # ★ 主流量
            tm = graph_module.get_submodule(node.target)
            if tm.__class__ in module_map:
                nir_node = module_map[tm.__class__](tm)  # 调 SJ 的 map_*
                nodes[str(node.name)] = nir_node
        elif node.op == "call_function":        # 仅允许 operator.add 与 getitem
            ...
    # 4) 二次遍历建 edges（处理 bypass / ignore 节点）
    for node in traced.nodes:
        for in_node in node.all_input_nodes:
            edges.append((in_node.name, node.name))
    # 5) Input 节点的 input_type 由后继节点的 input_type 推回
    ...
    return nir.NIRGraph(nodes=nodes, edges=edges, type_check=type_check)
```

关键设计点：

1. **`NIRTorchTracer` 是 `torch.fx.Tracer` 子类**（[L24](../../nirtorch/nirtorch/torch_tracer.py#L24)），关键是它把
   `module_map.keys()` 注册为 `leaf_modules` —— fx 追踪时不进入这些类的 forward，把它们
   作为不可分原子记录下来。这就是为什么 `neuron.LIFNode` 在 NIR 图里出现一次而不是
   被 trace 成「电压更新 / 阈值判断 / 复位」三段独立节点。

2. **节点分发是基于 `node.op + module.__class__`**，没有任何运行时类型检查 —— map_dict
   缺一类，trace 时立即抛
   `ValueError: Unknown module encountered: <class '...'>`（[L239-242](../../nirtorch/nirtorch/torch_tracer.py#L239-L242)）。

3. **`call_function` 几乎全禁**（[L218-220](../../nirtorch/nirtorch/torch_tracer.py#L218-L220)）：
   只允许 `operator.add`（残差连接）和 `operator.getitem`（用于 stateful module 解 tuple）。
   遇到任何 `torch._C._nn.avg_pool2d` 这样的内联函数会拒绝。这就是
   [`vgg16_via_nir.py`](../../examples/vgg16_snn/vgg16_via_nir.py) 必须用原生 `nn.AvgPool2d`
   而非 `layer.AvgPool2d` 的原因 —— SJ wrapper 在 fold-BN 的 tracer 下会被内联成
   `call_function`，而 NIRTorchTracer 此处只接 `call_module`。

4. **edges 列表是按 fx 节点的 input/output 关系自然导出的**，所以 NIR 图的拓扑天然是
   "data flow graph"，跟 PyTorch 计算图同构。

### 2.3 反向：`import_from_nir` 怎么把 NIRGraph 还原成 fx.GraphModule

镜像对称，[`from_nir.py:223-227`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/from_nir.py#L223-L227)：

```python
def import_from_nir(graph, dt=1e-4, device="cpu", dtype=torch.float32, step_mode="s"):
    mapper = _NodeMapper(dt=dt)
    gm = nirtorch.nir_to_torch(graph, mapper.map_dict, device=device, dtype=dtype)
    functional.set_step_mode(gm, step_mode)
    return gm
```

`_NodeMapper.map_dict` 给 NIR 节点类→SJ 模块的反向词典：

```python
{
    nir.Affine:    self.map_affine,      # → layer.Linear(bias=True)
    nir.Linear:    self.map_linear,      # → layer.Linear(bias=False)
    nir.Conv2d:    self.map_conv2d,      # → layer.Conv2d
    nir.AvgPool2d: self.map_avgpool2d,   # → layer.AvgPool2d
    nir.Flatten:   self.map_flatten,     # → layer.Flatten
    nir.IF:        self.map_if,          # → neuron.IFNode
    nir.LIF:       self.map_lif,         # → neuron.LIFNode
}
```

[`nirtorch.nir_to_torch`](../../nirtorch/nirtorch/nir_interpreter.py#L559) 用类似的图遍历把每个 NIR 节点构造成对应 SJ 模块，再用
[`nirtorch.graph_executor`](../../nirtorch/nirtorch/graph_executor.py) 把它们按 edges
拼成一个**自动生成 `forward()` 源码的 fx.GraphModule**。返回前 SJ 一次性 `set_step_mode(gm, 'm')`
把所有 `step_mode` 属性切到多步。

第 4 节会看到这个**真实生成的 forward Python 源码**。

---

## 3. 本示例真实生成的 NIRGraph（捕获自真机运行）

**捕获方法**：写一个 [`/tmp/capture_nir.py`](../IR-Trace/nir_lif_kernel/) 短脚本，复用
`vgg16_via_nir.py` 同款网络构造逻辑，调一次 `export_to_nir(folded, example_input, save_path=...)`
后 dump 整个 `NIRGraph.nodes` 与 `.edges`。完整产物在
[`Document/IR-Trace/nir_lif_kernel/`](../IR-Trace/nir_lif_kernel/)：

| 文件 | 内容 |
|---|---|
| `vgg16_snn.nir.h5` | NIR 协议规定的 HDF5 序列化（用 `nir.write` 写出）。可由任意支持 NIR 的框架直接 `nir.read` 加载。|
| `vgg16_snn.nir.repr.txt` | `NIRGraph` 节点 + 边的可读 dump（39 节点 / 38 边）。|
| `vgg16_snn.fx_graph_module.py` | nirtorch 反向重建出来的 `fx.GraphModule.forward` 的 Python 源码。|

### 3.1 39 个节点

按 NIR 节点类型归并：

| NIR 节点类型 | 数量 | 角色 |
|---|---:|---|
| `nir.Input` | 1 | 入口（input_type 由下游推回 `[3, 224, 224]`）|
| `nir.Conv2d` | 13 | VGG16 主干 13 个 3×3 卷积（已折 BN）|
| `nir.LIF` | 15 | 13 个 conv-后 LIF + 2 个 fc-后 LIF |
| `nir.AvgPool2d` | 5 | 5 个 2×2 池化（替代原 MaxPool）|
| `nir.Flatten` | 1 | 分类器入口 `[512,7,7] → [25088]` |
| `nir.Affine` | 3 | 3 个 fc 层（`Linear(bias=True)` 在 NIR 里叫 `Affine`）|
| `nir.Output` | 1 | 出口 |
| **合计** | **39** | |

注意：NIR 端区分 `Linear (bias=False)` 与 `Affine (bias=True)`，SJ 这边都映射到 `layer.Linear` —— 见
[`from_nir.py:42-46`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/from_nir.py#L42-L46)。

### 3.2 节点列表（前 8 + 后 4 节选，按 fx trace 顺序）

```text
'input_1': Input({'input': [  3, 224, 224]} -> {'input': [  3, 224, 224]})
'_0_0':    Conv2d({'input': [  3, 224, 224]} -> {'output': [ 64, 224, 224]})
'_0_2':    LIF({'input': [ 64, 224, 224]} -> {'output': [ 64, 224, 224]})
'_0_3':    Conv2d({'input': [ 64, 224, 224]} -> {'output': [ 64, 224, 224]})
'_0_5':    LIF({'input': [ 64, 224, 224]} -> {'output': [ 64, 224, 224]})
'_0_6':    AvgPool2d({'input': [ 64, 224, 224]} -> {'output': [ 64, 112, 112]})
'_0_7':    Conv2d({'input': [ 64, 112, 112]} -> {'output': [128, 112, 112]})
'_0_9':    LIF({'input': [128, 112, 112]} -> {'output': [128, 112, 112]})
...
'_1_0':    Flatten({'input': [512,   7,   7]} -> {'output': [25088]})
'_1_1':    Affine({'input': [25088]}          -> {'output': [4096]})
'_1_2':    LIF({'input': [4096]}              -> {'output': [4096]})
'_1_3':    Affine({'input': [4096]}           -> {'output': [4096]})
'_1_4':    LIF({'input': [4096]}              -> {'output': [4096]})
'_1_5':    Affine({'input': [4096]}           -> {'output': [1000]})
'output':  Output({'input': [1000]}           -> {'output': [1000]})
```

完整 39 节点见 [`vgg16_snn.nir.repr.txt`](../IR-Trace/nir_lif_kernel/vgg16_snn.nir.repr.txt)。

观察：
- **节点名格式 `_<seq_idx>_<sub_idx>`**：来自原 `nn.Sequential` 嵌套结构的 fx target，
  `_0_*` 对应 `features` 段，`_1_*` 对应 `classifier` 段。fold-BN 后 BN 索引被消除，
  所以中间 idx 是 0, 2, 3, 5, 6, 7, 9 ... 不连续（被吸收的 BN 是 1, 4, 8, ...）。
- **每个 LIF 节点的 input_type == output_type**（NIR 协议规定 LIF 不改变形状）。
- **shape 字段不含 T 或 B**（NIR 协议约定）。

### 3.3 38 条边

所有边都是「前一节点 → 后一节点」的线性链（VGG16 是单分支拓扑），形如：

```
('input_1', '_0_0'), ('_0_0', '_0_2'), ('_0_2', '_0_3'), ...
('_1_4', '_1_5'),    ('_1_5', 'output')
```

完整 38 条边见 `vgg16_snn.nir.repr.txt`。

### 3.4 HDF5 序列化

`save_path` 不为 None 时 `nir.write` 把整张图写成 HDF5：

```
$ python -c "import h5py; h = h5py.File('vgg16_snn.nir.h5'); print(list(h.keys()))"
['edges', 'node_data']
```

`/node_data/<node_name>/` 下挂每个节点的属性（如 `type=Conv2d`, `weight`, `stride`, ...
等 numpy 数组）。这就是 NIR 当前唯一的"跨框架交换格式" —— 任何支持 NIR 的运行时（Lava /
Sinabs / Norse / Rockpool 等）都能用同一份 `.h5` 加载。

---

## 4. NIRGraph → fx.GraphModule（捕获自真机运行）

`import_from_nir(graph, device='cuda', step_mode='m')` 返回的 `gm` 是个标准 `torch.fx.GraphModule`，
其 `gm.code` 是 **nirtorch 在运行时自动生成的 Python 源码**。完整 dump 在
[`vgg16_snn.fx_graph_module.py`](../IR-Trace/nir_lif_kernel/vgg16_snn.fx_graph_module.py)。前 25 行：

```python
def forward(self, input, state : typing_Dict[str,typing_Any] = None):
    ones = torch.ones(1);  ones = None
    input_1 = input
    is_none = _operator_is_(state, None)
    initialized_state = nirtorch_nir_interpreter_ternary_operator(
        is_none,
        {'input_1': None, '_0_0': None, '_0_2': None, ..., 'output': None},
        state); is_none = state = None
    _0_0 = self._0_0(input_1);  input_1 = None      # NIR Conv2d#0 → layer.Conv2d
    _0_2 = self._0_2(_0_0);  _0_0 = None            # NIR LIF#0    → neuron.LIFNode
    _0_3 = self._0_3(_0_2);  _0_2 = None            # NIR Conv2d#1
    _0_5 = self._0_5(_0_3);  _0_3 = None            # NIR LIF#1
    _0_6 = self._0_6(_0_5);  _0_5 = None            # NIR AvgPool2d#0
    _0_7 = self._0_7(_0_6);  _0_6 = None
    ...
    _1_5 = self._1_5(_1_4);  _1_4 = None            # NIR Affine#2 (final fc)
    return _1_5, initialized_state
```

观察：

1. **`_X_Y = self._X_Y(<prev>); <prev> = None`** 是 nirtorch 的标准 dispatch 模式 ——
   显式 `= None` 让 CPython 立刻 decref 前一层 activation。**但**这不等于 Inductor
   级别的 buffer reuse —— 当前层输出与上一层输出 / 内部临时 buffer 仍可能短时间共存，
   这就是 §6 中 NIR 路径在 BATCH=48 起 OOM 的根本原因。

2. **`state` 参数与 `initialized_state` 返回值**：这是 NIR 协议层面的状态传递接口
   —— 用于无内部状态的纯函数式运行时（如某些神经形态芯片模拟器）。SJ 的 `neuron.LIFNode`
   把状态藏在 `self.v` 实例属性里，所以这个 `state` 字典对 SJ 路径**无效**，永远会被
   `nirtorch_nir_interpreter_ternary_operator(is_none, default, state)` 用默认值覆盖。

3. **`self._X_Y` 各是什么**：通过 `nir_to_torch` 的 `_NodeMapper.map_*` 反向重建出来的
   SJ 模块。`_0_0` 是 `layer.Conv2d(3, 64, ...)`、`_0_2` 是 `neuron.LIFNode(tau=...)` 等。
   返回前一次性 `set_step_mode(gm, 'm')` 把所有 step_mode 切到 'm'。

---

## 5. 无状态层下沉：Conv / Pool / Linear → cuDNN / cuBLAS

NIR 路径没有 `torch.compile`，整网走 PyTorch eager。每个 stateless 层的执行链都是 ATen
默认后端。**最长的链以 Conv2d 为例**，从 fx forward 一行 `_0_0 = self._0_0(input_1)` 起：

```
fx.GraphModule.forward                       (上面 §4 的 forward 源码)
  └─ layer.Conv2d.forward                    spikingjelly/layer/stateless_wrapper.py:176
      └─ 多步分支: super().forward(x.flatten(0,1))
          └─ nn.Conv2d.forward               torch/nn/modules/conv.py
              └─ _conv_forward
                  └─ F.conv2d                torch/nn/functional.py
                      └─ torch._C._VariableFunctions.conv2d (进入 ATen)
                          └─ at::native::convolution           pytorch/aten/src/ATen/native/Convolution.cpp:1190
                              └─ _select_conv_backend
                                  → ConvBackend::Cudnn (fp32 + CUDA + cudnn enabled + nchw)
                              └─ switch(backend) → at::cudnn_convolution                  Convolution.cpp:1524
                                  └─ raw_cudnn_convolution_forward_out                     pytorch/aten/src/ATen/native/cudnn/Conv_v8.cpp:1214
                                      └─ libcudnn.so.9: cudnnBackendExecute / cudnnConvolutionForward
                                          └─ GPU 上的 cuDNN 内置 kernel
```

其他无状态层同模式，落到的最终库不同：

| SJ 子模块 | super().forward 落到 | 最终 GPU 后端 |
|---|---|---|
| `layer.Conv2d` | `nn.Conv2d` → `F.conv2d` → ATen `convolution` | **cuDNN** (`cudnnConvolutionForward`) |
| `layer.AvgPool2d` | `nn.AvgPool2d` → `F.avg_pool2d` → ATen `avg_pool2d_out_cuda` | **ATen native CUDA pool**（**非 cuDNN**）|
| `layer.Linear` | `nn.Linear` → `F.linear` → ATen `linear` → `addmm` | **cuBLAS** (`cublasSgemm` / Lt) |
| `layer.Flatten` / `nn.Flatten` | `tensor.view` / `reshape` | **无 GPU kernel**（只改 stride/shape）|

### 真机验证 cuDNN 调用

```bash
# 方法 A：cuDNN 自家的环境变量，每次 cuDNN API 调用打一行日志
CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=stderr \
    python examples/vgg16_snn/vgg16_via_nir.py 2>&1 \
    | grep -c "cudnnConvolutionForward"
# 期望 >> 0（13 个 Conv × T·B 次 forward 调用 × N iter）

# 方法 B：nsys 抓全部 CUDA / cuDNN / cuBLAS 调用时间线
nsys profile -t cuda,cudnn,cublas --output=/tmp/vgg16_nir \
    python examples/vgg16_snn/vgg16_via_nir.py
nsys stats /tmp/vgg16_nir.nsys-rep | head -40
```

---

## 6. LIF 层下沉：NIR → Triton 五级 IR → PTX → SASS（捕获自真机运行）

整个 NIR 路径里**只有 LIF 节点会触发 Triton 编译**。下面这条链是真实运行触发的：

```
fx.GraphModule.forward
  └─ self._0_2(_0_0)
      └─ neuron.LIFNode.forward
          └─ multi_step_forward    spikingjelly/activation_based/neuron/lif.py:562
              └─ eval/CUDA/spiking 分支: try triton_kernel.multistep_lif(...)
                  └─ multistep_lif_forward (torch.library.custom_op)   spikingjelly/.../triton_kernel/neuron_kernel/lif.py:281
                      └─ wrap_triton(_multistep_lif_forward_kernel)[grid](...)
                          └─ @triton.jit kernel JIT compile pipeline:
                              .source  →  .ttir  →  .ttgir  →  .llir  →  .ptx  →  .cubin
                              (Triton 编译器自身的 Pass 序列在每一级之间运行)
                          └─ libcuda: cuLaunchKernel(cubin, grid, args)
                              └─ SM 上执行 SASS 指令
```

### 6.1 五级 IR 的真机捕获方法

Triton JIT 会把每次编译的中间产物**全部**落到磁盘缓存 `~/.triton/cache/<hash>/`。捕获步骤：

```bash
# 1) 清干净缓存，保证这一次跑的产物是新建的
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache

# 2) 跑一次 vgg16_via_nir.py（forward 会触发 15 个 LIF kernel 各自的 JIT 编译 + autotune）
python examples/vgg16_snn/vgg16_via_nir.py

# 3) 列出本次新增的 _multistep_lif_forward_kernel cache 目录
find ~/.triton/cache -name "_multistep_lif_forward_kernel.json" -printf "%h\n" | sort -u
# 每个目录对应一个 (LIF 输入形状, autotune cfg) 组合 —— 本仓库实测 24 个
```

每个 cache 目录里恰好对应五级 IR：

| 文件 | 含义 | 编译阶段 |
|---|---|---|
| `_multistep_lif_forward_kernel.source` | MLIR-level 类型化源码（带 loc 标注，从 SJ 的 Python `@triton.jit` 函数直接 lower 而来）| 入口 |
| `_multistep_lif_forward_kernel.ttir` | **Triton IR**（TTIR）：算子级 MLIR，不含 GPU layout | 1 |
| `_multistep_lif_forward_kernel.ttgir` | **Triton GPU IR**（TTGIR）：加入 `#ttg.blocked` layout 等 GPU-specific 属性 | 2 |
| `_multistep_lif_forward_kernel.llir` | **LLVM IR**（NVPTX backend 用的）| 3 |
| `_multistep_lif_forward_kernel.ptx` | **PTX 汇编**（由 LLVM NVPTX 后端生成）| 4 |
| `_multistep_lif_forward_kernel.cubin` | 最终 CUDA binary（含 SASS 指令）| 5 |
| `_multistep_lif_forward_kernel.json` | 元数据：num_warps、num_stages、target arch、扩展库等 | — |

本仓库捕获到的一份**完整代表样本**在
[`Document/IR-Trace/nir_lif_kernel/sample_kernel/`](../IR-Trace/nir_lif_kernel/sample_kernel/)
（取自 cache hash `6AJWNJ7JHAP3LKTTIMEE2ZMWO4SCXTSWQJT4XWSFMIKHEZIEW5KQ`，对应
`num_warps=4, num_stages=3, target sm_120, BLOCK_NCL=128`）。下面节选每一级的真实片段。

### 6.2 第 0 级：`.source` —— SJ Python `@triton.jit` 函数被 Triton compiler 取到的源码

```mlir
#loc = loc("/.../spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py":34:1)
#loc21 = loc("/.../spikingjelly/activation_based/triton_kernel/triton_utils.py":59:1)
module {
  tt.func public @_multistep_lif_forward_kernel(
      %x_seq_ptr:    !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %v_init_ptr:   !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %s_seq_ptr:    !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %h_seq_ptr:    !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %v_seq_ptr:    !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %tau: f32, %v_threshold: f32, %v_reset: f32)
    attributes {noinline = false} {
    %pid_ncl = tt.get_program_id x : i32 loc(#loc31)
    %ncl_offset = arith.constant 128 : i32 loc(#loc32)
    ...
```

注意 `loc("triton_utils.py":59:1)` —— 那正是我们 patch 过的
[`convert_and_store`](../../spikingjelly/spikingjelly/activation_based/triton_kernel/triton_utils.py#L59)。
SJ 自带的小工具函数会被 inline 进 kernel 的 MLIR 源码，所以这次 patch 修对了所有 LIF / IF / PLIF
kernel。

### 6.3 第 1 级：`.ttir` —— Triton IR（前 30 行）

```mlir
#loc = loc("/.../lif.py":34:1)
module {
  tt.func public @_multistep_lif_forward_kernel(
      %x_seq_ptr: !tt.ptr<f32> {tt.divisibility = 16}, ...) {
    %cst   = arith.constant dense<12288>      : tensor<1x128xi64>
    %cst_0 = arith.constant dense<8192>       : tensor<1x128xi64>
    %cst_1 = arith.constant dense<0.000000e+00>: tensor<1x128xf32>
    %cst_2 = arith.constant dense<4096>       : tensor<1x128xi64>
    %cst_3 = arith.constant dense<1.000000e+00>: tensor<1x128xf32>
    %cst_4 = arith.constant dense<0>          : tensor<1x128xi64>
    %c128_i32 = arith.constant 128 : i32
    %pid_ncl = tt.get_program_id x : i32                          # 拿 thread block ID
    %ncl_offset = arith.muli %pid_ncl, %c128_i32 : i32            # NCL 维度偏移 = pid * BLOCK_NCL
    %r_tau = arith.divf %cst_5, %tau : f32                         # 1/tau
    %v_init_ptrs = arith.extsi %ncl_offset : i32 to i64
    %v = tt.splat %v_init_ptr : !tt.ptr<f32> -> tensor<1x128x!tt.ptr<f32>>
    %v_6 = tt.make_range {end = 128, start = 0} : tensor<128xi32>
    %v_7 = arith.extsi %v_6 : tensor<128xi32> to tensor<128xi64>
    ...
```

特征：
- 算子级 MLIR，`tt.ptr<f32>` / `tensor<1x128xf32>` 等 Triton 自家类型；
- **没有** GPU 内存 layout 概念（warp、CTA、register tile）；
- 常量都已合并（`12288 = 4096*3` 即 T·NCL 字节偏移）。

完整文件 144 行：[`sample_kernel/_multistep_lif_forward_kernel.ttir`](../IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.ttir)。

### 6.4 第 2 级：`.ttgir` —— Triton GPU IR（前 15 行）

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32],
                         warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1, "ttg.num-warps" = 4,
                    ttg.target = "cuda:120", "ttg.threads-per-warp" = 32} {
  tt.func public @_multistep_lif_forward_kernel(...) {
    %cst   = arith.constant dense<12288> : tensor<1x128xi64, #blocked>
    %cst_0 = arith.constant dense<8192>  : tensor<1x128xi64, #blocked>
    ...
    %v_6  = tt.make_range {end = 128, start = 0}
            : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
```

差别集中在两点：
1. **新增 `#blocked` layout 属性**：把每个 tensor 在「线程内寄存器 / 线程内 / warp 间 / CTA 间」
   四级上的分布显式标出来。`threadsPerWarp = [1, 32]` 意味着 NCL=128 维由 32 个线程平铺，每线程
   持 4 个元素（=128/32），整体 4 warps × 32 threads = 128 lanes 处理 1×128 tile；
2. **`tensor<128x..., #ttg.slice<...>>`**：在数学上是个 1D tensor，但 layout 描述它是从某个 2D
   blocked layout 中沿某个 dim 切下来的，便于后续 lowering 时正确生成寄存器索引。

这一级也是本仓库自定义 `MyNoOpPass` 真正会作用的层 —— 见
[`Document/SNN_Pass_Execution_Analysis.md`](../SNN_Pass_Execution_Analysis.md)。

完整文件 145 行：[`sample_kernel/_multistep_lif_forward_kernel.ttgir`](../IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.ttgir)。

### 6.5 第 3 级：`.llir` —— LLVM IR

由 Triton 的 `TritonGPUToLLVM` 转换 Pass 生成，进入 LLVM NVPTX backend 的输入格式。每条原
TTGIR 算子展开成几十条 LLVM 指令（`getelementptr` / `extractvalue` / `ptrtoint` 等）。
完整文件见 [`sample_kernel/_multistep_lif_forward_kernel.llir`](../IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.llir)。

### 6.6 第 4 级：`.ptx` —— NVPTX 汇编（前 8 行）

```ptx
//
// Generated by LLVM NVPTX Back-End
//

.version 9.1
.target sm_120a       ; ← Blackwell RTX 5070 Ti
.address_size 64

.visible .entry _multistep_lif_forward_kernel(...)
```

491 行 PTX：[`sample_kernel/_multistep_lif_forward_kernel.ptx`](../IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.ptx)。
完整 SASS 指令在 `.cubin` 二进制里，要看可以：

```bash
cuobjdump --dump-sass \
    Document/IR-Trace/nir_lif_kernel/sample_kernel/_multistep_lif_forward_kernel.cubin
```

### 6.7 同一个 kernel 为什么有 24 个 cache 目录

实测一次跑出 24 个 `_multistep_lif_forward_kernel` cache 目录。原因：

- VGG16 的 15 个 LIF 实例分 **6 种唯一 (C, H, W) 形状**（conv2_1/2_2 同形状共用；conv3_1/2/3 同；
  conv4 系列；conv5 系列；fc1-LIF 与 fc2-LIF 同 4096-vec），加上输入 `[T·B, C·H·W]` 在 BATCH=1
  下 NCL 取值集合不同；
- `@triton.autotune(configs=[...])` 会对**每种 NCL 形状**试 **4 个 cfg**（num_warps × BLOCK_NCL），
  每个 cfg 都触发一次独立 JIT 编译并各自落盘；
- 6 种 LIF shape × 4 autotune cfg ≈ 24 cache 目录。

实际选用哪个由 autotune 的 benchmark 结果决定；其余 23 个 cache entry 在后续运行中**不再
重编译**，但也**不会被使用**。我们抓的 sample kernel 是其中一个 cfg=(num_warps=4, num_stages=3,
BLOCK_NCL=128) 的产物。

---

## 7. 完整下沉链总览

```
┌────────────────────────────────────────────────────────────────────────────┐
│   用户高级代码：examples/vgg16_snn/vgg16_via_nir.py                          │
│   ├─ build_vgg16_snn()   构造 nn.Sequential（13 Conv + 13 BN + 15 LIF +     │
│   │                                          5 AvgPool + 3 Linear）         │
│   ├─ fuse_conv_bn_eval_modules()                  BN 折进 Conv               │
│   ├─ nir_exchange.export_to_nir(folded, ...)                                │
│   └─ nir_exchange.import_from_nir(graph, device='cuda', step_mode='m')      │
└────────────────────────────────────────────────────────────────────────────┘
                          │
   §2  转换层 (SJ + nirtorch)
                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│   nir.NIRGraph (39 nodes, 38 edges) —— §3                                   │
│   ├─ {Input, 13 Conv2d, 15 LIF, 5 AvgPool2d, Flatten, 3 Affine, Output}     │
│   └─ HDF5 序列化 → 任何 NIR-aware 框架可直接 nir.read 加载                  │
└────────────────────────────────────────────────────────────────────────────┘
                          │
   §4  nirtorch.nir_to_torch 重建
                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│   torch.fx.GraphModule.forward  (自动生成 Python 源码)                       │
│   _0_0 = self._0_0(input_1); _0_2 = self._0_2(_0_0); ...                    │
└────────────────────────────────────────────────────────────────────────────┘
                          │
   §5 / §6  PyTorch eager dispatch
              │
      ┌───────┴───────┐
      │               │
   (无状态层)       (LIF 节点)
      │               │
      ▼               ▼
  cuDNN /        SJ 手写 Triton kernel
  cuBLAS /          │
  ATen native    @triton.jit JIT 编译
  CUDA 池化         │
      │             ▼
      │      .source → .ttir → .ttgir → .llir → .ptx → .cubin
      │             │
      ▼             ▼
  GPU SM 执行（cuDNN/cuBLAS/ATen kernels  +  自定义 Triton kernel）
```

---

## 8. 复现命令（一段 bash 包打）

下面这段在仓库根目录跑，重新生成本文引用的所有产物（NIR 图 + fx 源码 + 一份完整 LIF
Triton IR 五级）：

```bash
# 工作目录 = 仓库根
cd /home/charlley/Code/Triton-Pass-Analysis

# 1) 清干净 Triton cache，保证新一轮 JIT 编译产物可识别
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache

# 2) 跑一次 NIR 版 VGG16-SNN forward —— 自动触发 export_to_nir → import_from_nir →
#    LIF kernel JIT 编译；同时把 NIR 图与 fx 源码 dump 到 Document/IR-Trace/nir_lif_kernel/
conda run -n triton-dev-cuda131 python <<'PY'
import sys, pathlib
sys.path.insert(0, "examples/vgg16_snn")
import torch, torch.nn as nn
from spikingjelly.activation_based import functional, neuron, nir_exchange
from spikingjelly.activation_based.functional.conv_bn_fusion import fuse_conv_bn_eval_modules

# 与 vgg16_via_nir.py 完全一致的网络
torch.manual_seed(42)
VGG16_CFG = [64,64,'P',128,128,'P',256,256,256,'P',512,512,512,'P',512,512,512,'P']
feats, in_ch = [], 3
for v in VGG16_CFG:
    if v == 'P': feats.append(nn.AvgPool2d(2, 2))
    else:
        feats += [nn.Conv2d(in_ch, v, 3, padding=1), nn.BatchNorm2d(v), neuron.LIFNode(step_mode='s')]
        in_ch = v
model = nn.Sequential(nn.Sequential(*feats), nn.Sequential(
    nn.Flatten(), nn.Linear(512*7*7, 4096), neuron.LIFNode(step_mode='s'),
    nn.Linear(4096, 4096), neuron.LIFNode(step_mode='s'), nn.Linear(4096, 1000))).eval()
folded = fuse_conv_bn_eval_modules(model)

OUT = pathlib.Path("Document/IR-Trace/nir_lif_kernel"); OUT.mkdir(parents=True, exist_ok=True)
graph = nir_exchange.export_to_nir(folded, example_input=torch.rand(1,3,224,224),
                                   dt=1e-4, save_path=str(OUT/"vgg16_snn.nir.h5"))
with open(OUT/"vgg16_snn.nir.repr.txt", "w") as f:
    f.write(f"# nodes: {len(graph.nodes)}  edges: {len(graph.edges)}\n\n## Nodes\n")
    for n, node in graph.nodes.items():
        f.write(f"  '{n}': {type(node).__name__}({getattr(node,'input_type','?')} -> {getattr(node,'output_type','?')})\n")
    f.write("\n## Edges\n")
    for s, d in graph.edges: f.write(f"  ({s!r}, {d!r})\n")
gm = nir_exchange.import_from_nir(graph, dt=1e-4, device="cuda", step_mode="m"); gm.eval()
(OUT/"vgg16_snn.fx_graph_module.py").write_text(gm.code)
with torch.no_grad(): gm(torch.randn(4, 1, 3, 224, 224, device="cuda"))
torch.cuda.synchronize()
print("done. captured artifacts in", OUT)
PY

# 3) 从 cache 里挑一份 LIF kernel 五级 IR 拷到 Document/IR-Trace/nir_lif_kernel/sample_kernel/
SAMPLE=$(find ~/.triton/cache -name "_multistep_lif_forward_kernel.json" -printf "%h\n" | sort -u | head -1)
DEST=Document/IR-Trace/nir_lif_kernel/sample_kernel
rm -rf "$DEST" && mkdir -p "$DEST"
cp "$SAMPLE"/_multistep_lif_forward_kernel.{source,ttir,ttgir,llir,ptx,cubin,json} "$DEST"/
cp "$SAMPLE"/__grp___multistep_lif_forward_kernel.json "$DEST"/
ls "$DEST"

# 4)（可选）查看 SASS
cuobjdump --dump-sass "$DEST"/_multistep_lif_forward_kernel.cubin | head -40
```

跑完后所有产物都落在
[`Document/IR-Trace/nir_lif_kernel/`](../IR-Trace/nir_lif_kernel/)，与本文引用一一对应。

---

## 9. 这份文档**不**保证的事

明确边界：

- **只覆盖推理路径**（eval, no_grad）。训练路径下 `LIFNode.multi_step_forward` 走的是
  [`lif.py:451`](../../spikingjelly/spikingjelly/activation_based/neuron/lif.py#L451) 的 training
  分支，与本文 §6 完全不同；surrogate function 还需要 autograd backward。
- **只覆盖 fp32**。dtype 切换后 SJ multistep_lif kernel 可能落入 SJ 的 fallback 分支（见
  [SpikingJelly-Triton-Patch.md](../../examples/vgg16_snn/SpikingJelly-Triton-Patch.md)），TTIR / TTGIR 也会因
  `tensor<...xf16>` 而不同。
- **`sample_kernel/` 是 24 份 LIF cache 中的一份**。每个 LIF 实例 + autotune cfg 都有自己的
  cache，本文展示的只是一种 `(num_warps=4, num_stages=3, BLOCK_NCL=128)` 的代表。其他 cfg
  的 PTX 行数会有差异（实测 491 vs 529）但算法逻辑相同。
- **算子可达性遵循 NIR 当前协议**（v1.0.8）。如果未来 NIR 加入 BatchNorm / MaxPool / 残差等新
  原语，§1.4 的限制会随之放宽，SJ 端也需要相应扩展 `_ModuleMapper.map_dict`。
