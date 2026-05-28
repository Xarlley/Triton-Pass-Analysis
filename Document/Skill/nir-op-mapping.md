# NIR 原语 ↔ PyTorch / SpikingJelly / ATen 映射详表

> 本文枚举 **NIR v1.0.8 协议**里的全部原语，对照它们在**出口方向**（PyTorch 模块 → NIR 节点）和**入口方向**（NIR 节点 → PyTorch 模块）上具体映射到哪个 `torch.nn.*` 或 `spikingjelly.activation_based.*` 类，以及在 PyTorch eager 下最终落到哪个 ATen op / GPU 库。
>
> 所有映射均通过查阅 [`nir/`](../../nir/) / [`nirtorch/`](../../nirtorch/) / [`spikingjelly/spikingjelly/activation_based/nir_exchange/`](../../spikingjelly/spikingjelly/activation_based/nir_exchange/) 三套源码直接整理。运行时 ATen → GPU 库的归宿与本仓库 [`Document/Skill/nir-call-stack-trace.md`](nir-call-stack-trace.md) 第 5–6 节的真实捕获一致。
>
> 关联文档：
> - [`Implementation-Modes.md`](../../examples/vgg16_snn/Implementation-Modes.md) §6 — NIR 实现的 cuDNN 具体调用路径
> - [`nir-call-stack-trace.md`](nir-call-stack-trace.md) §6 — LIF 节点下沉到 Triton 五级 IR 的真机捕获
> - [`spikingjelly-nir-implementation.md`](spikingjelly-nir-implementation.md) §1.4 — 协议级强约束清单

---

## 1. 概念分层

```
┌────────────────────────────────────────────────────────────────┐
│ NIR 协议 (nir 1.0.8) —— 一组带 numpy 字段的有向图节点描述          │
│   nir.Conv2d, nir.LIF, nir.Affine, nir.AvgPool2d, ...           │
└────────────────────────────────────────────────────────────────┘
        ▲                                              │
        │ map_dict (PyTorch → NIR)         (NIR → PyTorch) map_dict
        │                                              ▼
┌────────────────────────────────────────────────────────────────┐
│ nirtorch (DEFAULT_MAP) + 用户提交的 map 字典                      │
│   torch_to_nir / nir_to_torch 双向翻译；遍历 FX node 派发         │
└────────────────────────────────────────────────────────────────┘
        ▲                                              │
        │ _ModuleMapper.map_dict          _NodeMapper.map_dict
        │                                              ▼
┌────────────────────────────────────────────────────────────────┐
│ SpikingJelly nir_exchange (薄桥层) —— 提交两个映射词典              │
│   to_nir.py:62-74    {nn.Conv2d → map_conv2d, ...}              │
│   from_nir.py:32-40  {nir.LIF → map_lif, ...}                   │
└────────────────────────────────────────────────────────────────┘
        ▲                                              │
        │ 用户提供的网络 (nn.Sequential / VGG16SNN)       重建出 fx.GraphModule
        │                                              ▼
┌────────────────────────────────────────────────────────────────┐
│ PyTorch nn.Module (torch.nn.*  +  spikingjelly.activation_based.*) │
└────────────────────────────────────────────────────────────────┘
                              │ forward()
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ ATen + cuDNN / cuBLAS / ATen native CUDA / SJ Triton kernel       │
└────────────────────────────────────────────────────────────────┘
```

NIR 协议本身**只描述结构**，不带任何运行时语义；翻译到 PyTorch 后才接入 ATen，再经各后端落到 GPU。

---

## 2. NIR 原语完整清单 + 双向支持矩阵

NIR v1.0.8 在 [`nir/nir/__init__.py`](../../nir/nir/__init__.py) 导出 17 个节点类，分三类：

### 2.1 图骨架节点（任何 NIR 图都包含）

| 节点 | 定义位置 | 字段 |
|---|---|---|
| `nir.NIRGraph` | [graph.py:21](../../nir/nir/ir/graph.py#L21) | `nodes: dict, edges: list[tuple]` —— 子图本身也是 NIRNode，可嵌套 |
| `nir.Input` | [graph.py:498](../../nir/nir/ir/graph.py#L498) | `input_type: dict[str, np.ndarray]` |
| `nir.Output` | [graph.py:526](../../nir/nir/ir/graph.py#L526) | `output_type: dict[str, np.ndarray]` |
| `nir.Identity` | [graph.py:554](../../nir/nir/ir/graph.py#L554) | passthrough，常用于占位 |

### 2.2 无状态算子节点

| 节点 | 定义位置 | 关键字段 |
|---|---|---|
| `nir.Conv1d` | [conv.py:11](../../nir/nir/ir/conv.py#L11) | `input_shape, weight (C_out·C_in·N), stride, padding, dilation, groups, bias` |
| `nir.Conv2d` | [conv.py:75](../../nir/nir/ir/conv.py#L75) | `input_shape (N_x,N_y), weight (C_out·C_in·W_x·W_y), stride, padding, dilation, groups, bias` |
| `nir.Linear` | [linear.py:41](../../nir/nir/ir/linear.py#L41) | `weight` （无 bias）|
| `nir.Affine` | [linear.py:10](../../nir/nir/ir/linear.py#L10) | `weight, bias` |
| `nir.Scale` | [linear.py:62](../../nir/nir/ir/linear.py#L62) | `scale: np.ndarray` |
| `nir.Flatten` | [flatten.py:12](../../nir/nir/ir/flatten.py#L12) | `start_dim=1, end_dim=-1, input_type` |
| `nir.AvgPool2d` | [pooling.py:26](../../nir/nir/ir/pooling.py#L26) | `kernel_size, stride, padding`（均为 np.ndarray (H, W)）|
| `nir.SumPool2d` | [pooling.py:10](../../nir/nir/ir/pooling.py#L10) | `kernel_size, stride, padding` |
| `nir.Threshold` | [threshold.py:10](../../nir/nir/ir/threshold.py#L10) | `threshold: np.ndarray` |
| `nir.Delay` | [delay.py:10](../../nir/nir/ir/delay.py#L10) | `delay: np.ndarray`（时间延迟）|

### 2.3 神经元（有状态）节点

| 节点 | 定义位置 | 关键字段 | 语义 |
|---|---|---|---|
| `nir.I` | [neuron.py:122](../../nir/nir/ir/neuron.py#L122) | `r` | 不发放：积分器 `dv/dt = r·I` |
| `nir.LI` | [neuron.py:187](../../nir/nir/ir/neuron.py#L187) | `tau, r, v_leak` | 不发放：泄漏积分器 |
| `nir.IF` | [neuron.py:142](../../nir/nir/ir/neuron.py#L142) | `r, v_threshold, v_reset` | 整数发放，无泄漏 |
| `nir.LIF` | [neuron.py:216](../../nir/nir/ir/neuron.py#L216) | `tau, r, v_leak, v_threshold, v_reset` | LI + 阈值 + 重置 |
| `nir.CubaLI` | [neuron.py:10](../../nir/nir/ir/neuron.py#L10) | `tau_syn, tau_mem, r, v_leak, w_in` | 双时间常数（突触+膜），不发放 |
| `nir.CubaLIF` | [neuron.py:53](../../nir/nir/ir/neuron.py#L53) | `tau_syn, tau_mem, r, v_leak, v_threshold, v_reset, w_in` | CubaLI + 阈值发放 |

---

## 3. 出口方向：PyTorch 模块 → NIR 节点（`export_to_nir`）

按 NIR 节点类型分组列出每个目标 NIR 节点能从哪个 PyTorch 类映射过来。

### 3.1 映射来源（实际派发顺序）

`export_to_nir` 的派发字典是
[`SJ._ModuleMapper.map_dict | nirtorch.DEFAULT_MAP`](../../nirtorch/nirtorch/torch_tracer.py#L122)
两份合并（Python `a | b` **右侧覆盖左侧**，所以 `nirtorch.DEFAULT_MAP` 里的项会盖过 SJ 同 key 的项；详见 [nir-call-stack-trace.md §1.2](nir-call-stack-trace.md)）。

实际派发优先级：

```
PyTorch 模块的 class
  ↓ 查 SJ 提交 + nirtorch DEFAULT_MAP 合并的字典
  ↓ key 命中（按类的 type identity，不走继承）
  ↓ 调对应的工厂函数 ─→ 返回 nir.XXX 节点实例
```

### 3.2 完整出口映射表

| PyTorch 模块类 | 走哪个 mapper | 映射出的 NIR 节点 | 备注 |
|---|---|---|---|
| `torch.nn.Conv2d` | SJ `to_nir.py:85 map_conv2d` | `nir.Conv2d` | 自动从 `ShapeProp` 推 `input_shape (H,W)` |
| `spikingjelly.layer.Conv2d` | 同上 | `nir.Conv2d` | （`layer.Conv2d` 继承 `nn.Conv2d`，但 SJ 字典里两条 key 都显式注册）|
| `torch.nn.Conv1d` | ✗ 未注册 | — | SJ 不支持 1D 卷积导出；nirtorch DEFAULT_MAP 也没注册 |
| `torch.nn.Linear` | **nirtorch `_map_linear`**（覆盖了 SJ 同 key）| `nir.Affine` (bias 存在时) / `nir.Linear` (无 bias 时) | SJ 自己的 `map_linear` 因 dict 覆盖方向**被遮蔽** |
| `spikingjelly.layer.Linear` | SJ `to_nir.py:79 map_linear` | 同上 | `layer.Linear` 不在 nirtorch DEFAULT 里，SJ 版本生效 |
| `torch.nn.AvgPool2d` | SJ `to_nir.py:103 map_avgpool2d` | `nir.AvgPool2d` | |
| `spikingjelly.layer.AvgPool2d` | 同上 | `nir.AvgPool2d` | |
| `torch.nn.Flatten` | SJ `to_nir.py:110 map_flatten` | `nir.Flatten` | `start_dim/end_dim` 按多步模式 `'m'` 时 -1 偏移补正（去 T 维）|
| `spikingjelly.layer.Flatten` | 同上 | `nir.Flatten` | |
| `spikingjelly.neuron.IFNode` | SJ `to_nir.py:126 map_if` | `nir.IF` | `r = 1/dt`, `v_reset=None`(软复位)被强制写成 `0.0` ⚠️ |
| `spikingjelly.neuron.LIFNode` | SJ `to_nir.py:154 map_lif` | `nir.LIF` | `tau_ = tau·dt`, `r = 1.0 if decay_input else tau`，软复位→硬复位 ⚠️ |
| `spikingjelly.neuron.ParametricLIFNode` | SJ `to_nir.py:188 map_plif` | `nir.LIF` | tau 由 `sigmoid(w)` 反推；ParametricLIF 在 NIR 里**降级为普通 LIF**（无独立 PLIF 节点）|

### 3.3 不被支持的 PyTorch 模块（碰到立即 `KeyError` 失败）

下面这些模块在网络里**会让 `export_to_nir` 直接抛出 `ValueError: Unknown module encountered`**（[`torch_tracer.py:239`](../../nirtorch/nirtorch/torch_tracer.py#L239)）：

| 不支持的模块 | 原因 |
|---|---|
| `nn.BatchNorm{1,2,3}d`, `layer.BatchNorm{1,2,3}d` | **NIR 协议没有 BN 原语**。eval 模式下必须在导出前用 `fuse_conv_bn_eval_modules` fold 进 Conv 的 bias |
| `nn.MaxPool{1,2,3}d`, `layer.MaxPool{1,2,3}d` | **NIR 协议没有 MaxPool 原语**（只有 AvgPool/SumPool）。需手动替换为 AvgPool |
| `nn.LayerNorm`, `nn.GroupNorm` | NIR 协议无对应原语 |
| `nn.Conv1d`, `nn.Conv3d`, `nn.ConvTranspose*d` | SJ 没注册 Conv1d export（NIR 协议本身有 `Conv1d`，但 SJ 不映射）；NIR 没有 Conv3d/ConvTranspose 原语 |
| `nn.LSTM`, `nn.GRU`, `nn.MultiheadAttention` | NIR 协议无对应原语 |
| `nn.Dropout`, `nn.ReLU`, `nn.LeakyReLU` 等激活 | NIR 协议无 dropout 与显式激活原语（`nir.Threshold` 不被 SJ 注册）|
| `neuron.EIFNode`, `neuron.IzhikevichNode`, `neuron.QIFNode`, `neuron.LIFNodeRec` 等 SJ 进阶神经元 | SJ 没在 `_ModuleMapper.map_dict` 里注册 |

工作流约束：要导出 SJ 网络到 NIR，**网络结构受 §3.2 那 12 行支持矩阵限制**；其他算子要么手动重构（fold-BN、AvgPool 替 MaxPool），要么放弃 NIR 路径。

---

## 4. 入口方向：NIR 节点 → PyTorch 模块（`import_from_nir`）

`import_from_nir` 把每个 NIR 节点重建为一个 SJ / torch 模块，再用 `nirtorch.graph_executor` 把它们按 edges 拼回 `torch.fx.GraphModule`。派发字典是 `SJ._NodeMapper.map_dict | nirtorch.DEFAULT_MAP`（同样右侧覆盖）。

### 4.1 完整入口映射表

| NIR 节点 | 走哪个 mapper | 映射出的 PyTorch 模块 | 备注 |
|---|---|---|---|
| `nir.Input` | nirtorch `lambda i: nn.Identity()` | `torch.nn.Identity` | 不做任何计算 |
| `nir.Output` | nirtorch `lambda o: nn.Identity()` | `torch.nn.Identity` | 同上 |
| `nir.Conv2d` | SJ `from_nir.py:53 map_conv2d` | `spikingjelly.layer.Conv2d(bias=True)` | weight/bias 从 numpy 反序列化 |
| `nir.Conv1d` | **nirtorch `_default_map_conv1d`**（SJ 没注册）| `torch.nn.Conv1d` | 不经过 SJ wrapper |
| `nir.Affine` | SJ `from_nir.py:42 map_affine` | `spikingjelly.layer.Linear(bias=True)` | weight + bias 反序列化 |
| `nir.Linear` | SJ `from_nir.py:48 map_linear` | `spikingjelly.layer.Linear(bias=False)` | 仅 weight |
| `nir.Scale` | ✗ 都没注册 | — | SJ 不实现，nirtorch DEFAULT 不实现 |
| `nir.Flatten` | SJ `from_nir.py:86 map_flatten` | `spikingjelly.layer.Flatten` | `start_dim/end_dim` +1 偏移（恢复 T 维）|
| `nir.AvgPool2d` | SJ `from_nir.py:75 map_avgpool2d` | `spikingjelly.layer.AvgPool2d` | |
| `nir.SumPool2d` | **nirtorch `_default_map_sumpool2d`**（SJ 没注册）| `torch.nn.LPPool2d(norm_type=1)` | LPPool with p=1 即逐元素绝对值求和；与真正的 sum-pool 在符号上有差异 |
| `nir.IF` | SJ `from_nir.py:92 map_if` | `spikingjelly.neuron.IFNode(v_threshold, v_reset)` | 要求所有元素的 `v_threshold/v_reset/r` 一致，否则 `AssertionError` |
| `nir.LIF` | SJ `from_nir.py:110 map_lif` | `spikingjelly.neuron.LIFNode(tau, decay_input, v_reset, v_threshold)` | 反推规则：`tau = tau_/dt`, `decay_input = (r==1)` —— 若是 PLIF 导出来的 LIF，**reverse 不回去**，仍是普通 LIFNode |
| `nir.I`, `nir.LI` | ✗ 都没注册 | — | SJ 不实现 |
| `nir.CubaLI`, `nir.CubaLIF` | ✗ 都没注册 | — | SJ 不实现 |
| `nir.Threshold` | ✗ 都没注册 | — | SJ 不实现 |
| `nir.Delay` | ✗ 都没注册 | — | SJ 不实现 |
| `nir.Identity` | nirtorch 自动通过（在 fx 图里 bypass）| 无 |  |

### 4.2 入口端不支持的 NIR 节点

NIR 协议里**存在**但 **SJ 端无法翻译**的 7 个原语：

```
nir.Scale, nir.SumPool2d*, nir.I, nir.LI, nir.CubaLI, nir.CubaLIF, nir.Threshold, nir.Delay
```

碰到这些会抛 `ValueError: Unknown node type ...`。`*` 标的 `nir.SumPool2d` nirtorch 有 default fallback，但落到 `nn.LPPool2d(norm_type=1)` —— 数学上**只在输入非负时**等价于 sum pool（因 LPPool 是 `(Σ|x|^p)^(1/p)`，对 SNN 的二值脉冲 0/1 输入恰好等价）。

---

## 5. 运行时：PyTorch 模块 → ATen op → GPU 后端

`import_from_nir` 返回的 fx.GraphModule 调用时（eager 路径）每个 SJ 模块最终的 GPU 后端，来自本仓库 [`Document/Skill/nir-call-stack-trace.md §2.1-2.4`](nir-call-stack-trace.md) 的真机捕获结合 PyTorch [ATen Convolution.cpp:1190-1530](../../pytorch/aten/src/ATen/native/Convolution.cpp#L1190-L1530) 后端选择逻辑。

| SJ / torch 模块 | super().forward 调到 | ATen op | 最终 GPU 后端 |
|---|---|---|---|
| `layer.Conv2d` / `nn.Conv2d` | `F.conv2d` | `aten::convolution` → `aten::cudnn_convolution` | **cuDNN** (`cudnnConvolutionForward`, 实测 `cutlass__5x_cudnn::Kernel<cutlass_tensorop_*>` / `sm80_xmma_fprop_implicit_gemm_*` / `implicit_convolve_sgemm`)|
| `layer.AvgPool2d` / `nn.AvgPool2d` | `F.avg_pool2d` | `aten::avg_pool2d` | **ATen native CUDA pool**（**非 cuDNN**，PyTorch 自家实现）|
| `layer.Linear` / `nn.Linear` | `F.linear` | `aten::linear` → `aten::addmm` | **cuBLAS** (`cublasSgemm` / `cuBLASLt`，实测 `gemmSN_TN_kernel<float, 128, 16, ...>`) |
| `layer.Flatten` / `nn.Flatten` | `tensor.view` / `reshape` | `aten::view` / `aten::flatten` | **无 GPU kernel**（仅改 stride/shape 元信息）|
| `nn.Identity` | passthrough | `aten::clone`（如果 dynamo 不消掉）/ 直通 | 无 |
| `neuron.IFNode` (multi-step, eval, CUDA, spiking surrogate) | `triton_kernel.multistep_if` | `torch.library.custom_op` → `@triton.jit` | **SJ 手写 Triton kernel** |
| `neuron.LIFNode` (同上条件) | `triton_kernel.multistep_lif` | `torch.library.custom_op` → `@triton.jit` | **SJ 手写 Triton kernel**（`_multistep_lif_forward_kernel`，见 [`Document/IR-Trace/nir_lif_kernel/sample_kernel/`](../IR-Trace/nir_lif_kernel/sample_kernel/) 完整五级 IR）|

在 `torch.compile` 路径下，所有这些 ATen op 会被 Inductor 接管重新 codegen 为 Triton kernel，仅 SJ 的 LIF / IF custom_op 保留为黑盒 launcher（详见 [`nir-call-stack-trace.md §7.8-7.9`](nir-call-stack-trace.md)）。

---

## 6. 协议级强约束清单

下列限制由 NIR 协议本身定义，与具体实现无关：

| 约束 | 出处 | 影响 |
|---|---|---|
| NIR 节点的 `input_type/output_type` **不含 T 与 B 维** | NIR 协议规范 | 多步 / 批量信息只能由调用者在 `example_input` 里推；导入时 `step_mode='m'` 由调用者指定 |
| **`nir.LIF` 不区分软/硬复位** | [LIF 节点定义](../../nir/nir/ir/neuron.py#L216) | SJ 导出时若 `v_reset=None`（软复位）会被强制改成 `v_reset=0.0`（硬复位）+ `v_leak=0.0`，**信息丢失** ⚠️ |
| **NIR 没有 MaxPool 原语** | [pooling.py](../../nir/nir/ir/pooling.py) 只有 SumPool2d/AvgPool2d | 含 MaxPool 的网络无法导出，必须手动改用 AvgPool 或 SumPool（数学不再等价）|
| **NIR 没有 BatchNorm 原语** | nir.ir 无 BN 类 | 必须 fold-BN（eval 模式）才能导出；training 网络无法直接导出 |
| **`Affine` / `Linear` 按 bias 区分** | linear.py 两个独立类 | 同一个 `nn.Linear(bias=...)` 视 bias 存在与否映射到不同 NIR 类型 |
| `Conv1d/Conv2d` **必须显式存 `input_shape`** | conv.py:37 / conv.py:102 | 出口时由 `ShapeProp` 推；NIR 不能描述未知形状的 Conv |
| 残差 / 多入口仅支持 **`operator.add`** | [`torch_tracer.py:188`](../../nirtorch/nirtorch/torch_tracer.py#L188) | concat / pointwise mul / 其他 binary op 一律 `ValueError: The only supported function is addition` |
| `stateless_module` 是单个 `Callable` 才能 dynamo 完全特化 | SJ `seq_to_ann_forward` 的 isinstance 检查 | 不影响 NIR 协议本身，但影响 path B 的 torch.compile（详见 [`SpikingJelly-Triton-Patch.md`](../../examples/vgg16_snn/SpikingJelly-Triton-Patch.md)）|

---

## 7. 实证：VGG16-SNN 在本仓库各原语的真实使用次数

来自 [`Document/IR-Trace/nir_lif_kernel/vgg16_snn.nir.repr.txt`](../IR-Trace/nir_lif_kernel/vgg16_snn.nir.repr.txt)（一次真实 `export_to_nir(vgg16_via_nir.py 的 folded 网络)` 的 39 节点 38 边 dump）：

| NIR 节点 | 出现次数 | 对应 VGG16 部位 |
|---|---:|---|
| `nir.Input` | 1 | 网络入口（推回 `[3, 224, 224]`）|
| `nir.Output` | 1 | 网络出口（`[1000]`）|
| `nir.Conv2d` | 13 | 全部 conv 层（已 fold BN，bias 字段含 BN 等价系数）|
| `nir.LIF` | 15 | 13 conv-后 LIF + 2 fc-后 LIF |
| `nir.AvgPool2d` | 5 | 5 个 2×2 池化（手动替了原 VGG 的 MaxPool）|
| `nir.Flatten` | 1 | 分类器入口 `[512,7,7] → [25088]` |
| `nir.Affine` | 3 | 3 个 fc 层（bias=True 时 SJ 映射到 `nir.Affine`，非 `nir.Linear`）|
| **合计** | **39** | |

复现命令见 [`spikingjelly-nir-implementation.md` §8](spikingjelly-nir-implementation.md)。

---

## 8. 总结：当前 SJ + nirtorch 支持矩阵速查

```
出口方向 (PyTorch → NIR):

  PyTorch                          NIR              落到哪个 mapper
  ───────────────────────────────────────────────────────────────────
  nn.Conv2d / layer.Conv2d         → nir.Conv2d     SJ map_conv2d
  nn.AvgPool2d / layer.AvgPool2d   → nir.AvgPool2d  SJ map_avgpool2d
  nn.Flatten / layer.Flatten       → nir.Flatten    SJ map_flatten
  nn.Linear (bias=True)            → nir.Affine     nirtorch DEFAULT (覆盖 SJ)
  nn.Linear (bias=False)           → nir.Linear     nirtorch DEFAULT (覆盖 SJ)
  layer.Linear                     → nir.Affine/Linear  SJ map_linear (无覆盖)
  neuron.IFNode                    → nir.IF         SJ map_if
  neuron.LIFNode                   → nir.LIF        SJ map_lif
  neuron.ParametricLIFNode         → nir.LIF        SJ map_plif (降级)
  ───────────────────────────────────────────────────────────────────
  其他模块 (BN, MaxPool, GroupNorm, LSTM, Dropout, ...)：均不支持

入口方向 (NIR → PyTorch):

  NIR                              PyTorch                落到哪个 mapper
  ───────────────────────────────────────────────────────────────────
  nir.Input / nir.Output           → nn.Identity         nirtorch DEFAULT
  nir.Conv2d                       → layer.Conv2d        SJ map_conv2d
  nir.Conv1d                       → nn.Conv1d           nirtorch DEFAULT (SJ 没注册)
  nir.Affine                       → layer.Linear(bias=True)  SJ map_affine
  nir.Linear                       → layer.Linear(bias=False) SJ map_linear
  nir.AvgPool2d                    → layer.AvgPool2d     SJ map_avgpool2d
  nir.SumPool2d                    → nn.LPPool2d(p=1)    nirtorch DEFAULT (SJ 没注册)
  nir.Flatten                      → layer.Flatten       SJ map_flatten
  nir.IF                           → neuron.IFNode       SJ map_if
  nir.LIF                          → neuron.LIFNode      SJ map_lif
  ───────────────────────────────────────────────────────────────────
  nir.Scale, nir.Threshold, nir.Delay, nir.I, nir.LI, nir.CubaLI, nir.CubaLIF
    : 全部无映射 → ValueError
```

也就是说**两条方向上都能跑通**的 NIR 原语集是：

```
{ Input, Output, Conv2d, AvgPool2d, Flatten, Affine, Linear, IF, LIF }
```

—— 这是当前 SpikingJelly + nirtorch 组合所能完整 round-trip 的全部协议子集，共 9 类节点。VGG16-SNN 的 fold-BN+AvgPool 等价形式正好完全落在这个交集里。
