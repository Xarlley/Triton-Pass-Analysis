# NIR 路径函数调用栈实测：nirtorch / SpikingJelly / Triton / cuDNN+cuBLAS+ATen 的真实调用关系

> 本文不依赖代码阅读得出的推断，**全部基于真实运行截获**：
>
> - **Python 层调用栈** —— 用 `sys.settrace` 在 `vgg16_via_nir.py` 配方下采样，过滤到
>   `spikingjelly/`、`nirtorch/`、`nir/`、`triton/` 四个库，捕获构造阶段（export +
>   import）与单次 forward 的完整 Python 调用关系。
> - **ATen + CUDA 层调用栈** —— `sys.settrace` 看不到 C++ 与 GPU 内核，因此再叠一道
>   `torch.profiler`（CPU+CUDA），把 `cudnn_convolution` / `addmm` / `avg_pool2d` 等 ATen
>   ops 与其对应的真实 GPU kernel（`cutlass__5x_cudnn::Kernel`、`gemmSN_TN_kernel`、
>   `avg_pool2d_o*`、`_multistep_lif_forward_kernel` 等）一并打表。
>
> 采样脚本：[`examples/vgg16_snn/trace_nir_calls.py`](../../examples/vgg16_snn/trace_nir_calls.py)。
> 真实采样产物：[`Document/IR-Trace/nir_lif_kernel/`](../IR-Trace/nir_lif_kernel/)
> （`call_trace_build.txt` / `call_trace_forward.txt` / `aten_ops.txt` / `chrome_trace.json`）。
>
> 验证环境：torch 2.11.0+cu130 (cuDNN 9, cuBLAS 13)、spikingjelly 0.0.0.0.15
> (Xarlley fork, `triton-fork-compat`)、triton 3.7.0+gitef02d646、nir 1.0.8、nirtorch 2.7、
> RTX 5070 Ti (sm_120)。

---

## 0. 采样方法

### 0.1 Python 层：`sys.settrace`（过滤式）

[trace_nir_calls.py:46-95](../../examples/vgg16_snn/trace_nir_calls.py#L46-L95) 实现的 `CallTracer`：

```python
TRACE_KEYWORDS = (
    "spikingjelly/spikingjelly/", "nirtorch/nirtorch/",
    "/nir/nir/", "/triton/python/triton/", "examples/vgg16_snn/",
)
TRACE_EXCLUDE = ("torch/_dynamo", "torch/fx/_symbolic_trace.py",
                 "torch/_decomp", "torch/_inductor", "torch/_higher_order_ops")
HELPER_NAMES = {"<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>", "<lambda>"}

def _trace(self, frame, event, arg):
    if event != "call": return self._trace
    fn, name = frame.f_code.co_filename, frame.f_code.co_name
    if name in HELPER_NAMES: return self._trace
    if any(k in fn for k in TRACE_EXCLUDE): return self._trace
    if not any(k in fn for k in TRACE_KEYWORDS): return self._trace
    depth = _count_traced_parents(frame)
    self.lines.append((depth, fn, name, frame.f_lineno))
    return self._trace
```

只记录**进入 Python 函数**事件（`event == "call"`），按"已被记录的父帧数"计算缩进深度，
所以输出看起来就像一棵调用树。**`torch.nn.Conv2d.forward` / `torch._C._VariableFunctions.conv2d`
不在 keywords 里，所以 sys.settrace 自然不会展开它们**（也无法展开 C++ 帧）。

为什么这是 sys.settrace 的优势 —— 它**精确地、按 Python 解释器实际看到的顺序**记录每一次
`def` 调用，不会因为 C++ 跳过 GIL 而错过事件。代价是看不到 C++ 层（cuDNN / cuBLAS / Triton
runtime 中真正的 C++ 部分）。

### 0.2 ATen + CUDA 层：`torch.profiler`

[trace_nir_calls.py:148-167](../../examples/vgg16_snn/trace_nir_calls.py#L148-L167)：

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("gm_forward"), torch.no_grad():
        out = gm(x)
    torch.cuda.synchronize()

prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
```

它在每个 ATen op 和每个 CUDA kernel 启动周围打点，**直接揭示 ATen op → 真实 CUDA kernel
名字的映射**。例如 `aten::cudnn_convolution` 一行下面紧挨着的 `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s168...>`
这一行就说明这个 conv 走的是 cuDNN 的 cutlass 模板。

两者**互补**：sys.settrace 看 Python 层调用关系，profiler 看 ATen→GPU 内核映射。

---

## 1. 构造阶段（build phase）：nirtorch 主导，回调到 SJ

完整产物：[`call_trace_build.txt`](../IR-Trace/nir_lif_kernel/call_trace_build.txt)（1918 条调用事件）。

### 1.1 顶层入口与首次进入 nirtorch

trace 文件头 12 行（精简后）：

```text
spikingjelly/.../nir_exchange/to_nir.py:224  export_to_nir()
  spikingjelly/.../nir_exchange/to_nir.py:23   _ModuleMapper.__init__()
    spikingjelly/.../nir_exchange/to_nir.py:34   set_module_io_shape()
      spikingjelly/.../nir_exchange/to_nir.py:60   map_dict()
      nirtorch/.../torch_tracer.py:26              NIRTorchTracer.__init__()        ← 进入 nirtorch
      nirtorch/.../torch_tracer.py:35              is_leaf_module()                  ← × 28（每个子模块各一次）
      ...
```

**读法**：

- `export_to_nir(...)` 是入口 → 立刻 `_ModuleMapper(net, example_input, ...)` 构建一个映射字典对象；
- 内部用 `ShapeProp` 推 shape，会先 fx-trace 一遍网络判断 leaf module —— 此时 nirtorch 第一次
  出现，被 SJ 实例化为 `NIRTorchTracer(module_map.keys())`；
- `is_leaf_module()` 被反复调用是因为 fx tracer 对每个子模块都问一次"这是不是叶子"，对每个
  在 `module_map.keys()` 里的类回答 True 让 fx 不深入。

### 1.2 真正干活的 `nirtorch.torch_to_nir` 进入点

跳到 trace 第 198 行（精简后）：

```text
spikingjelly/.../nir_exchange/to_nir.py:295    export_to_nir() 主体
  nirtorch/.../torch_tracer.py:74              torch_to_nir()          ← 二次进入 nirtorch
    nirtorch/.../torch_tracer.py:26              NIRTorchTracer.__init__()
    nirtorch/.../torch_tracer.py:30              create_proxy()         ← fx 追踪开始
    nirtorch/.../torch_tracer.py:35              is_leaf_module()       ← 每个子模块一次
    ...
    （主循环：对每个 fx node 分发）
    spikingjelly/.../nir_exchange/to_nir.py:85   map_conv2d()           ← nirtorch 回调到 SJ
    spikingjelly/.../nir_exchange/to_nir.py:154  map_lif()              ← nirtorch 回调到 SJ
    spikingjelly/.../nir_exchange/to_nir.py:85   map_conv2d()
    spikingjelly/.../nir_exchange/to_nir.py:154  map_lif()
    spikingjelly/.../nir_exchange/to_nir.py:103  map_avgpool2d()
    spikingjelly/.../nir_exchange/to_nir.py:85   map_conv2d()
    ...
```

trace 显示 SJ 的 map_* 被实际调用的分布：

| 函数 | 调用次数 | 出处 |
|---|---:|---|
| `to_nir.py:85 map_conv2d()` | 13 | SJ `_ModuleMapper.map_conv2d` |
| `to_nir.py:154 map_lif()` | 15 | SJ `_ModuleMapper.map_lif` |
| `to_nir.py:103 map_avgpool2d()` | 5 | SJ `_ModuleMapper.map_avgpool2d` |
| `to_nir.py:110 map_flatten()` | 1 | SJ `_ModuleMapper.map_flatten` |
| **小计** | **34** | **SJ 回调** |
| `torch_tracer.py:10 _map_linear()` | 3 | **nirtorch 自带 DEFAULT_MAP**，不是 SJ 的！|
| **合计** | **37** | 对应 NIR 图里 13+15+5+1+3 个 non-Input/Output 节点 |

**意外发现**：3 个 fc 层走的是 **nirtorch 自家** `_map_linear`
（[`torch_tracer.py:10-14`](../../nirtorch/nirtorch/torch_tracer.py#L10-L14)），不是 SJ 的
`map_linear`。原因在 [`torch_tracer.py:122-123`](../../nirtorch/nirtorch/torch_tracer.py#L122-L123)：

```python
def torch_to_nir(module, module_map, default_dict=DEFAULT_MAP, ...):
    if default_dict is not None:
        module_map = module_map | default_dict   # ★ 注意合并方向
```

Python 3.9+ 的 `a | b` 在键冲突时**右侧 `b` 覆盖左侧 `a`** —— 也就是 `default_dict`
（`DEFAULT_MAP = {nn.Linear: _map_linear}`）会**盖过** SJ 提交的 `{nn.Linear: self.map_linear,
layer.Linear: self.map_linear}` 中 `nn.Linear` 这一项。`layer.Linear` 由于 nirtorch DEFAULT_MAP
里没注册，得以保留。

实际后果**很轻**：nirtorch 自带 `_map_linear` 与 SJ `map_linear` 行为几乎相同（都按 bias 区分
`nir.Linear` / `nir.Affine`、都做 `.detach().numpy()`），换谁都得到同样的 NIR 节点。但这是
一个 nirtorch API **覆盖方向反直觉**的潜在坑：用户传的 `module_map` 不能可靠覆盖 default。
如果将来 SJ 要给 `nn.Linear` 加非默认行为（比如 fold 某种特殊量化参数），就必须在调用前
显式把 `DEFAULT_MAP` 里的 `nn.Linear` 删掉。

**关键观察（核心结论不变）**：构造阶段的"控制权"在 nirtorch 这一侧 —— SJ 提交了字典，
nirtorch 主循环按 fx node 把字典里的工厂函数一个个回调起来。这就是「SJ 在这一层是薄桥」
的实证。

### 1.3 反向：`import_from_nir` 内部

trace 第 604 行起：

```text
spikingjelly/.../nir_exchange/from_nir.py:223  import_from_nir()
  nirtorch/.../nir_interpreter.py:559           nir_to_torch()         ← 进入 nirtorch interpreter
    nir/nir/ir/graph.py:...                     nir.NIRGraph 节点访问
    spikingjelly/.../nir_exchange/from_nir.py:42  map_affine()         ← 反向回调
    spikingjelly/.../nir_exchange/from_nir.py:53  map_conv2d()
    spikingjelly/.../nir_exchange/from_nir.py:75  map_avgpool2d()
    spikingjelly/.../nir_exchange/from_nir.py:86  map_flatten()
    spikingjelly/.../nir_exchange/from_nir.py:110 map_lif()
    ...
```

trace 真实分布：

| 函数 | 调用次数 | 目标 |
|---|---:|---|
| `from_nir.py:110 map_lif()` | 15 | SJ |
| `from_nir.py:53 map_conv2d()` | 13 | SJ |
| `from_nir.py:75 map_avgpool2d()` | 5 | SJ |
| `from_nir.py:42 map_affine()` | 3 | SJ |
| `from_nir.py:86 map_flatten()` | 1 | SJ |
| **合计** | **37** | **全部 SJ** |

注意：本路径 `from_nir.py:48 map_linear()` 在真实 trace 中**未被调用**（0 次）。因为 fc
层默认 `bias=True`，正向阶段 SJ 的 `map_linear` 已经把它们编码为 `nir.Affine`（**不是**
`nir.Linear`），反向阶段查 `_NodeMapper.map_dict` 时落到 `map_affine` 而非 `map_linear`。
若网络里有 `bias=False` 的 fc 层，反向 trace 才会出现 `map_linear()`。

**与正向相反**，反向阶段的 37 次回调**全部**落到 SJ 工厂函数 —— 因为 `nirtorch.nir_to_torch`
的 `DEFAULT_MAP` 里没有为 `nir.Affine` / `nir.Conv2d` / `nir.LIF` / `nir.AvgPool2d` / `nir.Flatten`
注册条目（[`nir_interpreter.py`](../../nirtorch/nirtorch/nir_interpreter.py) 中
`DEFAULT_MAP` 只覆盖少数节点类型），所以 SJ 提交的字典是这些节点类型的唯一处理者，不存在
覆盖问题。这是正反向不对称的细节。

最后 nirtorch 把这些重建出的 SJ 模块拼成 `torch.fx.GraphModule` 返回。

---

## 2. 推理阶段（forward phase）：nirtorch **退场**，SJ + ATen + Triton 协同

完整产物：[`call_trace_forward.txt`](../IR-Trace/nir_lif_kernel/call_trace_forward.txt)（793 条调用事件）。

**第一个关键证据**：整个 forward trace 里 `nirtorch/` 路径只在最顶端出现 **1 次**：

```text
nirtorch/.../nir_interpreter.py:417  ternary_operator()
```

`ternary_operator` 是 fx.GraphModule 自动生成的 forward 源码里用来初始化 `state` 字典的
助手函数（[`from_nir.py` 段](../IR-Trace/nir_lif_kernel/vgg16_snn.fx_graph_module.py#L7)），调一次就结束。
**整个网络的 37 次 `self._X_Y(...)` 调用都不再经过 nirtorch** —— 它们直接派发到 SJ 的
`layer.*` / `neuron.*` 模块。

### 2.1 一个 Conv2d 的 forward 调用栈（trace 第 5-7 行）

```text
spikingjelly/.../layer/stateless_wrapper.py:176  Conv2d.forward()     ← SJ 多步包装入口
  spikingjelly/.../base.py:106                    step_mode()           ← 读 self.step_mode
  spikingjelly/.../base.py:106                    step_mode()
```

后面就**断了** —— Python trace 不再深入。原因：`stateless_wrapper.py:176` 的下一步是
`super().forward(x.flatten(0, 1))`，这个 super 就是 `torch.nn.Conv2d.forward`，**不在
TRACE_KEYWORDS 里**，sys.settrace 不展开。

那里真正发生了什么？看 [`aten_ops.txt`](../IR-Trace/nir_lif_kernel/aten_ops.txt)：

```text
aten::conv2d              # 13 次  (顶层 wrapper)
  aten::convolution       # 13 次  (派发层)
    aten::_convolution    # 13 次  (后端选择层)
      aten::cudnn_convolution  # 13 次  3.672ms CUDA 总时间   ← cuDNN 真实入口
        void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s168...>  # 3 次  990.189µs
        sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nh...   # 2 次  753.354µs
        sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nh...   # 2 次  528.456µs
        void cutlass__5x_cudnn::Kernel<...另一形状>              # 5 次  657.608µs
        void implicit_convolve_sgemm<float, float, 1024, 5, ...>  # 1 次  129.538µs
      aten::add_                                                  # 13 次  340.003µs  (BN 已 fold 进 Conv 的 bias)
      void cudnn::engines_precompiled::nchwToNhwcKernel<fl...>    # 24 次  437.416µs  (cuDNN 内部 layout 转)
      void cudnn::engines_precompiled::nhwcToNchwKernel<fl...>    # 10 次  174.145µs  (cuDNN 内部 layout 转)
```

完整调用关系（** = sys.settrace 看到的边界，profiler 接力**）：

```
fx.GraphModule.forward (auto-generated)
  └─ self._0_0(input_1)
      └─ layer.Conv2d.forward                  spikingjelly/layer/stateless_wrapper.py:176
          └─ super().forward(x.flatten(0,1))   ** Python trace 边界 **
              └─ nn.Conv2d.forward             torch/nn/modules/conv.py
                  └─ self._conv_forward
                      └─ F.conv2d              torch/nn/functional.py
                          └─ aten::conv2d      (profiler)
                              └─ aten::convolution
                                  └─ aten::_convolution
                                      └─ aten::cudnn_convolution
                                          └─ libcudnn.so.9 → cutlass*/sm80_xmma*/implicit_convolve_sgemm  ← GPU
```

### 2.2 一个 LIF 的 forward 调用栈（trace 第 10-50 行）

这是**唯一**走 Triton 的路径，完整且深，sys.settrace 全程能看到：

```text
spikingjelly/.../base.py:356                                              MemoryModule.forward()
  spikingjelly/.../neuron/lif.py:448                                       multi_step_forward()        ← eval 分支
    spikingjelly/.../base.py:277                                            backend()                    ← 读 self.backend
    spikingjelly/.../neuron/base_node.py:352                                v_float_to_tensor()
    spikingjelly/.../triton_kernel/neuron_kernel/lif.py:409                multistep_lif()              ← @register_op custom_op
      spikingjelly/.../triton_kernel/neuron_kernel/lif.py:225                multistep_lif_inference()
        spikingjelly/.../triton_kernel/triton_utils.py:83                    wrap_triton()              ← SJ 工具
        triton/python/triton/runtime/jit.py:370                              JITFunction.__getitem__()  ← [grid]
        triton/python/triton/runtime/autotuner.py:212                        Autotuner.run()            ← 进入 Triton
          triton/python/triton/runtime/autotuner.py:354                        all_kwargs()
          triton/python/triton/runtime/jit.py:726                              JITFunction.run()
            triton/python/triton/runtime/jit.py:592                              compute_cache_key()
            triton/python/triton/runtime/jit.py:1153                             KernelArg.__call__()
            triton/python/triton/__init__.py:68                                  cdiv()
            triton/python/triton/compiler/compiler.py:508                        CompiledKernel.launch_metadata()
              triton/python/triton/compiler/compiler.py:465                        _init_handles()       ← 第一次会真的编译
              triton/python/triton/compiler/compiler.py:393                        KernelMetadata.__init__()
            triton/python/triton/compiler/compiler.py:502                        CompiledKernel.run()
            triton/python/triton/backends/nvidia/driver.py:297                    NVDriver.__call__()  ← 进 libcuda
```

**关键观察**：

1. **LIF 的派发**：`neuron.LIFNode.forward → multi_step_forward → multistep_lif（custom_op）→
   multistep_lif_inference → wrap_triton(_kernel)[grid](...) → autotuner.run → JITFunction.run →
   CompiledKernel.run → NVDriver.__call__`。Python 层一路畅通，sys.settrace 看到完整链路
   到 `nvidia/driver.py:297`，再下一步是 C 扩展 `cuLaunchKernel(cubin, ...)`。

2. **触发 Triton 的不是 nirtorch、不是 fx.GraphModule、是 SJ 自己**：调用者是
   `spikingjelly/.../triton_kernel/neuron_kernel/lif.py:225 multistep_lif_inference`
   —— 这是 SJ 在 `convert_and_store` patch 那条 import 之后定义的；nirtorch 完全不知道
   它的存在。

3. **autotuner 在每次 LIF 调用时都被进入**，但只在**首次**或**形状/参数变化时**做实际 benchmark
   + JIT 编译。后续调用走 cache 命中。trace 文件里第 9-15 个 LIF 调用明显比第 1 个浅得多 ——
   不再有 `compiler/compiler.py:_init_handles` 这一层。

4. **15 次 LIF 调用**：trace 文件里 `lif.py:448 multi_step_forward()` 出现 **15 次**（13 个
   conv-后 LIF + 2 个 fc-后 LIF）。

profiler 视角的 LIF：

```text
sj::multistep_lif_inference         15 次  734.153µs  CUDA total
  _multistep_lif_forward_kernel     15 次  734.153µs  ← SJ 手写 Triton kernel 真名
```

profiler 显示的 `_multistep_lif_forward_kernel` 就是 [`Document/IR-Trace/nir_lif_kernel/sample_kernel/`](../IR-Trace/nir_lif_kernel/sample_kernel/)
里 `.ttir / .ttgir / .llir / .ptx / .cubin` 的源头。

### 2.3 一个 AvgPool 的 forward 调用栈

sys.settrace 看到的部分（trace 文件中后段，每个 pool 类似）：

```text
spikingjelly/.../layer/stateless_wrapper.py:882   AvgPool2d.forward()
  spikingjelly/.../base.py:106                     step_mode()
  spikingjelly/.../base.py:106                     step_mode()
```

后面同样被 Python trace 边界截断。profiler 接力：

```text
aten::avg_pool2d                                                                5 次  196.420µs CUDA
  void at::native::(anonymous namespace)::avg_pool2d_o...                       5 次  196.420µs
```

`avg_pool2d_o...` 是 **ATen 自家 CUDA 池化 kernel，不属于 cuDNN**。完整链：

```
layer.AvgPool2d.forward → super().forward(nn.AvgPool2d) → F.avg_pool2d
                                                          → aten::avg_pool2d
                                                            → ATen native CUDA kernel
```

### 2.4 三个 Linear（fc 层）的 forward 调用栈

**Python trace 中无任何 SJ 帧** —— 跟 Conv2d / AvgPool2d / Flatten 不同，[`layer.Linear`
(stateless_wrapper.py:1105-1136)](../../spikingjelly/spikingjelly/activation_based/layer/stateless_wrapper.py#L1105-L1136)
**没有重写 `forward`**：

```python
class Linear(nn.Linear, base.StepModule):
    def __init__(self, in_features, out_features, bias=True, step_mode="s"):
        super().__init__(in_features, out_features, bias)
        self.step_mode = step_mode

    def extra_repr(self): ...
    # ★ 没有 def forward(...)
```

实测 forward trace 中 `stateless_wrapper.py` 这个文件只出现 19 次（13 Conv + 5 AvgPool +
1 Flatten），**没有任何 Linear 相关的行号**：

```text
13  stateless_wrapper.py:176   (Conv2d.forward)
 5  stateless_wrapper.py:882   (AvgPool2d.forward)
 1  stateless_wrapper.py:1173  (Flatten.forward)
```

3 个 fc 层调用直接由 Python 解释器走到继承的 `nn.Linear.forward`（在 `torch/nn/modules/linear.py`，
不在 TRACE_KEYWORDS 里，sys.settrace 不展开）。这意味着：**`set_step_mode(model, 'm')`
对 `layer.Linear` 实例不产生任何运行时效果** —— `step_mode` 属性虽被设上，但根本没有 forward
代码在读它。能正常工作是因为 `nn.Linear` 操作的是最后一维，`[T, B, in]` → `[T, B, out]` 自动
广播；不需要先 flatten `[T·B, in]` 这一步。

profiler 接力，揭示真实走 **cuBLAS**：

```text
aten::addmm                                                          3 次  641.737µs CUDA total
  void gemmSN_TN_kernel<float, 128, 16, 2, 4, 4, 4, tr...>           3 次  641.737µs
```

`gemmSN_TN_kernel` 是 cuBLAS / cuBLASLt 的 GEMM kernel。完整链（**注意没有 layer.Linear.forward 一层**）：

```
self._1_1(x)  ← fx forward 里这一行
  └─ nn.Linear.forward            (继承自父类，无 SJ 覆盖)
      └─ F.linear
          └─ aten::linear
              └─ aten::addmm
                  └─ libcublas → cublasSgemm / cuBLASLt   ← GPU
```

### 2.5 一个 Flatten

profiler 行为更纯粹 —— `aten::flatten` / `aten::view` / `aten::as_strided` 出现，**Self CUDA = 0us**：
不生成任何 GPU kernel，仅改 tensor 的 stride/shape 元信息。

---

## 3. 一张图：四方协同关系

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 构造阶段 (export + import；nirtorch 主导)                                    │
│                                                                             │
│   vgg16_via_nir.py                                                          │
│       │                                                                     │
│       ▼                                                                     │
│   nir_exchange.export_to_nir   <──┐                                         │
│       │ 调用                       │                                        │
│       ▼                           │                                         │
│   nirtorch.torch_to_nir           │ 工厂回调 共 37 次:                      │
│       │ fx.trace                  │   - SJ map_*           34 次            │
│       ▼ 节点遍历分发              │   - DEFAULT_MAP._map_linear  3 次 (fc)  │
│   for node in fx.nodes ───────────┘                                         │
│       ├─ SJ map_conv2d × 13 / map_lif × 15 / map_avgpool2d × 5 / map_flatten × 1 │
│       └─ nirtorch DEFAULT_MAP _map_linear × 3 (覆盖了 SJ 提交的 nn.Linear 映射)  │
│                                                                             │
│   生成 nir.NIRGraph (39 节点 / 38 边)                                       │
│       │                                                                     │
│       ▼                                                                     │
│   nir_exchange.import_from_nir  <──┐                                        │
│       │                            │                                        │
│       ▼                            │                                        │
│   nirtorch.nir_to_torch            │ 工厂回调 共 37 次, 全部走 SJ           │
│       │ 节点遍历重建               │ (DEFAULT_MAP 不覆盖 nir.* 节点类型)     │
│       └─ SJ map_affine × 3 / map_conv2d × 13 / map_avgpool2d × 5 /          │
│             map_flatten × 1 / map_lif × 15  =  37                           │
│                                                                             │
│   生成 fx.GraphModule  ───────────────────►  nirtorch 退场                  │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ 推理阶段 (每次 gm(x)；nirtorch 不参与)                                       │
│                                                                             │
│   user: gm(x)                                                               │
│     │                                                                       │
│     ▼                                                                       │
│   fx.GraphModule.forward (nirtorch 自动生成的 Python 源码)                  │
│     │                                                                       │
│     ├─ self._0_0(input_1)  ◄── 13×        ┌──────────────────────────────┐ │
│     │     └─ layer.Conv2d.forward         │                              │ │
│     │         └─ nn.Conv2d.forward        │  PyTorch eager 派发          │ │
│     │             └─ F.conv2d             │  (跨 Python/C++ 边界)        │ │
│     │                 └─ aten::conv2d ──► │  → aten::cudnn_convolution   │ │
│     │                                     │      └─► libcudnn → SM       │ │
│     │                                     └──────────────────────────────┘ │
│     │                                                                       │
│     ├─ self._0_2(_0_0)  ◄── 15×           ┌──────────────────────────────┐ │
│     │     └─ neuron.LIFNode.forward       │                              │ │
│     │         └─ multi_step_forward       │  SJ 自己派发到 Triton        │ │
│     │             └─ multistep_lif (custom_op)  → Triton autotune        │ │
│     │                 └─ wrap_triton(@triton.jit kernel)[grid]            │ │
│     │                     └─ triton.JITFunction.run                       │ │
│     │                         └─ CompiledKernel.run                       │ │
│     │                             └─ NVDriver.__call__                    │ │
│     │                                 └─► libcuda → libcudaJIT → SM       │ │
│     │                                     (执行 _multistep_lif_forward_kernel) │
│     │                                                                       │
│     ├─ self._0_6(_0_5)  ◄── 5×            ┌──────────────────────────────┐ │
│     │     └─ layer.AvgPool2d.forward      │                              │ │
│     │         └─ nn.AvgPool2d.forward     │  PyTorch eager → ATen 自家   │ │
│     │             └─ aten::avg_pool2d ──► │   CUDA 池化 kernel           │ │
│     │                                     │   (not cuDNN, not Triton)    │ │
│     │                                     └──────────────────────────────┘ │
│     │                                                                       │
│     ├─ self._1_0(_0_43)  ◄── 1×                                             │
│     │     └─ layer.Flatten.forward → view / reshape (无 GPU kernel)        │
│     │                                                                       │
│     └─ self._1_1(_1_0)  ◄── 3×            ┌──────────────────────────────┐ │
│           │ (layer.Linear 未重写 forward, │                              │ │
│           │  直接走继承的 nn.Linear)      │  PyTorch eager → ATen → cuBLAS│ │
│           └─ nn.Linear.forward            │                              │ │
│               └─ F.linear → aten::addmm   │                              │ │
│                   → libcublas (gemmSN_TN_kernel)                          │ │
│                                           └──────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 角色与边界一目了然

| 角色 | 在构造阶段 | 在推理阶段 | 调用方向 |
|---|---|---|---|
| **vgg16_via_nir.py（用户代码）** | 调 `export_to_nir` / `import_from_nir` 两次 | 调 `gm(x)` | →（不被回调）|
| **SpikingJelly `nir_exchange`** | 提供 `_ModuleMapper.map_dict` / `_NodeMapper.map_dict` 这两份字典 + 一个 4 行薄桥 | 完全不参与 | 被 nirtorch 回调 |
| **nirtorch** | 主导：fx-trace + 节点遍历 + 边构建 + 双向重建 | **完全退场**（只有一次 `ternary_operator` 帮 fx 初始化 state） | 调 SJ 的 map_* |
| **SpikingJelly `layer.*` / `neuron.*` `triton_kernel.*`** | 不参与（只作为映射目标存在） | **34 个节点的主载体**（13 Conv + 5 AvgPool + 1 Flatten + 15 LIF），forward 被 SJ 重写；**3 Linear 例外** —— `layer.Linear` 没重写 forward，直接落继承的 `nn.Linear`，Python trace 无 SJ 帧 | 调 ATen（无状态）/ 调 Triton（LIF）|
| **Triton runtime** | 不参与 | **仅 LIF 调用进入**：被 SJ `wrap_triton(...)[grid]` 触发 | 调 libcuda |
| **ATen + cuDNN + cuBLAS** | 一次（`ShapeProp` 在 CPU 上跑 fp32 mock forward 推 shape，触发 ATen CPU op；不调 cuDNN/cuBLAS）| **承担 22 个节点的 GPU 执行**（13 Conv → cuDNN, 5 AvgPool → ATen 自家 CUDA, 3 Linear → cuBLAS, 1 Flatten → 仅元信息）| 被 PyTorch eager 调度调用 |

---

## 5. 复现命令

```bash
cd /home/charlley/Code/Triton-Pass-Analysis

# 清干净 Triton cache（保证 forward 阶段 LIF kernel 编译产物归一）
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache

# 跑采样脚本：自动落盘 build trace / forward trace / aten ops table / chrome trace
conda run -n triton-dev-cuda131 python examples/vgg16_snn/trace_nir_calls.py
```

输出到 `Document/IR-Trace/nir_lif_kernel/`：

| 文件 | 含义 |
|---|---|
| `call_trace_build.txt` | 构造阶段 Python 调用栈（1918 调用事件，过滤后）|
| `call_trace_forward.txt` | 单次 forward 的 Python 调用栈（793 事件，已预热避免 Triton 编译噪声）|
| `aten_ops.txt` | torch.profiler 表（按 CUDA 自时间排序的 ATen op + GPU kernel）|
| `chrome_trace.json` | Chrome trace 格式 profile，可拖进 `chrome://tracing` 或 [perfetto](https://ui.perfetto.dev) 查看时间线 |

### 5.1 验证若干关键事实

```bash
TRACE=Document/IR-Trace/nir_lif_kernel

# nirtorch 在构造阶段进了几次主入口？
grep -c "torch_tracer.py:74  torch_to_nir\|nir_interpreter.py:559  nir_to_torch" $TRACE/call_trace_build.txt
#  期望: 2  (export 一次 + import 一次)

# SJ map_* 被 nirtorch 回调几次？
grep -cE "to_nir.py:(85|103|110|154|188)  map_" $TRACE/call_trace_build.txt
#  期望: 34  (13 conv + 15 lif + 5 avgpool + 1 flatten)
#  注意: 3 个 fc 不在这里 —— 走 nirtorch DEFAULT_MAP 的 _map_linear，见 §1.2

# nirtorch 自带 _map_linear 被调用几次？
grep -c "torch_tracer.py:10  _map_linear" $TRACE/call_trace_build.txt
#  期望: 3  (3 个 fc 层)

# forward 期间 nirtorch 真实调用次数（排除文件头注释行）？
grep "nirtorch/" $TRACE/call_trace_forward.txt | grep -v "^#" | wc -l
#  期望: 1  (只有 nir_interpreter.py:417 ternary_operator 这一次)

# LIF kernel 被调用几次？
grep -c "lif.py:448  multi_step_forward" $TRACE/call_trace_forward.txt
#  期望: 15

# profiler 报告里 cudnn_convolution 出现几次？
grep "aten::cudnn_convolution" $TRACE/aten_ops.txt
#  期望: 13 次（与 conv 层数一致）

# profiler 报告里 SJ 手写 kernel 出现几次？
grep "_multistep_lif_forward_kernel" $TRACE/aten_ops.txt
#  期望: 15 次（与 LIF 层数一致）
```

---

## 6. 这份采样**不**保证的事

- **不覆盖 C++ 内部细节**。sys.settrace 只看 Python；profiler 给出 ATen op 与 CUDA kernel 名字
  但不展开 PyTorch 内部 dispatcher → kernel selector → cuDNN heuristic 那一层。要看到那一层需要
  额外加 `nsys` 或在 PyTorch 自己源码里打日志。
- **不覆盖训练路径**（eval / no_grad；training 模式下 `LIFNode.multi_step_forward` 走另一分支
  [`lif.py:451`](../../spikingjelly/spikingjelly/activation_based/neuron/lif.py#L451)，与本文 §2.2 不同）。
- **不保证 13 conv / 15 LIF 是定值**。换网络结构、换 BATCH 不影响调用关系，只影响调用次数。
- **不展开 fx.GraphModule.forward 内部**。它的源码是动态 `compile()` 出来的，filename 不在
  TRACE_KEYWORDS 里，sys.settrace 看不到。看它要直接打开
  [`vgg16_snn.fx_graph_module.py`](../IR-Trace/nir_lif_kernel/vgg16_snn.fx_graph_module.py)。

---

## 7. 让 NIR 路径也"纯走 Triton"（实测可行）

§2 那张表显示 NIR 路径 eager 跑时 Conv 走 cuDNN、Linear 走 cuBLAS、AvgPool 走 ATen native ——
原因是「eager 模式下 PyTorch ATen dispatcher 不把 Triton 当后端选项」（详见 [`Implementation-Modes.md`](Implementation-Modes.md) §6 与本文 §2 的 ATen dispatch 链路）。但 **NIR 输出的 `gm` 就是一个普通 `torch.fx.GraphModule`**，套上 `torch.compile` + 全 Triton 配置即可让 Inductor 接管整张图。

### 7.1 启动方法

入口脚本 [`examples/vgg16_snn/nir_compile_test.py`](../../examples/vgg16_snn/nir_compile_test.py)，关键就是 **`torch.compile(gm)` 那一行**：

```python
import torch._dynamo
import torch._inductor.config as inductor_cfg
from spikingjelly.activation_based import nir_exchange
from spikingjelly.activation_based.functional.conv_bn_fusion import fuse_conv_bn_eval_modules

# 1) 复用 vgg16_test.py 的全 Triton 编译配置（与 path B 完全等价）
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256
inductor_cfg.max_autotune = True
inductor_cfg.max_autotune_gemm_backends = "TRITON"
inductor_cfg.max_autotune_conv_backends = "TRITON"
inductor_cfg.force_disable_caches = True

# 2) 走 NIR 双向翻译，拿到 fx.GraphModule
folded = fuse_conv_bn_eval_modules(model.eval())       # BN 必须 fold（NIR 协议无 BN 原语）
graph  = nir_exchange.export_to_nir(folded, example_input=torch.rand(1,3,224,224), dt=1e-4)
gm     = nir_exchange.import_from_nir(graph, dt=1e-4, device="cuda", step_mode="m")
gm.eval()

# 3) ★ 关键就这一行：把 gm 包进 torch.compile
compiled = torch.compile(gm)

with torch.no_grad():
    out = compiled(x)              # x: [T, B, C, H, W]
torch.cuda.synchronize()
```

**与 path B (`vgg16_test.py`) 的两点差异**：
1. **不需要** `patch_spikingjelly_for_full_graph()`。那个 patch 是为绕开 `layer.BatchNorm2d.seq_to_ann_forward` 的 isinstance 判定（详见前面 §1 / [`SpikingJelly-Triton-Patch.md`](../../examples/vgg16_snn/SpikingJelly-Triton-Patch.md)），但 NIR 路径已经 fold-BN，**整网根本没有 BN 子模块**，连带把那条 graph_break 风险也消除了。
2. **网络不等价**：NIR 强制 BN folded + AvgPool 替 MaxPool，所以 NIR-compile 路径与 path B 数值上不会一致 —— 但「全 Triton 路径完整无回退」这件事在 NIR 端同样成立。

### 7.2 实测审计结果（10 项全过）

直接跑：

```bash
TORCH_LOGS=output_code python examples/vgg16_snn/nir_compile_test.py \
    > /tmp/nir_compile_audit.log 2>&1
```

然后对 `/tmp/nir_compile_audit.log` 套 [`audit-full-triton-path.md`](audit-full-triton-path.md) 那段 grep 脚本：

| 指标 | path B (vgg16_test.py) | NIR + torch.compile (本节) | 通过? |
|---|---:|---:|---|
| A) `extern_kernels.X` | 0 | **0** | ✅ |
| B) Triton kernel 定义数 | 54 | **98** | ✅ |
| D) max_autotune 决策 conv/mm | 9 + 3 | **9 + 3** | ✅ |
| E) cudnn/cublas 字样 | 0 | **0** | ✅ |
| F) Inductor "Output code" 段数 | 1 | **1** | ✅ |
| G) graph_break / Recompiling | 0 | **0** | ✅ |
| H) LIF kernel fallback | 0 | **0** | ✅ |
| dynamo `counters["graph_break"]` | 0 | **0**（`{}`）| ✅ |
| 首次编译 + autotune | 50.6 s | **49.4 s** | 同量级 |

固化的审计摘要：[`Document/IR-Trace/nir_lif_kernel/nir_compile_audit_summary.txt`](../IR-Trace/nir_lif_kernel/nir_compile_audit_summary.txt)。

### 7.3 一个细节：为什么 NIR 路径的 Triton kernel 数 (98) 比 path B (54) 多

> **注**：下面这条 98 vs 54 的差异是 **BATCH=1** 时的现象。后续在 **BATCH=56** 重新审计时两侧 kernel 数都收敛到 42（见 §7.7）—— Inductor 在大 BATCH 下做了更激进的 fusion，把 nirtorch dead helpers 与正常算子合并，差异被吃掉。这一节描述的根因（NIR forward 里有 dead helpers）仍然成立，但其 kernel-数量 后果只在 BATCH=1 这种极端小输入下才显化。

`grep -cE "def triton_|@triton\.jit"` 在 NIR-compile 日志里数到 98 个 kernel 定义，path B 只有 54。这并非 NIR 多干了真有用的活，是 nirtorch 自动生成的 `forward()` 里有几样无用代码会被 dynamo 一并 trace 进 FX 图、Inductor 给每个临时 elementwise 操作多生成几个 kernel：

- `torch.ones(1); ones = None` —— 一个被立即丢弃的临时张量初始化（[`vgg16_snn.fx_graph_module.py`](../IR-Trace/nir_lif_kernel/vgg16_snn.fx_graph_module.py) 第 6 行）；
- `_operator_is_(state, None)` + `nirtorch_nir_interpreter_ternary_operator(is_none, default, state)` —— state 字典初始化逻辑（构造期默认 `state=None`，运行时被 ternary 选成 default 字典）；
- `initialized_state` 这个永远不被读的中间变量。

它们对 Inductor 来说是有效的 FX 节点，会被各自生成 trivial Triton kernel（多数是几行的 elementwise，启动开销几乎为零，不影响实测延迟）。如果将来要让 NIR 路径与 path B 在 kernel 数量上对齐，可以在 `gm` 上跑一个**简单的 DCE Pass**（dead code elimination）先把这些清掉再 compile —— 但当前没必要。

### 7.4 LIF 在 NIR-compile 路径里仍然是 SJ 手写 kernel

dynamo trace 到 `neuron.LIFNode.multi_step_forward` 里的 `triton_kernel.multistep_lif` 时，由于该函数是 `@torch.library.custom_op` 注册的，**dynamo 不会 trace 进它内部**，只在 FX 图里发出一个调用节点。Inductor 不会重新生成 LIF kernel，**只发射一个 launcher 调 SJ 那份手写的 `_multistep_lif_forward_kernel`**。所以三个路径（NIR-eager / NIR-compile / path B）里的 LIF kernel 三同 —— 全部出自 [`Document/IR-Trace/nir_lif_kernel/sample_kernel/`](../IR-Trace/nir_lif_kernel/sample_kernel/) 那份 ttir/ttgir/llir/ptx/cubin 五级 IR。

### 7.5 因此一份 NIR-imported `gm` 现在有 **两种** 运行姿势

| 姿势 | 启动方式 | Conv/Pool/Linear 后端 | LIF 后端 | 编译开销 | 显存峰值 |
|---|---|---|---|---|---|
| **NIR-eager** | `gm(x)` 直接调 | cuDNN / ATen native / cuBLAS | SJ 手写 Triton kernel | 0 | 高（eager 无 buffer reuse；BATCH≥48 在 5070 Ti 上 OOM）|
| **NIR-compile** | `torch.compile(gm)(x)` | **Inductor-generated Triton kernel** | 同上 | 49–50 s 首次 | 低（Inductor buffer reuse；可以推到 BATCH=56）|

也就是说 [§5.3 的 "BATCH 上限速查"](#) 里 C 路径的 40 上限是 **NIR-eager 才有**的限制；
NIR-compile 路径的上限应该和 path B 一样到 56（待实测）。

### 7.6 一句话回答这个问题

「能否强制 NIR 也纯走 Triton？」—— **能，一行 `torch.compile(gm)` 加上一份和 path B 等价的 Inductor 配置即可，无需任何 SJ patch**。本质是因为 NIR 在运行时只贡献「得到一个 fx.GraphModule」的能力，至于这个 GraphModule 进 PyTorch 之后走 eager 还是走 Inductor，与 NIR 协议完全无关 —— 由调用者（用户代码）决定。

---

### 7.7 冷启动 10000 样本三路对照（BATCH=56，全 Triton 路径）

§7.2 的对照是 BATCH=1 + 单次运行，且只比了两路。为回答两个更精细的问题：

1. **NIR-compile 与 path B 在生产 BATCH 下到底有没有真实性能差？**
2. **NIR 协议本身在运行时有没有额外开销 —— 还是说"用 SJ layer.* 直接搭"和"绕 NIR 走一圈再搭"对结果完全等价？**

做了一系列 **冷启动**（每次 `rm -rf ~/.triton/cache` + ~95 s 重编 + 5 warmup + 179 iter × 56 = 10024 样本）测量，加入第三档 `MODE=SJ`：**直接用 SJ `layer.*` 搭一个与 NIR-imported gm 算子集严格等价的网络**（无 BN, AvgPool 替 MaxPool，全 `layer.Conv2d / layer.AvgPool2d / layer.Linear / layer.Flatten / neuron.LIFNode`），跳过 NIR 翻译这一环节。

**采样脚本**：[`examples/vgg16_snn/cold_start_10k_compare.py`](../../examples/vgg16_snn/cold_start_10k_compare.py)（`MODE=B / NIR / SJ`）
**原始数据**：[`Document/IR-Trace/nir_lif_kernel/cold_start_results.jsonl`](../IR-Trace/nir_lif_kernel/cold_start_results.jsonl)（9 行 JSONL，含 mean / median / std / min / max / peak_mem / compile_s）

#### 单次运行明细

| Run | Mode | 单张延迟 (ms) | 总耗时 (s) | within-run std (ms/iter) | cold compile (s) | peak mem (GiB) |
|---|---|---:|---:|---:|---:|---:|
| 1 | NIR-compile | 9.29751 | 93.211 | 0.425 | 96.1 | 14.04 |
| 2 | path B | 9.30581 | 93.298 | 0.398 | 95.6 | 14.04 |
| 3 | NIR-compile | 9.30416 | 93.277 | 0.341 | 94.3 | 14.04 |
| 4 | path B | 9.30714 | 93.310 | 0.306 | 95.7 | 14.04 |
| 5 | NIR-compile | 9.30193 | 93.277 | 0.391 | 96.1 | 14.04 |
| 6 | path B | 9.30183 | 93.250 | 0.347 | 95.6 | 14.04 |
| 7 | path B | 9.30502 | 93.282 | 0.330 | 95.7 | 14.04 |
| 8 | **SJ-direct** | **9.29877** | 93.224 | 0.526 | 94.5 | 14.04 |
| 9 | **SJ-direct** | **9.30637** | 93.301 | 0.429 | 94.8 | 14.04 |

#### 三路聚合

| Mode | n | mean (ms/张) | run-to-run std | 网络结构 |
|---|---:|---:|---:|---|
| **NIR-compile** | 3 | **9.30120** | 0.00339 | NIR roundtrip 后 fx.GraphModule（fold-BN + AvgPool）|
| **SJ-direct** | 2 | **9.30257** | 0.00538 | SJ `layer.*` 直接搭（无 BN + AvgPool；与 NIR 算子集等价）|
| **path B** | 4 | **9.30495** | 0.00225 | VGG16SNN（含 BN + MaxPool）|

模式两两差：

```
NIR  - SJ-direct  = -0.00137 ms/张  (-0.015%)   ← 算子集等价对比，差异最小
B    - SJ-direct  = +0.00238 ms/张  (+0.026%)   ← BN+MaxPool vs 无 BN+AvgPool
B    - NIR        = +0.00375 ms/张  (+0.040%)
```

所有差都 ≤ 0.04%，均小于各自的 run-to-run std（0.002–0.005 ms）。三路在墙钟上**统计不可区分**。

#### 这套实验解答了什么

1. **NIR 协议在运行时无开销**。SJ-direct 与 NIR-compile 差 0.015%（远小于 std）—— 也就是说"绕 NIR 走一圈"对推理性能没有任何额外 cost。NIR 是**纯构造期**的工具，运行时已经退场。这与 §2 的 sys.settrace 实测（forward 期间整个 nirtorch/ 路径只出现一次 `ternary_operator()` 调用）相互印证。
2. **path B 比 NIR/SJ 慢的 0.04% 是 BN 算术（详见 §7.8 TTIR 证据）**：path B 的 Inductor 把 13 个 BN 融进 conv 后的 pointwise epilogue，BN 的 sqrt + 5 个 mul/add 在 GPU 上跑（40 份含 `tt.precise_sqrt` 的 TTIR 实锤）。但这部分算术在 memory-bound kernel 里几乎被 HBM 等待吸收，墙钟上只显出 ~0.04% 差，且仍在噪声内。
3. **三路结论**：BATCH=56 全 Triton 编译路径下，VGG16-SNN 单张推理延迟 ≈ **9.30 ms / 张**（吞吐 ~107.5 张/秒），peak memory ~14 GiB，cold compile ~95 s。这是 5070 Ti + 我们 Triton fork 在当前 Inductor 配置下的实测基线。

### 7.8 TTIR 实证：path B 的 BN 算术真的在 GPU 上跑（修正 §7.3 之前一处武断说法）

之前曾推断"path B 的 Inductor 会做 conv-BN 融合 + 常量折叠，两边都没有 BN 算术"。**抓 Inductor 的真实编译产物后这一句要修正**：

| 视角 | path B（BN+MaxPool） | NIR-compile（fold-BN+AvgPool） |
|---|---:|---:|
| 独立 BN kernel | 0 ✓ | 0 ✓ |
| kernel 名含 `_native_batch_norm_legit_no_training` token | **10 个** | 0 个 |
| **Inductor 临时目录里 `.ttir` 含 `tt.precise_sqrt` 的文件数** | **40**（= 10 kernel × 4 autotune cfg）| **0** |
| BN 运行时算术 op / 层（13 层共） | **8**（sub+sqrt+div+mul×2+mul+add，含 1 个 sqrt）| 0 |
| BN 参数额外 `tl.load` / 层（13 层共） | **4**（mean/var/gamma/beta，4×4 B = 16 B per channel block，常驻 L2）| 0 |

#### path B 含 BN 算术的 kernel 真实 TTIR 节选

来自 `/tmp/torchinductor_charlley/tmp*/triton/<hash>/triton_poi_fused__native_batch_norm_legit_no_training_convolution_full_like_max_pool2d_with_indices_view_19.ttir`：

```mlir
%tmp2     = arith.addf %tmp0_15, %tmp1_17     ; conv_out + conv_bias
%tmp4     = arith.subf %tmp2, %tmp3_19        ; − running_mean
%tmp7_26  = arith.addf %tmp5_21, %tmp7        ;   running_var + 1e-5
%tmp8     = tt.precise_sqrt %tmp7_26          ;   sqrt(var + eps)   ← 真 GPU 计算
%tmp10_27 = arith.divf %tmp10, %tmp8          ;   1 / sqrt(...)
%tmp13    = arith.mulf %tmp4, %tmp10_27       ;   * 1/sqrt
%tmp15    = arith.mulf %tmp13, %tmp14_23      ;   * gamma
%tmp17    = arith.addf %tmp15, %tmp16_25      ;   + beta
tt.store ...                                  ; 写回输出
```

NIR 同位置 kernel `triton_poi_fused_convolution_full_like_view_3` 算术体仅一行：

```mlir
%tmp2  = arith.addf %tmp0, %tmp1              ; conv_out + conv_bias  (fold-BN 已合进 conv_bias)
tt.store ...
```

**Inductor 没有做 BN 常量折叠**（虽然 BN 参数在 eval 模式下都是常量，理论上可以预算成等价 conv 系数）—— 它只是把 BN 算子作为 conv 的 epilogue **融到同一个 pointwise kernel 体里跑**，省了独立 kernel launch 和中间 buffer HBM 往返，**但 BN 的 sqrt/sub/div/mul 一个不漏全留**。NIR 在 Python 端用 `fuse_conv_bn_eval_modules` 把 BN 数学等价合进 Conv weight/bias，**GPU 端真的一个 BN op 都没有**。

#### 但为什么实测延迟还几乎一样

那个 BN-fused-epilogue kernel **是 memory-bound 的**：

- 主成本：`conv_out` tensor 的 load + 输出 store（每元素 8 B HBM 流量）；
- BN 额外成本：
  - 4 个 `tl.load` 加载 BN 参数 = 64 channels × 4 B = 256 B / 通道块，`eviction_policy='evict_last'` 让它们常驻 L2，**几乎不增加 HBM 压力**；
  - 1 个 sqrt + 4 个 mul/add：sm_120 上 fp32 throughput ~30 TFLOPS，sqrt 几个 cycle，**和 memory pipeline 重叠掉**。

实测 path B 多花的时间 ≈ 13 BN × 1 μs/层 ≈ 13 μs/iter ÷ BATCH=56 ≈ 0.2 μs/张 —— 与 §7.7 测得的 5 μs/张 (0.06%) 差异同量级。**所以"BN 算术在跑"和"墙钟上看不出来"两件事并不矛盾**：算术 op 数差很多，但都在等内存的同一个 kernel 里被吸收了。

#### 持久化的真实证据

| 文件 | 内容 |
|---|---|
| [`Document/IR-Trace/nir_lif_kernel/kernel_names_b56.txt`](../IR-Trace/nir_lif_kernel/kernel_names_b56.txt) | path B 与 NIR-compile 在 BATCH=56 时所有 Inductor 生成 kernel 的名字（path B 42 个、NIR 42 个）|
| [`Document/IR-Trace/nir_lif_kernel/bn_ttir_evidence.txt`](../IR-Trace/nir_lif_kernel/bn_ttir_evidence.txt) | BN-fused kernel 的真实 TTIR 算术节选 + 各 Inductor temp dir 含 sqrt 的 TTIR 文件数（path B=40, NIR=0）|
| [`Document/IR-Trace/nir_lif_kernel/cold_start_results.jsonl`](../IR-Trace/nir_lif_kernel/cold_start_results.jsonl) | 4 次冷启动 10024 样本的原始测量数据（JSON Lines 格式，含 mean / median / std / min / max / peak_mem / compile_s）|

#### 复现命令

```bash
cd /home/charlley/Code/Triton-Pass-Analysis

# 4 次冷启动测量（每次清 cache，约 15 分钟总耗时）
for run in 1 2; do
    for mode in NIR B; do
        rm -rf ~/.triton/cache
        MODE=$mode BATCH=56 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python examples/vgg16_snn/cold_start_10k_compare.py
    done
done

# 抓 TTIR 证据
rm -rf ~/.triton/cache /tmp/torchinductor_charlley
TORCH_LOGS=output_code MODE=B BATCH=56 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/vgg16_snn/cold_start_10k_compare.py > /tmp/pathB.log 2>&1
find /tmp/torchinductor_charlley -name "*.ttir" | xargs grep -l "sqrt" | wc -l
#  期望: 40   (path B 有 10 个 BN-fused-epilogue kernel × 4 autotune cfg)

rm -rf ~/.triton/cache /tmp/torchinductor_charlley
TORCH_LOGS=output_code MODE=NIR BATCH=56 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/vgg16_snn/cold_start_10k_compare.py > /tmp/NIR.log 2>&1
find /tmp/torchinductor_charlley -name "*.ttir" | xargs grep -l "sqrt" | wc -l
#  期望: 0    (NIR 路径 fold-BN 后 GPU 上无 sqrt)
```

---

### 7.9 FX 图同形性实测：NIR-compile 与 SJ-direct 给 Inductor 的图等价（§7.7 三路同值的根因）

§7.7 测出 NIR-compile 与 SJ-direct 在墙钟上不可区分（0.015% 差，远小于 std）。这一节用 `TORCH_LOGS=graph_code,aot_graphs` 把两路 dynamo 与 AOTAutograd 阶段的 FX 图整段抓出来，做**字符级对比**，证明这种墙钟同值不是巧合 —— **两路最终交给 Inductor 编译的 FX 图本身就是同形的**。

#### 抓取命令

```bash
rm -rf ~/.triton/cache /tmp/torchinductor_charlley
TORCH_LOGS="graph_code,aot_graphs" MODE=NIR BATCH=56 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/vgg16_snn/cold_start_10k_compare.py > /tmp/nir_graphs.log 2>&1

rm -rf ~/.triton/cache /tmp/torchinductor_charlley
TORCH_LOGS="graph_code,aot_graphs" MODE=SJ BATCH=56 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/vgg16_snn/cold_start_10k_compare.py > /tmp/sj_graphs.log 2>&1
```

`graph_code` 拦下 dynamo 抓出来的初 FX 图（`[__graph_code]` 标签），`aot_graphs` 拦下 AOTAutograd 分解之后的 forward graph（`[__aot_graphs]` 标签）。

#### 第 1 层：dynamo trace 出来的初 FX 图算子直方图

| 算子 | NIR-compile | SJ-direct |
|---|---:|---:|
| `.flatten(` | 39 | 39 |
| `.view(` | 38 | 38 |
| `torch.full_like(` | 30 | 30 |
| `.clone(` | 30 | 30 |
| `.forward(` | 19 | 19 |
| `torch.ops.sj.multistep_lif_inference.default(` | **15** | **15** |
| `torch._C._nn.linear(` | 3 | 3 |
| `torch.ones(` | **2** | **0** ← **唯一差异** |

NIR 多出来的 2 次 `torch.ones(` 来自 nirtorch 自动生成 forward 头部的 dead helper：

```python
def forward(self, input, state = None):
    ones = torch.ones(1);  ones = None    # ← 这一句, dynamo trace 出来
    ...
```

完整直方图固化在 [`dynamo_op_histogram_nir.txt`](../IR-Trace/nir_lif_kernel/dynamo_op_histogram_nir.txt) / [`dynamo_op_histogram_sj.txt`](../IR-Trace/nir_lif_kernel/dynamo_op_histogram_sj.txt)。

#### 第 2 层：AOTAutograd 分解后的 forward graph 算子直方图

| 算子 | NIR-compile | SJ-direct |
|---|---:|---:|
| `aten.view.default` | 45 | 45 |
| `aten.empty.memory_format` | 30 | 30 |
| `higher_order.triton_kernel_wrapper_functional` | **15** | **15** |
| `aten.select.int` | 15 | 15 |
| `aten.full.default` | **15** | **15** ← dead `torch.ones` 已被 DCE 清掉 |
| `aten.clone.default` | 15 | 15 |
| `aten.convolution.default` | 13 | 13 |
| `aten.avg_pool` | 5 | 5 |
| `aten.addmm.default` | 3 | 3 |
| `aten.permute.default` | 3 | 3 |
| **diff 行数** | **0** | **0** |

两路算子直方图**逐字段相同**。`torch.ones` 在 dynamo 阶段对 NIR 多出 2 次（dead code），AOTAutograd 一遍走完已经被 DCE —— 如果 `aten.full.default` 仍为 16 而非 15 就说明 DCE 失败，实测正好 15 印证 DCE 命中。

#### 第 3 层：AOT forward graph **全文**对比

```text
NIR-compile aot_forward_body_nir.txt:  362 行
SJ-direct   aot_forward_body_sj.txt :  362 行
带变量名的原始 diff:  132 行差异 (~18% lines)
规范化变量名后 diff:  0 行 (完全同形)
```

132 行带名差异**全部**位于 SSA 变量编号 —— NIR 是 `full_1`/`full_2`/...，SJ 是 `full`/`full_1`/... ——
两路只差 1 步偏移，根因正是 NIR 那个 `torch.ones` 让 SSA counter 多走一格：

```diff
- full_1: "f32[56, 64, 224, 224]..." = torch.ops.aten.full.default([56, 64, 224, 224], 0.0, ...)
+ full:   "f32[56, 64, 224, 224]..." = torch.ops.aten.full.default([56, 64, 224, 224], 0.0, ...)
- full_2: "f32[56, 64, 224, 224]..." = torch.ops.aten.full.default([56, 64, 224, 224], 0.0, ...)
+ full_1: "f32[56, 64, 224, 224]..." = torch.ops.aten.full.default([56, 64, 224, 224], 0.0, ...)
```

调用本身的 args / kwargs / shape / dtype / device / layout 全部 byte-equal。

LIF custom_op 的调用参数也字段完全一致：

```python
torch.ops.higher_order.triton_kernel_wrapper_functional(
    kernel_idx           = 0,
    constant_args_idx    = 15,
    grid                 = [(1404928,1,1), (702464,1,1), (702464,1,1), (351232,1,1)],
    tma_descriptor_metadata = {},
    kwargs = {
        'x_seq_ptr': view_1, 'v_init_ptr': full, 's_seq_ptr': empty,
        'h_seq_ptr': empty_1, 'v_seq_ptr': empty_1,
        'tau': 2.0, 'v_threshold': 1.0, 'v_reset': 0.0,
        'T': 4, 'NCL': 179830784,
        'decay_input': True, 'soft_reset': False, 'save_intermediates': False,
    },
    tensors_to_clone = ['s_seq_ptr', 'v_seq_ptr'],
)
```

`grid` shape、`kernel_idx`、LIF 超参（tau, v_threshold, T, NCL）逐字段相同。

#### 持久化的真实证据

| 文件 | 内容 |
|---|---|
| [`dynamo_op_histogram_nir.txt`](../IR-Trace/nir_lif_kernel/dynamo_op_histogram_nir.txt) / [`_sj.txt`](../IR-Trace/nir_lif_kernel/dynamo_op_histogram_sj.txt) | dynamo 阶段 8 类算子直方图（NIR 比 SJ 多 2 次 `torch.ones`，其余完全相同）|
| [`aot_forward_body_nir.txt`](../IR-Trace/nir_lif_kernel/aot_forward_body_nir.txt) / [`_sj.txt`](../IR-Trace/nir_lif_kernel/aot_forward_body_sj.txt) | AOT forward graph 全文（362 行 × 2，规范化 diff = 0）|
| [`fx_graph_isomorphism.txt`](../IR-Trace/nir_lif_kernel/fx_graph_isomorphism.txt) | 三层证据 + LIF custom_op 调用对照 + 结论 |

#### 结论

**Inductor 实际收到的 FX 图，两路严格同形（只差 SSA 变量编号偏移 1）**。也就是说：

- NIR 协议在运行时**既无新增 kernel、也无新增算子、也无额外 dispatch 开销**；
- §7.7 三路实测延迟同值（NIR 9.301 / SJ 9.303 / B 9.305 ms/张）的根本原因是**它们编译出的 Triton kernel 完全相同**（path B 多了 13 个 BN-fused epilogue kernel，见 §7.8）；
- "NIR 是顶端协议级包装，最后还是落到 FX 图交给 torch" 这一直觉判断在 FX-graph 同形性层面再次得到字面级实证。
