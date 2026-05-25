# 子模块版 SpikingJelly 与本仓库定制版 Triton 的兼容性补丁

记录在 SpikingJelly 子模块（`spikingjelly/`，editable 安装到 conda 环境）中所做的两处
本地修改、它们解决的具体问题、以及如何在新克隆时复现。

## 1. 触发场景

仓库的 `spikingjelly/` 子模块（含 PSN 等较新内容的版本）在 2026-05-25 被以
editable 模式安装进 `triton-dev-cuda131` conda 环境（site-packages 里出现
`__editable__.spikingjelly-0.0.0.0.15.pth`）。安装后，所有 `import spikingjelly`
都解析到本仓库子模块版本——而该版本在 `spikingjelly/activation_based/triton_kernel/`
下自带一套手写 Triton kernel，会在以下路径被自动启用：

- `neuron.LIFNode.multi_step_forward()`（eval 模式下 CUDA 上首选）
- `neuron.IFNode.multi_step_forward()`
- `neuron.ParametricLIFNode.multi_step_forward()`
- FlexSN 系列

跑 `benchmark_inference.py COMPILE=1`（torch.compile + 全 Triton 路径）时立刻撞到：

```
Name=_multistep_lif_forward_kernel
triton.compiler.errors.CompilationError:
  AttributeError("'dtype' object has no attribute 'element_ty'")
```

## 2. 根因：块指针 dtype 的 API 层数不一致

报错位置是一个共享小工具 `convert_and_store`：

```python
@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # 把不支持隐式类型转换的 block pointer 的 value 手动转换后再 store
    value = value.to(pointer.dtype.element_ty.element_ty)   # ← 这里
    tl.store(pointer, value, boundary_check=boundary_check)
```

`pointer` 是 `tl.make_block_ptr(...)` 的产物。对块指针的 dtype 访问元素类型，存在
两套上游 API：

| 上游 Triton 版本 | `pointer.dtype.element_ty` 给的是 | 取标量元素类型需要 |
|---|---|---|
| 早期 | 块类型（`block_type(scalar)`）| `.element_ty.element_ty`（两层）|
| ≥ 3.7（块指针聚合化、`.dtype` 直接转发到底层标量指针）| 标量元素类型本身 | `.element_ty`（一层）|

本仓库 `triton/` 子模块基于上游 `5d69e1cf4` 切出的 `snn-optimization` 分支，
其块指针 dtype 是**单层**结构——`pointer.dtype.element_ty` 已经就是标量类型。
SpikingJelly 原始的两层访问 `pointer.dtype.element_ty.element_ty` 第二个
`.element_ty` 落在标量 dtype 上 → `AttributeError`。

> 注：SpikingJelly 的 `convert_and_store` 是被全部多步 neuron kernel 共用的工具函数，
> 这一处不修，LIF/IF/PLIF 在本仓库 Triton 上都无法编译。

## 3. 改了哪两处

### `spikingjelly/activation_based/triton_kernel/triton_utils.py`

`convert_and_store` 的核心实现，所有 `neuron_kernel/*.py` 都靠
`from ..triton_utils import convert_and_store` 用它。

```diff
 def convert_and_store(pointer, value, boundary_check):
     # For block pointers created by tl.make_block_pointer(),
     # implicit type casting is not supported when calling tl.store().
     # This function manually converts dtype and then stores the data.
-    value = value.to(pointer.dtype.element_ty.element_ty)
+    # In Triton >= 3.7 (block-pointer became an aggregate that forwards .dtype
+    # to its scalar base pointer), ``pointer.dtype.element_ty`` is already the
+    # scalar element type, so only one layer of ``.element_ty`` is needed.
+    value = value.to(pointer.dtype.element_ty)
     tl.store(pointer, value, boundary_check=boundary_check)
```

### `spikingjelly/activation_based/triton_kernel/flexsn/template.py`

`flexsn` 模板字符串里独立的一份 `convert_and_store` 拷贝（在生成 kernel 时
被实例化为最终 kernel 的一部分）。

```diff
 def convert_and_store(pointer, value, boundary_check):
     # ...
-    value = value.to(pointer.dtype.element_ty.element_ty)
+    value = value.to(pointer.dtype.element_ty)
     tl.store(pointer, value, boundary_check=boundary_check)
```

两处改动**完全等价**——都是把两层 `.element_ty` 简化为一层。

## 4. 验证

在已训练 T=4 VGG16-SNN（`vgg16_snn_imagenet.pth`，46.6% top-1）上跑
`COMPILE=1 python benchmark_inference.py 10000 50`：

| | 编译 + 首次前向 | 10000 样本前向 | 单张推理 | top-1 |
|---|---|---|---|---|
| 修补前 | 编译期崩溃 | — | — | — |
| 强制 torch fallback（绕开 Triton kernel）| 201.1 s | 139.3 s | 13.93 ms | 49.28% |
| **修补后（spikingjelly 手写 LIF Triton kernel 正常跑）** | **112.3 s** | **94.1 s** | **9.41 ms** | **49.29%** |

- LIF 的手写 Triton kernel 现在正常编译执行。
- top-1 与 eager（49.29%）完全一致，数值正确性确认。
- 单张推理 9.41 ms 比早先用旧版 spikingjelly 时的 9.65 ms 还略快——手写 LIF
  kernel 比 Inductor 自动生成的 BN+LIF 融合 kernel 略优。
- 编译时间从 ~340 s 降到 112 s（-67%）：手写 kernel 的 `@triton.autotune` 搜索
  空间小（仅 4 种 num_warps × BLOCK_NCL 组合），不像 Inductor 对 conv 那样跑大
  规模 max_autotune。

注：torch.compile 全 Triton 路径仍比 eager（cuDNN, 7.41 ms）慢约 27%——Triton 通
用 conv 模板打不过 cuDNN，这条结论在
[`Document/IR-Trace/Optimization-Insights.md`](../../Document/IR-Trace/Optimization-Insights.md)
里量化过，与本补丁无关。

## 5. 注意事项

- **方向特定**。这个补丁让 SpikingJelly 适配「块指针 dtype 单层」的 Triton API
  （本仓库定制版与上游 ≥ 3.7 均如此）。若日后 `triton/` 子模块被回切到老式块指针
  API（双层），该 kernel 又会报对称的错误，需要把两层加回来。
- **可移植写法**（兼容两种 API）——本仓库当前没采用，留作日后参考：
  ```python
  ety = pointer.dtype.element_ty
  if hasattr(ety, "element_ty"):
      ety = ety.element_ty
  value = value.to(ety)
  ```
  暂取「与 SpikingJelly 上游修复方向一致的最简形态」，以便后续追上游变更时易于
  跟踪 / 合并。
- **未向 SpikingJelly 上游提 PR**，亦不在本仓库主 commit 历史中——属于子模块工作树
  的本地修改（`git status` 在子模块内显示，主仓库 `git status` 显示
  `M spikingjelly` 子模块脏）。

## 6. 复现（新克隆 / 子模块重置后）

```bash
cd <repo>/spikingjelly/spikingjelly/activation_based/triton_kernel
sed -i 's/pointer\.dtype\.element_ty\.element_ty/pointer.dtype.element_ty/' \
       triton_utils.py flexsn/template.py
```

editable 安装无需重装，改动立即生效。

## 7. 启动 benchmark：吞吐 / 延迟两种姿势

补丁修好后，`benchmark_inference.py` 在 Triton 路径下有两种典型跑法。**两者都用补丁后的
spikingjelly 手写 LIF Triton kernel；区别只在 BATCH。**

### 7.1 吞吐模式（最大化 GPU 利用率）

把 BATCH 推到显存能装下的最大值——本卡 RTX 5070 Ti（16 GiB）上 **BATCH=56 是实际最大**。
BATCH=64 起，LIF kernel 第一层的 `[T,N,64,224,224]` 输出 buffer 分配（约 3 GiB）
让总显存超 16 GiB → OOM。建议同时打开 `expandable_segments` 减少分片浪费：

```bash
cd examples/vgg16_snn
COMPILE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python benchmark_inference.py 10000 56
```

实测（10024 张样本，DataLoader 凑够整 batch）：

| 项 | 值 |
|---|---|
| 编译 + 首次前向（一次性） | 120.8 s |
| 纯 GPU 前向 | 94.10 s |
| 单次前向调用（batch=56） | 526 ms |
| **单张样本平均** | **9.39 ms** |
| 吞吐 | 107 张/秒 |
| top-1（健全性检查） | 49.20% |

此时 GPU 等效批为 `T×N = 4×56 = 224`，VGG16 已接近饱和；和 BATCH=50 的 9.41 ms/张
基本没差（已在噪声内），说明本配置下吞吐已经到顶。

### 7.2 延迟模式（串行单条推理）

`BATCH=1` 时每次 forward 就是一张样本，单次耗时直接读为「单条请求的端到端推理延迟」：

```bash
cd examples/vgg16_snn
COMPILE=1 python benchmark_inference.py 10000 1
```

实测（10000 次单样本推理）：

| 项 | 值 |
|---|---|
| 编译 + 首次前向（一次性） | 52.7 s |
| 纯 GPU 前向 | 124.01 s |
| **单次推理平均** | **12.40 ms** |
| 吞吐 | 81 张/秒 |
| top-1（健全性检查） | 49.29% |

比吞吐模式折算到单张慢约 32%——`T×N = 4` 远未填满 GPU，kernel 启动开销占比更高、
Tensor Core 也喂不饱。延迟模式编译反而更快（52.7 vs 120.8 s）：max_autotune 在
小 M 维度下要试的 Triton conv 配置少得多。

### 7.3 何时选哪种

| 场景 | 模式 | 关心的指标 |
|---|---|---|
| 离线批量推理 / 跑数据集 / 服务端打满显卡 | 吞吐（最大 BATCH） | 张 / 秒 |
| 在线单条请求 / 实时机器人 / 自动驾驶单帧响应 | 延迟（BATCH=1） | ms / 次 |

两个数都正确、都来自补丁后的 Triton 路径，只是回答不同的问题。
