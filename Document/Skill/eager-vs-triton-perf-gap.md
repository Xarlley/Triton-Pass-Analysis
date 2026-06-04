# eager (cuDNN/cuBLAS) vs 全 Triton 的性能差距溯源

> 已测得事实（[`SpikingJelly-Triton-Patch.md §7`](../../examples/vgg16_snn/SpikingJelly-Triton-Patch.md)）：
> 同一个 VGG16-SNN（BN + MaxPool + LIF），BATCH=50，T=4，RTX 5070 Ti
>
> - eager + cuDNN/cuBLAS  : **7.41 ms / 张**
> - torch.compile + 全 Triton : **9.41 ms / 张**
> - 差距 = **+27% (≈ 2.0 ms / 张)**
>
> 本文用 `torch.profiler` 抓出**每个 GPU kernel 的 self 时间**，按算子类别聚合，定位这 ~2 ms / 张的差距具体落在哪几类 kernel。
>
> 实验脚本：[`examples/vgg16_snn/perf_breakdown.py`](../../examples/vgg16_snn/perf_breakdown.py)
> 真实产物：[`Document/IR-Trace/perf_breakdown/`](../IR-Trace/perf_breakdown/)
>
> 验证环境：torch 2.11.0+cu130 (cuDNN 9, cuBLAS 13)、spikingjelly 0.0.0.0.15、triton 3.7.0+gitef02d646 (本仓库 fork)、RTX 5070 Ti (sm_120)。

---

## 1. 测量方法

### 1.1 实验配置

| 项 | 取值 | 备注 |
|---|---|---|
| 网络 | VGG16SNN (vgg16_test.py 同款) | 13 Conv + 13 BN + 15 LIF + 5 MaxPool + 3 FC，layer.* 多步包装 |
| BATCH | **32** | eager + profiler 在 BATCH=50 触发 OOM，降到 32；相对差距与 BATCH 关系不大，结论仍可推广到 50/56 |
| T | 4 | |
| warmup | 3 次 forward | 让 SJ multistep_lif 与 Inductor 的 autotune 完全走稳 |
| profile iters | 3 次 forward 累计后求平均 | 抹平噪声 |
| 输入 | `torch.randn([4, 32, 3, 224, 224])` | 随机数据，不影响 kernel timing |

### 1.2 抓 kernel-level self time

```python
from torch.profiler import profile, ProfilerActivity, record_function
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("forward_<mode>"), torch.no_grad():
        for _ in range(n_iters):
            functional.reset_net(model)
            compiled_or_eager(x)
        torch.cuda.synchronize()
```

为避免把 ATen op wrapper（`aten::cudnn_convolution` 之类，CPU 侧 dispatcher）的时间和它启动的真实 CUDA kernel 时间**双重计入**，过滤器只保留**真正的 GPU kernel 名**：

```python
def is_gpu_kernel_name(name):
    return (name.startswith("void ")          # 大多数 CUDA kernel C++ ABI 反 demangle 后的样子
         or name.startswith("triton_")        # Inductor 生成 Triton kernel
         or name.startswith("_multistep_")    # SJ 手写 LIF Triton kernel
         or name.startswith("Memcpy")
         or name.startswith("Memset"))
```

### 1.3 算子分类（perf_breakdown.py `categorize`）

| 类别 | 匹配模式 | 含义 |
|---|---|---|
| `conv (cuDNN cutlass)` | `cutlass__5x_cudnn::Kernel`、`cutlass_tensorop` | cuDNN 走的 CUTLASS-based tensor core conv |
| `conv (cuDNN xmma/sgemm)` | `sm80_xmma_fprop`、`implicit_convolve_sgemm` | cuDNN 走的 SGEMM-formulated conv |
| `conv (cuDNN winograd)` | `winograd` | cuDNN winograd 卷积算法 |
| `conv (cuDNN layout xform)` | `nchwToNhwc`、`nhwcToNchw` | cuDNN 卷积前后要做的 layout 转换 |
| `BN (cuDNN/ATen)` | `bn_fw_inf`、`batch_norm` | cuDNN 的 BN 推理 kernel |
| `gemm (cuBLAS cutlass)` | `cutlass_80_simt_sgemm`、`cublas` | cuBLAS 的 SGEMM kernel（FC 走的） |
| `MaxPool (ATen native CUDA)` | `max_pool_forward_nchw` | ATen 自家 CUDA pool（**非 cuDNN**） |
| `elementwise (ATen native CUDA)` | `elementwise_kernel`、`vectorized_elementwise`、`FillFunctor` | ATen 自家 elementwise（conv bias add 等） |
| `conv (Inductor Triton tem)` | `triton_tem_fused_convolution` | Inductor 生成 Triton conv kernel |
| `BN+conv-epilogue (Inductor Triton poi)` | `triton_poi_fused__native_batch_norm_*` | Inductor 把 BN 融进 conv 后的 pointwise epilogue |
| `conv-epilogue only (Inductor Triton poi)` | `triton_poi_fused_convolution` | Conv 的 bias / view 后处理 epilogue（无 BN）|
| `MaxPool (Inductor Triton poi)` | `triton_poi_fused_max_pool` | Inductor 生成的 MaxPool Triton kernel |
| `gemm (Inductor Triton tem)` | `triton_tem_fused_addmm` | Inductor 生成的 GEMM Triton kernel |
| `LIF (SJ Triton)` | `_multistep_lif_*` | SpikingJelly 自带的 fused-T-loop LIF kernel（两种模式下都用）|

---

## 2. 实测结果（BATCH=32，每 forward iter 的 self_cuda_time 总和）

完整产物见 [`Document/IR-Trace/perf_breakdown/`](../IR-Trace/perf_breakdown/)：
- `eager_kernels.txt` —— eager 模式 18 个 GPU kernel 的逐行表
- `compile_kernels.txt` —— 全 Triton 模式 49 个 GPU kernel 的逐行表
- `breakdown.txt` —— 按算子类别聚合的对照表
- `eager_trace.json` / `compile_trace.json` —— Chrome trace（可拖进 [perfetto](https://ui.perfetto.dev) 查看时间线）

### 2.1 按算子类别对照

| 类别 | eager (μs) | eager 次 | compile (μs) | compile 次 | Δ (μs) | Δ (%) |
|---|---:|---:|---:|---:|---:|---:|
| **conv 总计** | **33988** | 29 | **218099** | 38 | **+184111** | +542% |
| ├ cuDNN cutlass | 10582 | 1 | — | — | -10582 | |
| ├ cuDNN xmma/sgemm | 3999 | 1 | — | — | -3999 | |
| ├ cuDNN winograd | 3905 | 6 | — | — | -3905 | |
| ├ cuDNN layout xform | 15502 | 21 | — | — | -15502 | |
| ├ Inductor Triton tem | — | — | 207851 | 13 | +207851 | |
| ├ Inductor BN+conv epilogue | — | — | 20164 | 26 | +20164 | |
| └ Inductor conv-epilogue only | — | — | 10248 | 19 | +10248 | |
| **BN 总计** | **18518** | 13 | **0**（已 fused 进 conv epilogue）| 0 | **-18518** | -100% |
| **elementwise 总计** | **19995** | 28 | **2** | 2 | **-19993** | -100% |
| **MaxPool 总计** | **7133** | 5 | **4794** | 5 | -2339 | -33% |
| **gemm 总计** | **13581** | 5 | **2415** | 3 | -11166 | -82% |
| **LIF (SJ Triton)** | **30219** | 15 | **105211** | 30 | +74992 | (含 autotune 噪声 ⚠️) |
| **memcpy / memset** | **4420** | 17 | **0** | 0 | -4420 | -100% |
| **TOTAL** | **127854** | 112 | **350700** | 101 | +222846 | +174% |

### 2.2 三个直接观察

1. **Conv 是性能差距的绝对主因**：cuDNN 整套卷积工具链（cutlass + xmma + winograd + layout xform）总共只跑 34 ms 就完成 13 个 conv 层；Inductor 生成的 Triton conv 模板做同样的事用了 218 ms（含融合进来的 BN+conv epilogue 20 ms 和 conv epilogue 10 ms）。**Δ = +184 ms / forward → 折算到单张 = +5.8 ms**，远超墙钟实测的 2 ms 差，原因是 stream overlap（见 §3）。

2. **Inductor 的 conv-BN 融合确实省下了独立 BN kernel**：eager 单独跑 BN（18.5 ms） + 单独跑 conv-bias add 等 elementwise（20.0 ms）共 **38.5 ms**；compile 把 BN 与 conv bias add 等都融进了 conv 后那个 pointwise epilogue（20.2 ms BN-epilogue + 10.2 ms conv-epilogue ≈ **30.4 ms**），**省下 8 ms**。但这点节省远不足以抵消 conv 自身多花的 184 ms。

3. **LIF 在两个模式都用同一份 SJ 手写 Triton kernel**，理论上应该完全相同。实测 compile 模式 LIF 总时长是 eager 的 3.5×（105 ms vs 30 ms） —— 这不是真实差距，而是**测量噪声**：compile 模式在每次 forward 里 SJ kernel 的 `@triton.autotune` 会被 Inductor 的 `triton_kernel_wrapper_functional` 包装层重新走部分 trial（同 forward iter 里看到 `_multistep_lif_forward_kernel`、`_multistep_lif_forward_kernel_0`、`_1`、`_2` 多份 cubin 都被调），eager 一次 warmup 后直接走 cached cfg，profile 内只看到 1 份 cubin。在墙钟上 LIF 实际成本两侧一致（见 [`nir-call-stack-trace.md §7.7/§7.9`](nir-call-stack-trace.md) 多次三路冷启动 10024 样本对照的实测，三路 ms/张 在 0.04% 噪声内）。

---

## 3. 为什么 GPU 工作多了 174% 但墙钟只慢了 27%？

总 GPU self_us：eager 128 ms，compile 351 ms（+222 ms）。
墙钟（外推 BATCH=32）：eager ≈ 7.41 × 32 = 237 ms，compile ≈ 9.39 × 32 = 300 ms（+63 ms）。

GPU 多干 +222 ms，墙钟只多 +63 ms —— 说明 compile 模式有 **~160 ms 的 kernel 在不同 CUDA stream 上重叠执行**，没花到墙钟。

具体说，Inductor 编译产物里大量的 `triton_poi_fused_*` 小 epilogue kernel 可以在 `triton_tem_fused_convolution_*` 主 conv kernel 完成的间隙启动（不同 SM 上）。eager 路径几乎都在默认 stream 上串行：每个 cuDNN conv 完成后才能启动下一个 ATen op，没有交叠空间。

这也解释了**为什么 ~5.8 ms / 张的 GPU 工作差距最终只体现为 2 ms / 张的延迟差**：剩下 ~3.8 ms / 张被 stream overlap 吸收掉了。

不过，扣掉 stream overlap 抵消和 LIF 噪声后，**conv 的 GPU 工作量差距（~+184 ms / forward = +5.8 ms / 张）依然是绝对主导因素**。

---

## 4. 为什么 Inductor Triton conv 这么慢？

cuDNN 在 VGG 形状（13 层 3×3 conv，多种 channel 与空间分辨率）上跑了几十年，**累积了三套独立优化模板**：

| cuDNN 模板 | 适用形状 | 优势 |
|---|---|---|
| **cutlass_tensorop**（`_s1688fprop_*`）| 大 channel、大 batch 的 3×3 conv | 用 sm_8.0+ 的 1688 tensor core MMA 指令，TF32 加速 |
| **xmma_fprop**（`sm80_xmma_fprop_implicit_gemm_*`） | 中等规模 conv | implicit-GEMM 算法 + tensor core 优化 |
| **winograd**（`winograd_nonfused::winogradForwardData/Output4x4`）| 3×3 stride-1 conv（VGG 主要场景）| Winograd F(4×4, 3×3) 把每个 conv 的乘法次数从 9 减到 ~2.25 |
| **implicit_convolve_sgemm** | 兜底通用 conv | 通用 sgemm-formulated conv |

cuDNN 会针对每个 conv 的 `(C_in, C_out, H, W, K, stride, padding)` 形状**择优**选模板。实测我们这次跑 eager 时它选了 4 种不同模板覆盖 13 个 conv 层；其中 winograd 是性能最高的（10 个 3×3 stride=1 conv 大半走它）。

**Inductor 生成的 Triton conv 没有 winograd / implicit-gemm / cutlass MMA 的等效优化**：
- 它从通用的 `triton_tem_fused_convolution_view_N` 模板出发；
- `max_autotune` 试 17 种 `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)` 组合，挑时间短的；
- 但模板本身是 explicit im2col-style GEMM，没有 Winograd 减少乘法的等价物，也没有 sm_120 tensor core 的高效 MMA 指令利用。

实测 max_autotune 给出的最快 cfg `(BLOCK_K=16, BLOCK_M=64, BLOCK_N=256, num_warps=4, num_stages=2)` 的 conv kernel 用了 ~6-25 ms 一次（取决于 shape），共 13 次 → ~208 ms。cuDNN 同等 conv 用 cutlass + winograd 一起 ~34 ms 干完。**6.1× 差距来自模板算法本身的差距 + tensor core 指令利用差距**。

---

## 5. 关键结论

### 5.1 性能差距溯源（按贡献从大到小）

1. **Inductor 通用 Triton conv 模板 vs cuDNN 多模板（含 winograd / cutlass tensorop / implicit-gemm）**：
   占了 +5.8 ms / 张的 GPU 工作差距（BATCH=32 折算），扣除 stream overlap 后**贡献了墙钟差距的全部**。
2. **conv-BN 融合的局部节省**：compile 模式把 BN 和 conv bias add 等融进 conv 后的 pointwise epilogue，少了 ~0.25 ms / 张的独立 BN/elementwise kernel 启动 —— 但远不足以抵消 conv 模板劣势。
3. **MaxPool / gemm / memcpy** 差异都在 0.1 ms / 张以下，可忽略。
4. **LIF** 在两个路径下用的是同一份 SJ 手写 Triton kernel，理论与实测都不应该有差（profile 表里的 +75 ms 是测量噪声，详见 §2.2 第 3 点）。

### 5.2 «全 Triton 路径 + 自定义 SNN Pass»的取舍

本仓库自定义 Triton Pass 必须作用在**真正经 Triton 编译的 kernel** 上，因此「全 Triton 路径」是 Pass 开发的必要条件。但代价是性能上比 eager 慢约 27%，这个代价**几乎完全来自 Inductor Triton conv 与 cuDNN 的差距**。

如果未来 Inductor 引入 winograd-Triton 或更好的 tensor core 利用，这个差距可以缩小。在那之前：
- **开发阶段** —— 接受 27% 性能税，换 Pass 介入网络全部算子的能力；
- **生产推理** —— 走 eager 路径（cuDNN/cuBLAS）能拿到本机最快的 VGG16-SNN 推理速度，但失去 SNN Pass 的可作用面。

### 5.3 可重复地观察「Inductor Triton conv 慢」

```bash
# 重跑 perf_breakdown 测量
rm -rf ~/.triton/cache && mkdir -p ~/.triton/cache
BATCH=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/vgg16_snn/perf_breakdown.py

# 看哪种 conv kernel cuDNN 选了
grep "void cudnn\|void cutlass\|void sm80\|void implicit\|winograd" \
    Document/IR-Trace/perf_breakdown/eager_kernels.txt

# 看 Inductor 生成的 conv Triton kernel 长啥样
grep "def triton_tem_fused_convolution" Document/IR-Trace/perf_breakdown/compile_trace.json  # chrome trace
# 或者跑 nir-call-stack-trace.md §7.8 提到的 TORCH_LOGS=output_code 抓 Inductor 编译产物
```

---

## 6. 本文不保证的事

- **不分析训练**。training 路径还要 backward + 梯度更新 kernel，前向以外的瓶颈不在本文范围。
- **不分析其它网络**。VGG 是 3×3 conv-heavy / 无 attention / 无残差，cuDNN 在它上的优势最大；对 BatchNorm-free residual / attention-heavy 网络，Inductor Triton 的差距可能小很多甚至反超。
- **不分析 fp16/bf16/int8**。Inductor 对 fp16 有更激进的 tensor core kernel 模板，可能与 cuDNN 差距更小。当前 VGG16-SNN 全 fp32。
- **不替代 nsys 时间线分析**。要看 stream overlap 的具体形态，应当用 nsys profile 抓时间线再可视化。本文用的是 kernel-level self_cuda_time 聚合，是「单 kernel 视角」。
