# VGG16-SNN 的全部 48 个 Triton Kernel

> 本文展示**一次真实 VGG16-SNN 推理**（`examples/vgg16_snn/vgg16_test.py`）经
> `torch.compile` 生成的**全部 48 个 Triton kernel 的完整代码**，用以解释：为什么
> 一个只有少数几种「操作」的网络会编译出 48 个 kernel。
>
> 捕获方式：`TORCH_LOGS=output_code python examples/vgg16_snn/vgg16_test.py`，
> 即 TorchInductor 为本次推理生成的 Triton 源码。本次推理输出经黄金输出逐字节校验
> 一致（见 [IR-Trace/README.md](./README.md)）。

## 1. 为什么是 48 个 kernel

VGG16-SNN 展开后共有 **49 个层算子**：13 卷积 + 13 BatchNorm + 15 LIF + 5 MaxPool
+ 3 全连接。`torch.compile`（经 TorchInductor）把它们调度、融合后生成 **48 个 Triton
kernel**。

**关键事实——kernel 是按「层位置 × 张量形状」生成的，不是按「操作种类」：**

- 一个 Triton kernel 的张量形状（通道数、H、W、block 尺寸）是**编译期常量**，烧进
  代码里。第 1 个卷积（`3→64, 224×224`）的 kernel 物理上无法处理第 5 个卷积
  （`128→256, 56×56`）的数据——**不同形状 = 不同 kernel**。
- 实证：VGG16 有 13 个卷积层，本次只生成 **9 个卷积模板 kernel**（见 §3.1）。因为
  conv6=conv7、conv9=conv10、conv11=conv12=conv13 这几组卷积形状完全相同，复用同一
  kernel：13 − 2 − 1 − 1 = 9。**kernel 数 = 不同形状数，不是操作种类数。**

**48 = 12 个模板 kernel + 36 个逐元素 kernel：**

- **12 个模板 kernel**（`triton_tem_fused_*`）：9 卷积 + 3 全连接——重计算、GEMM 类，
  含 K 维归约循环。
- **36 个逐元素 kernel**（`triton_poi_fused_*`）：BatchNorm、LIF 脉冲（充电 / Heaviside
  发放 / 复位）、MaxPool，外加**布局转换 kernel**（把张量重排成卷积模板需要的内存
  布局）与**膜电位初始化 kernel**（`zeros_like`，为每个 LIF 层把 `v` 置 0）。
  Inductor 把相邻的 BN+LIF、BN+LIF+MaxPool 纵向融合（减少数量），同时产生布局 /
  初始化辅助 kernel（增加数量），净得 36。

> 一句话：**kernel 数量跟踪的是网络的「深度」（层数 × 不同形状），不是「操作种类」。**
> 这就像「C 程序只有 if/for/赋值几种语句，却编译出几千条指令」——每条都是一个具体
> 位置上的具体实例。

注：每个 kernel 名末尾的 `_N` 是 Inductor 的 buffer 序号；名字里列出的是该 kernel
**融合的全部 ATen 算子**（例如逐元素 kernel 名里出现 `convolution` 是因为它消费了
卷积的输出、作为其 epilogue，并非自己做卷积）。

## 2. 48 个 kernel 总览

| # | kernel 名 | 类型 | 代码行数 |
|---:|---|---|---:|
| 0 | [`triton_poi_fused_convolution_view_0`](#triton_poi_fused_convolution_view_0) | 逐元素 | 16 |
| 1 | [`triton_poi_fused_convolution_view_1`](#triton_poi_fused_convolution_view_1) | 逐元素 | 16 |
| 2 | [`triton_tem_fused_convolution_view_2`](#triton_tem_fused_convolution_view_2) | 模板·卷积 | 111 |
| 3 | [`triton_poi_fused__native_batch_norm_legit_no_training_convolution_view_3`](#triton_poi_fused__native_batch_norm_legit_no_training_convolution_view_3) | 逐元素 | 27 |
| 4 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4) | 逐元素 | 59 |
| 5 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_zeros_like_5`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_zeros_like_5) | 逐元素 | 65 |
| 6 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_select_sub_view_6`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_select_sub_view_6) | 逐元素 | 16 |
| 7 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_select_sub_view_7`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_select_sub_view_7) | 模板·卷积 | 111 |
| 8 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_zeros_like_8`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_zeros_like_8) | 逐元素 | 62 |
| 9 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_9`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_9) | 逐元素 | 18 |
| 10 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_10`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_10) | 逐元素 | 16 |
| 11 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_11`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_11) | 模板·卷积 | 111 |
| 12 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_12`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_12) | 逐元素 | 27 |
| 13 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_13`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_13) | 逐元素 | 59 |
| 14 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_14`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_14) | 逐元素 | 65 |
| 15 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_15`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_15) | 逐元素 | 16 |
| 16 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_16`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_16) | 模板·卷积 | 111 |
| 17 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_17`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_17) | 逐元素 | 62 |
| 18 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_18`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_18) | 逐元素 | 18 |
| 19 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_19`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_19) | 逐元素 | 16 |
| 20 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_20`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_20) | 模板·卷积 | 111 |
| 21 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_21`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_21) | 逐元素 | 27 |
| 22 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_22`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_22) | 逐元素 | 59 |
| 23 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_23`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_23) | 逐元素 | 65 |
| 24 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_24`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_24) | 逐元素 | 16 |
| 25 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_25`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_25) | 模板·卷积 | 111 |
| 26 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_26`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_26) | 逐元素 | 62 |
| 27 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_27`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_27) | 逐元素 | 18 |
| 28 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_28`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_28) | 逐元素 | 16 |
| 29 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_29`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_29) | 模板·卷积 | 111 |
| 30 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_30`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_30) | 逐元素 | 27 |
| 31 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_31`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_31) | 逐元素 | 59 |
| 32 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_32`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_32) | 逐元素 | 65 |
| 33 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_33`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_33) | 逐元素 | 16 |
| 34 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_34`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_34) | 模板·卷积 | 111 |
| 35 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_35`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_35) | 逐元素 | 62 |
| 36 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_36`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_36) | 逐元素 | 18 |
| 37 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_37`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_37) | 模板·卷积 | 111 |
| 38 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_38`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_38) | 逐元素 | 27 |
| 39 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_39`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_39) | 逐元素 | 59 |
| 40 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_40`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_40) | 逐元素 | 65 |
| 41 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_41`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_41) | 逐元素 | 62 |
| 42 | [`triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_42`](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_42) | 逐元素 | 23 |
| 43 | [`triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_addmm_convolution_div_ge_max_pool2d_with_indices_select_sub_t_view_43`](#triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_addmm_convolution_div_ge_max_pool2d_with_indices_select_sub_t_view_43) | 模板·矩阵乘法 | 81 |
| 44 | [`triton_poi_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_view_44`](#triton_poi_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_view_44) | 逐元素 | 58 |
| 45 | [`triton_poi_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_view_zeros_like_45`](#triton_poi_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_view_zeros_like_45) | 逐元素 | 59 |
| 46 | [`triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46`](#triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46) | 模板·矩阵乘法 | 81 |
| 47 | [`triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_47`](#triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_47) | 模板·矩阵乘法 | 83 |

合计：模板·卷积 **9** 个、模板·矩阵乘法 **3** 个、逐元素 **36** 个 —— 共 **48** 个。

## 3. 全部 48 个 kernel 的完整代码

下列代码是 Inductor 生成的 Triton kernel 源码，逐字摘自本次真实推理的 `output_code`。
所有 kernel 共享同一段导入样板，此处统一列出、各 kernel 不再重复：

```python
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
```

每个 kernel 给出其 `@triton.jit` 函数全文（Inductor 的 `@triton_heuristics` 配置
装饰器中的关键参数已在每个 kernel 的引述行列出）。

### 3.1 模板 kernel —— 卷积（9 个）

9 个卷积模板 kernel，对应 VGG16 的 13 个卷积层去重后的 9 种不同形状。每个含一个 K 维归约 `for` 循环（`tl.dot` 累加）。

#### triton_tem_fused_convolution_view_2

> Inductor buffer #2｜模板·卷积｜num_warps=4，num_stages=4

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused_convolution_view_2(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 3
    IN_H = 224
    IN_W = 224
    OUT_C = 64
    OUT_H = 224
    OUT_W = 224

    # Strides:
    stride_xn = 150528
    stride_xc = 1
    stride_xh = 672
    stride_xw = 3
    stride_wc_out = 27
    stride_wc_in = 1
    stride_wh = 9
    stride_ww = 3

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 224*idx_h + 50176*idx_c + 3211264*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 64*idx_w + 14336*idx_h + 3211264*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_select_sub_view_7

> Inductor buffer #7｜模板·卷积｜num_warps=4，num_stages=2

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_select_sub_view_7(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 64
    IN_H = 224
    IN_W = 224
    OUT_C = 64
    OUT_H = 224
    OUT_W = 224

    # Strides:
    stride_xn = 3211264
    stride_xc = 1
    stride_xh = 14336
    stride_xw = 64
    stride_wc_out = 576
    stride_wc_in = 1
    stride_wh = 192
    stride_ww = 64

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 224*idx_h + 50176*idx_c + 3211264*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 64*idx_w + 14336*idx_h + 3211264*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_11

> Inductor buffer #11｜模板·卷积｜num_warps=4，num_stages=2

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_11(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 64
    IN_H = 112
    IN_W = 112
    OUT_C = 128
    OUT_H = 112
    OUT_W = 112

    # Strides:
    stride_xn = 802816
    stride_xc = 1
    stride_xh = 7168
    stride_xw = 64
    stride_wc_out = 576
    stride_wc_in = 1
    stride_wh = 192
    stride_ww = 64

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 112*idx_h + 12544*idx_c + 1605632*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 128*idx_w + 14336*idx_h + 1605632*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_16

> Inductor buffer #16｜模板·卷积｜num_warps=4，num_stages=2

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_16(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 128
    IN_H = 112
    IN_W = 112
    OUT_C = 128
    OUT_H = 112
    OUT_W = 112

    # Strides:
    stride_xn = 1605632
    stride_xc = 1
    stride_xh = 14336
    stride_xw = 128
    stride_wc_out = 1152
    stride_wc_in = 1
    stride_wh = 384
    stride_ww = 128

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 112*idx_h + 12544*idx_c + 1605632*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 128*idx_w + 14336*idx_h + 1605632*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_20

> Inductor buffer #20｜模板·卷积｜num_warps=4，num_stages=2

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_20(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 128
    IN_H = 56
    IN_W = 56
    OUT_C = 256
    OUT_H = 56
    OUT_W = 56

    # Strides:
    stride_xn = 401408
    stride_xc = 1
    stride_xh = 7168
    stride_xw = 128
    stride_wc_out = 1152
    stride_wc_in = 1
    stride_wh = 384
    stride_ww = 128

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 56*idx_h + 3136*idx_c + 802816*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 256*idx_w + 14336*idx_h + 802816*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_25

> Inductor buffer #25｜模板·卷积｜num_warps=4，num_stages=2

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_25(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 256
    IN_H = 56
    IN_W = 56
    OUT_C = 256
    OUT_H = 56
    OUT_W = 56

    # Strides:
    stride_xn = 802816
    stride_xc = 1
    stride_xh = 14336
    stride_xw = 256
    stride_wc_out = 2304
    stride_wc_in = 1
    stride_wh = 768
    stride_ww = 256

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 56*idx_h + 3136*idx_c + 802816*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 256*idx_w + 14336*idx_h + 802816*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_29

> Inductor buffer #29｜模板·卷积｜num_warps=4，num_stages=2

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_29(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 256
    IN_H = 28
    IN_W = 28
    OUT_C = 512
    OUT_H = 28
    OUT_W = 28

    # Strides:
    stride_xn = 200704
    stride_xc = 1
    stride_xh = 7168
    stride_xw = 256
    stride_wc_out = 2304
    stride_wc_in = 1
    stride_wh = 768
    stride_ww = 256

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 28*idx_h + 784*idx_c + 401408*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 512*idx_w + 14336*idx_h + 401408*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_34

> Inductor buffer #34｜模板·卷积｜num_warps=4，num_stages=2

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_34(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 512
    IN_H = 28
    IN_W = 28
    OUT_C = 512
    OUT_H = 28
    OUT_W = 28

    # Strides:
    stride_xn = 401408
    stride_xc = 1
    stride_xh = 14336
    stride_xw = 512
    stride_wc_out = 4608
    stride_wc_in = 1
    stride_wh = 1536
    stride_ww = 512

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 28*idx_h + 784*idx_c + 401408*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 512*idx_w + 14336*idx_h + 401408*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_37

> Inductor buffer #37｜模板·卷积｜num_warps=4，num_stages=2

签名：`{'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_37(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 512
    IN_H = 14
    IN_W = 14
    OUT_C = 512
    OUT_H = 14
    OUT_W = 14

    # Strides:
    stride_xn = 100352
    stride_xc = 1
    stride_xh = 7168
    stride_xw = 512
    stride_wc_out = 4608
    stride_wc_in = 1
    stride_wh = 1536
    stride_ww = 512

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 14*idx_h + 196*idx_c + 100352*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 512*idx_w + 7168*idx_h + 100352*idx_n, [BLOCK_M, BLOCK_N])), acc, mask)
```

### 3.2 模板 kernel —— 全连接 / 矩阵乘法（3 个）

3 个矩阵乘法模板 kernel，对应分类器的 3 个全连接层（`addmm`）。

#### triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_addmm_convolution_div_ge_max_pool2d_with_indices_select_sub_t_view_43

> Inductor buffer #43｜模板·矩阵乘法｜num_warps=2，num_stages=5

签名：`{'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_add_addmm_convolution_div_ge_max_pool2d_with_indices_select_sub_t_view_43(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4
    N = 4096
    K = 25088
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 25088
    stride_ak = 1
    stride_bk = 1
    stride_bn = 25088

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 25088*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 4096*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 25088*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 4096*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46

> Inductor buffer #46｜模板·矩阵乘法｜num_warps=2，num_stages=5

签名：`{'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_46(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4
    N = 4096
    K = 4096
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 4096
    stride_ak = 1
    stride_bk = 1
    stride_bn = 4096

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 4096*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 4096*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 4096*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 4096*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
```

#### triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_47

> Inductor buffer #47｜模板·矩阵乘法｜num_warps=2，num_stages=5

签名：`{'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}`

```python
@triton.jit
def triton_tem_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_t_view_47(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 4
    N = 1000
    K = 4096
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 4096
    stride_ak = 1
    stride_bk = 1
    stride_bn = 4096

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 4096*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 4096*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1000*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
```

### 3.3 逐元素 kernel（36 个）

36 个逐元素 kernel：BatchNorm、LIF 脉冲计算、MaxPool、布局转换、膜电位初始化等。无归约循环，在二维 grid 上逐元素并行。

#### triton_poi_fused_convolution_view_0

> Inductor buffer #0｜逐元素｜size_hints={'y': 16, 'x': 65536}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused_convolution_view_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)
```

#### triton_poi_fused_convolution_view_1

> Inductor buffer #1｜逐元素｜size_hints={'y': 256, 'x': 16}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused_convolution_view_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training_convolution_view_3

> Inductor buffer #3｜逐元素｜size_hints={'x': 16777216}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4

> Inductor buffer #4｜逐元素｜size_hints={'y': 65536, 'x': 64}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4(in_out_ptr0, in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 64*y0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3211264 + x1 + 64*y0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (6422528 + x1 + 64*y0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (9633792 + x1 + 64*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1, 1], 1.0, tl.float32)
    tmp4 = tmp2 >= tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full([1, 1], 0.0, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 - tmp5
    tmp9 = tmp8 * tmp2
    tmp10 = tmp7 + tmp9
    tmp12 = tmp10 - tmp6
    tmp13 = tmp11 - tmp12
    tmp14 = tmp13 * tmp1
    tmp15 = tmp10 + tmp14
    tmp16 = tmp15 >= tmp3
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp6
    tmp19 = tmp3 - tmp17
    tmp20 = tmp19 * tmp15
    tmp21 = tmp18 + tmp20
    tmp23 = tmp21 - tmp6
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp1
    tmp26 = tmp21 + tmp25
    tmp27 = tmp26 >= tmp3
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp3 - tmp28
    tmp30 = tmp29 * tmp26
    tmp31 = tmp28 * tmp6
    tmp32 = tmp31 + tmp30
    tmp34 = tmp32 - tmp6
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 * tmp1
    tmp37 = tmp32 + tmp36
    tmp38 = tmp37 >= tmp3
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39 * tmp6
    tmp41 = tmp3 - tmp39
    tmp42 = tmp41 * tmp37
    tmp43 = tmp40 + tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 64*y0), tmp32, xmask)
    tl.store(out_ptr0 + (y0 + 50176*x1), tmp43, xmask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_zeros_like_5

> Inductor buffer #5｜逐元素｜size_hints={'y': 256, 'x': 65536}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_zeros_like_5(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y1 = yindex // 64
    x2 = xindex
    y0 = (yindex % 64)
    y3 = yindex
    tmp3 = tl.load(in_ptr0 + (y0 + 64*x2), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3211264 + y0 + 64*x2), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (6422528 + y0 + 64*x2), xmask & ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (y0 + 64*x2), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (9633792 + y0 + 64*x2), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y1
    tmp1 = tl.full([1, 1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1, 1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1, 1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr0 + (y0 + 64*x2 + 3211264*y1), tmp49, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_select_sub_view_6

> Inductor buffer #6｜逐元素｜size_hints={'y': 4096, 'x': 16}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_select_sub_view_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_zeros_like_8

> Inductor buffer #8｜逐元素｜size_hints={'x': 16777216}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_zeros_like_8(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex // 3211264
    x3 = (xindex % 3211264)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 50176)
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (3211264 + x3), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (6422528 + x3), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (9633792 + x3), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr1 + (x4), tmp49, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_9

> Inductor buffer #9｜逐元素｜size_hints={'x': 4194304}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 112)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 28672*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 28672*x2), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 128*x1 + 28672*x2), None)
    tmp5 = tl.load(in_ptr0 + (14400 + x0 + 128*x1 + 28672*x2), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_10

> Inductor buffer #10｜逐元素｜size_hints={'y': 8192, 'x': 16}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_12

> Inductor buffer #12｜逐元素｜size_hints={'x': 8388608}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_13

> Inductor buffer #13｜逐元素｜size_hints={'y': 16384, 'x': 128}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_13(in_out_ptr0, in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 128*y0), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (1605632 + x1 + 128*y0), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (3211264 + x1 + 128*y0), xmask & ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (4816896 + x1 + 128*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1, 1], 1.0, tl.float32)
    tmp4 = tmp2 >= tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full([1, 1], 0.0, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 - tmp5
    tmp9 = tmp8 * tmp2
    tmp10 = tmp7 + tmp9
    tmp12 = tmp10 - tmp6
    tmp13 = tmp11 - tmp12
    tmp14 = tmp13 * tmp1
    tmp15 = tmp10 + tmp14
    tmp16 = tmp15 >= tmp3
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp6
    tmp19 = tmp3 - tmp17
    tmp20 = tmp19 * tmp15
    tmp21 = tmp18 + tmp20
    tmp23 = tmp21 - tmp6
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp1
    tmp26 = tmp21 + tmp25
    tmp27 = tmp26 >= tmp3
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp3 - tmp28
    tmp30 = tmp29 * tmp26
    tmp31 = tmp28 * tmp6
    tmp32 = tmp31 + tmp30
    tmp34 = tmp32 - tmp6
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 * tmp1
    tmp37 = tmp32 + tmp36
    tmp38 = tmp37 >= tmp3
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39 * tmp6
    tmp41 = tmp3 - tmp39
    tmp42 = tmp41 * tmp37
    tmp43 = tmp40 + tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 128*y0), tmp32, xmask & ymask)
    tl.store(out_ptr0 + (y0 + 12544*x1), tmp43, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_14

> Inductor buffer #14｜逐元素｜size_hints={'y': 512, 'x': 16384}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_14(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y1 = yindex // 128
    x2 = xindex
    y0 = (yindex % 128)
    y3 = yindex
    tmp3 = tl.load(in_ptr0 + (y0 + 128*x2), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1605632 + y0 + 128*x2), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (3211264 + y0 + 128*x2), xmask & ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (y0 + 128*x2), xmask & ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (4816896 + y0 + 128*x2), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y1
    tmp1 = tl.full([1, 1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1, 1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1, 1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1605632*y1), tmp49, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_15

> Inductor buffer #15｜逐元素｜size_hints={'y': 16384, 'x': 16}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_17

> Inductor buffer #17｜逐元素｜size_hints={'x': 8388608}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_17(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex // 1605632
    x3 = (xindex % 1605632)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 12544)
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (1605632 + x3), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (3211264 + x3), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (4816896 + x3), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr1 + (x4), tmp49, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_18

> Inductor buffer #18｜逐元素｜size_hints={'x': 2097152}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 56)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 28672*x2), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 28672*x2), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 256*x1 + 28672*x2), None)
    tmp5 = tl.load(in_ptr0 + (14464 + x0 + 256*x1 + 28672*x2), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_19

> Inductor buffer #19｜逐元素｜size_hints={'y': 32768, 'x': 16}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_21

> Inductor buffer #21｜逐元素｜size_hints={'x': 4194304}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_22

> Inductor buffer #22｜逐元素｜size_hints={'y': 4096, 'x': 256}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_22(in_out_ptr0, in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (802816 + x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (1605632 + x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (2408448 + x1 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1, 1], 1.0, tl.float32)
    tmp4 = tmp2 >= tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full([1, 1], 0.0, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 - tmp5
    tmp9 = tmp8 * tmp2
    tmp10 = tmp7 + tmp9
    tmp12 = tmp10 - tmp6
    tmp13 = tmp11 - tmp12
    tmp14 = tmp13 * tmp1
    tmp15 = tmp10 + tmp14
    tmp16 = tmp15 >= tmp3
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp6
    tmp19 = tmp3 - tmp17
    tmp20 = tmp19 * tmp15
    tmp21 = tmp18 + tmp20
    tmp23 = tmp21 - tmp6
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp1
    tmp26 = tmp21 + tmp25
    tmp27 = tmp26 >= tmp3
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp3 - tmp28
    tmp30 = tmp29 * tmp26
    tmp31 = tmp28 * tmp6
    tmp32 = tmp31 + tmp30
    tmp34 = tmp32 - tmp6
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 * tmp1
    tmp37 = tmp32 + tmp36
    tmp38 = tmp37 >= tmp3
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39 * tmp6
    tmp41 = tmp3 - tmp39
    tmp42 = tmp41 * tmp37
    tmp43 = tmp40 + tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 256*y0), tmp32, xmask & ymask)
    tl.store(out_ptr0 + (y0 + 3136*x1), tmp43, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_23

> Inductor buffer #23｜逐元素｜size_hints={'y': 1024, 'x': 4096}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_23(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y1 = yindex // 256
    x2 = xindex
    y0 = (yindex % 256)
    y3 = yindex
    tmp3 = tl.load(in_ptr0 + (y0 + 256*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (802816 + y0 + 256*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (1605632 + y0 + 256*x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (y0 + 256*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (2408448 + y0 + 256*x2), xmask, eviction_policy='evict_last')
    tmp0 = y1
    tmp1 = tl.full([1, 1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1, 1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1, 1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr0 + (y0 + 256*x2 + 802816*y1), tmp49, xmask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_24

> Inductor buffer #24｜逐元素｜size_hints={'y': 65536, 'x': 16}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_26

> Inductor buffer #26｜逐元素｜size_hints={'x': 4194304}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_26(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex // 802816
    x3 = (xindex % 802816)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 3136)
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (802816 + x3), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (1605632 + x3), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (2408448 + x3), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr1 + (x4), tmp49, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_27

> Inductor buffer #27｜逐元素｜size_hints={'x': 1048576}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 28)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 28672*x2), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 28672*x2), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 512*x1 + 28672*x2), None)
    tmp5 = tl.load(in_ptr0 + (14592 + x0 + 512*x1 + 28672*x2), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_28

> Inductor buffer #28｜逐元素｜size_hints={'y': 131072, 'x': 16}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_30

> Inductor buffer #30｜逐元素｜size_hints={'x': 2097152}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_31

> Inductor buffer #31｜逐元素｜size_hints={'y': 1024, 'x': 512}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_31(in_out_ptr0, in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (401408 + x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (802816 + x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (1204224 + x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1, 1], 1.0, tl.float32)
    tmp4 = tmp2 >= tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full([1, 1], 0.0, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 - tmp5
    tmp9 = tmp8 * tmp2
    tmp10 = tmp7 + tmp9
    tmp12 = tmp10 - tmp6
    tmp13 = tmp11 - tmp12
    tmp14 = tmp13 * tmp1
    tmp15 = tmp10 + tmp14
    tmp16 = tmp15 >= tmp3
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp6
    tmp19 = tmp3 - tmp17
    tmp20 = tmp19 * tmp15
    tmp21 = tmp18 + tmp20
    tmp23 = tmp21 - tmp6
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp1
    tmp26 = tmp21 + tmp25
    tmp27 = tmp26 >= tmp3
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp3 - tmp28
    tmp30 = tmp29 * tmp26
    tmp31 = tmp28 * tmp6
    tmp32 = tmp31 + tmp30
    tmp34 = tmp32 - tmp6
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 * tmp1
    tmp37 = tmp32 + tmp36
    tmp38 = tmp37 >= tmp3
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39 * tmp6
    tmp41 = tmp3 - tmp39
    tmp42 = tmp41 * tmp37
    tmp43 = tmp40 + tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 512*y0), tmp32, xmask & ymask)
    tl.store(out_ptr0 + (y0 + 784*x1), tmp43, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_32

> Inductor buffer #32｜逐元素｜size_hints={'y': 2048, 'x': 1024}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_32(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y1 = yindex // 512
    x2 = xindex
    y0 = (yindex % 512)
    y3 = yindex
    tmp3 = tl.load(in_ptr0 + (y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (401408 + y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (802816 + y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (1204224 + y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp0 = y1
    tmp1 = tl.full([1, 1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1, 1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1, 1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr0 + (y0 + 512*x2 + 401408*y1), tmp49, xmask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_33

> Inductor buffer #33｜逐元素｜size_hints={'y': 262144, 'x': 16}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_33(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_35

> Inductor buffer #35｜逐元素｜size_hints={'x': 2097152}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_35(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex // 401408
    x3 = (xindex % 401408)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 784)
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (401408 + x3), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (802816 + x3), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (1204224 + x3), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr1 + (x4), tmp49, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_36

> Inductor buffer #36｜逐元素｜size_hints={'x': 524288}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 14)
    x2 = xindex // 7168
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1 + 28672*x2), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 1024*x1 + 28672*x2), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + 1024*x1 + 28672*x2), None)
    tmp5 = tl.load(in_ptr0 + (14848 + x0 + 1024*x1 + 28672*x2), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_38

> Inductor buffer #38｜逐元素｜size_hints={'x': 524288}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_39

> Inductor buffer #39｜逐元素｜size_hints={'y': 256, 'x': 512}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_39(in_out_ptr0, in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (100352 + x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (200704 + x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (301056 + x1 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1, 1], 1.0, tl.float32)
    tmp4 = tmp2 >= tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full([1, 1], 0.0, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 - tmp5
    tmp9 = tmp8 * tmp2
    tmp10 = tmp7 + tmp9
    tmp12 = tmp10 - tmp6
    tmp13 = tmp11 - tmp12
    tmp14 = tmp13 * tmp1
    tmp15 = tmp10 + tmp14
    tmp16 = tmp15 >= tmp3
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp6
    tmp19 = tmp3 - tmp17
    tmp20 = tmp19 * tmp15
    tmp21 = tmp18 + tmp20
    tmp23 = tmp21 - tmp6
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp1
    tmp26 = tmp21 + tmp25
    tmp27 = tmp26 >= tmp3
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp3 - tmp28
    tmp30 = tmp29 * tmp26
    tmp31 = tmp28 * tmp6
    tmp32 = tmp31 + tmp30
    tmp34 = tmp32 - tmp6
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 * tmp1
    tmp37 = tmp32 + tmp36
    tmp38 = tmp37 >= tmp3
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39 * tmp6
    tmp41 = tmp3 - tmp39
    tmp42 = tmp41 * tmp37
    tmp43 = tmp40 + tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 512*y0), tmp32, xmask & ymask)
    tl.store(out_ptr0 + (y0 + 196*x1), tmp43, xmask & ymask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_40

> Inductor buffer #40｜逐元素｜size_hints={'y': 2048, 'x': 256}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_40(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y1 = yindex // 512
    x2 = xindex
    y0 = (yindex % 512)
    y3 = yindex
    tmp3 = tl.load(in_ptr0 + (y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (100352 + y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (200704 + y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (301056 + y0 + 512*x2), xmask, eviction_policy='evict_last')
    tmp0 = y1
    tmp1 = tl.full([1, 1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1, 1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1, 1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1, 1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr0 + (y0 + 512*x2 + 100352*y1), tmp49, xmask)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_41

> Inductor buffer #41｜逐元素｜size_hints={'x': 524288}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_mul_rsub_select_sub_view_zeros_like_41(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex // 100352
    x3 = (xindex % 100352)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 196)
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (100352 + x3), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (200704 + x3), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (301056 + x3), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1], 0.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 - tmp8
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 - tmp9
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp4
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 >= tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp6 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp24 - tmp9
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp4
    tmp29 = tmp24 + tmp28
    tmp30 = tmp29 >= tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1], 1, tl.int32)
    tmp33 = tmp0 == tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.where(tmp35, tmp8, tmp9)
    tmp37 = tl.where(tmp33, tmp20, tmp36)
    tmp38 = tl.where(tmp2, tmp31, tmp37)
    tmp39 = tl.full([1], 3, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp41 - tmp9
    tmp44 = tmp42 - tmp43
    tmp45 = tmp44 * tmp4
    tmp46 = tmp41 + tmp45
    tmp47 = tmp46 >= tmp6
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tl.where(tmp40, tmp48, tmp38)
    tl.store(out_ptr1 + (x4), tmp49, None)
```

#### triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_42

> Inductor buffer #42｜逐元素｜size_hints={'y': 256, 'x': 512}

签名：`{'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_max_pool2d_with_indices_select_sub_view_42(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 7)
    y4 = yindex // 7
    y2 = yindex // 49
    y5 = (yindex % 49)
    tmp0 = tl.load(in_ptr0 + (x3 + 1024*y0 + 14336*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (512 + x3 + 1024*y0 + 14336*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (7168 + x3 + 1024*y0 + 14336*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (7680 + x3 + 1024*y0 + 14336*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y5 + 49*x3 + 25088*y2), tmp6, xmask & ymask)
```

#### triton_poi_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_view_44

> Inductor buffer #44｜逐元素｜size_hints={'x': 4096}

签名：`{'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_view_44(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp13 = tl.load(in_ptr1 + (4096 + x0), None)
    tmp25 = tl.load(in_ptr1 + (8192 + x0), None)
    tmp37 = tl.load(in_ptr1 + (12288 + x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 >= tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1], 0.0, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp5 - tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tmp14 = tmp0 + tmp13
    tmp15 = tmp12 - tmp8
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp3
    tmp18 = tmp12 + tmp17
    tmp19 = tmp18 >= tmp5
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp8
    tmp22 = tmp5 - tmp20
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 + tmp23
    tmp26 = tmp0 + tmp25
    tmp27 = tmp24 - tmp8
    tmp28 = tmp26 - tmp27
    tmp29 = tmp28 * tmp3
    tmp30 = tmp24 + tmp29
    tmp31 = tmp30 >= tmp5
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp8
    tmp34 = tmp5 - tmp32
    tmp35 = tmp34 * tmp30
    tmp36 = tmp33 + tmp35
    tmp38 = tmp0 + tmp37
    tmp39 = tmp36 - tmp8
    tmp40 = tmp38 - tmp39
    tmp41 = tmp40 * tmp3
    tmp42 = tmp36 + tmp41
    tmp43 = tmp42 >= tmp5
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp8
    tmp46 = tmp5 - tmp44
    tmp47 = tmp46 * tmp42
    tmp48 = tmp45 + tmp47
    tl.store(out_ptr0 + (x0), tmp30, None)
    tl.store(out_ptr1 + (x0), tmp48, None)
```

#### triton_poi_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_view_zeros_like_45

> Inductor buffer #45｜逐元素｜size_hints={'x': 16384}

签名：`{'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}`

```python
@triton.jit
def triton_poi_fused__to_copy_add_addmm_div_ge_mul_rsub_select_sub_view_zeros_like_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 4096
    x0 = (xindex % 4096)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (4096 + x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr2 + (12288 + x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0.5, tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 >= tmp4
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp4 - tmp15
    tmp19 = tmp18 * tmp13
    tmp20 = tmp17 + tmp19
    tmp22 = tmp9 + tmp21
    tmp23 = tmp20 - tmp16
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp12
    tmp26 = tmp20 + tmp25
    tmp27 = tmp26 >= tmp4
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.full([1], 0, tl.int32)
    tmp30 = tmp0 == tmp29
    tmp31 = tl.where(tmp30, tmp15, tmp16)
    tmp32 = tl.where(tmp8, tmp28, tmp31)
    tmp33 = tl.where(tmp2, tmp6, tmp32)
    tmp34 = tl.full([1], 3, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tmp6 * tmp16
    tmp37 = tmp4 - tmp6
    tmp38 = tmp37 * tmp3
    tmp39 = tmp36 + tmp38
    tmp41 = tmp9 + tmp40
    tmp42 = tmp39 - tmp16
    tmp43 = tmp41 - tmp42
    tmp44 = tmp43 * tmp12
    tmp45 = tmp39 + tmp44
    tmp46 = tmp45 >= tmp4
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.where(tmp35, tmp47, tmp33)
    tl.store(in_out_ptr0 + (x2), tmp48, None)
```

## 4. kernel 代码与 TTIR 的关系：TTIR 是「48 个 kernel 的组合」吗？

**不是。TTIR 不是多个 kernel 的组合——它是单个 Triton kernel 的 1:1 MLIR 翻译。**

§3 给出的 48 个 `@triton.jit` kernel，每一个都**独立编译**：其 Python 源码 →（前端
`ast_to_ttir`）→ **它自己的一份 TTIR** → TTGIR → LLVM IR → 它自己的 cubin。48 个 kernel
= 48 份互不相干的 IR、48 次独立编译。`Document/IR-Trace/<kernel>/` 下每个
`stage_0_entry.ttir` 都只是**那一个** kernel 的 TTIR——例如
[`bn_lif/stage_0_entry.ttir`](./bn_lif/stage_0_entry.ttir) 就是 §3.3 里 buffer #4 那
[一个 kernel](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4)
的 TTIR，不含其它任何 kernel。

### 4.1 真正「组合」48 个 kernel 的是 Inductor 的 `call()` 函数

把 48 个 kernel 串成完整 VGG16-SNN 推理的，**不是任何 IR、也不是 Triton kernel**，而是
Inductor 生成的 host 侧 Python 编排函数 `call()`：它分配中间张量缓冲、按拓扑序逐个
`.run()` 启动 kernel：

```python
buf0 = empty_strided_cuda((4, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
triton_poi_fused_convolution_view_0.run(arg0_1, buf0, 12, 50176, stream=stream0)
buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
triton_poi_fused_convolution_view_1.run(arg1_1, buf1, 192, 9, stream=stream0)
buf2 = empty_strided_cuda((4, 64, 224, 224), (3211264, 1, 14336, 64), torch.float32)
...   # 依次 .run() 启动全部 48 个 kernel，中间张量 buf* 在它们之间传递
```

`call()` 运行在 PyTorch 运行时层、Triton 之外，不参与 Triton 的 IR 流水线。所以「组合」
发生在 host 侧 Python，而非任何 IR。

### 4.2 kernel 代码 ↔ TTIR 逐行对应（以 buffer #4 为例）

`@triton.jit` 源码与 TTIR 是同一个 kernel 的两种形态、1:1 翻译。下表把 buffer #4
[BN+LIF kernel 的源码](#triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_convolution_div_ge_mul_rsub_select_sub_view_4)
与它的真实 TTIR [`bn_lif/stage_0_entry.ttir`](./bn_lif/stage_0_entry.ttir) 逐行对照
（右列链接指向真实 IR 的具体行号）：

| `@triton.jit` 源码 | TTIR（`tt` 方言，真实行号） | 说明 |
|---|---|---|
| `def ..._view_4(in_out_ptr0, in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK, XBLOCK)` | [`tt.func public @..._view_4(%in_out_ptr0, %in_ptr0, %out_ptr0, %ynumel, %xnumel)`（L8）](./bn_lif/stage_0_entry.ttir#L8) | 函数签名；`YBLOCK/XBLOCK` 是 `constexpr`，已折叠为常量、不进运行时签名 |
| `yoffset = tl.program_id(1) * YBLOCK` | [`%yoffset = tt.get_program_id y`（L20）](./bn_lif/stage_0_entry.ttir#L20)、[`arith.muli ..., %c16_i32`（L21）](./bn_lif/stage_0_entry.ttir#L21) | `YBLOCK` 折叠为常量 16 |
| `tl.arange(0, YBLOCK)[:, None]` | [`tt.make_range {end=16}`（L22）](./bn_lif/stage_0_entry.ttir#L22)、[`tt.expand_dims {axis=1}`（L23）](./bn_lif/stage_0_entry.ttir#L23) | `[:, None]` = 增加一维 |
| `xmask = xindex < xnumel` | [`%xmask_12 = arith.cmpi slt`（L32）](./bn_lif/stage_0_entry.ttir#L32) | 越界掩码 |
| `tmp0 = tl.load(in_ptr0+(x1+64*y0), xmask, eviction_policy='evict_last')` | [`%tmp0_19 = tt.load ... evictionPolicy = evict_last`（L40）](./bn_lif/stage_0_entry.ttir#L40) | 时间步 0 取数（地址计算 L33–39）|
| `tmp11 = tl.load(in_ptr0+(3211264+...), ...)` | [`%tmp11_24 = tt.load`（L45）](./bn_lif/stage_0_entry.ttir#L45)；偏移常量 [`%tmp11 = dense<3211264>`（L15）](./bn_lif/stage_0_entry.ttir#L15) | 时间步 1 取数 |
| `tmp22 = tl.load(in_ptr0+(6422528+...), ...)` | [`%tmp22_29 = tt.load`（L50）](./bn_lif/stage_0_entry.ttir#L50) | 时间步 2 取数 |
| `tmp33 = tl.load(in_ptr0+(9633792+...), ...)` | [`%tmp33_34 = tt.load`（L55）](./bn_lif/stage_0_entry.ttir#L55) | 时间步 3 取数 |
| `tmp1 = tl.full([1,1], 0.5, ...)` | [`%cst_2 = arith.constant dense<5.0e-01>`（L12）](./bn_lif/stage_0_entry.ttir#L12) | LIF 缩放常量 0.5 |
| `tmp2 = tmp0 * tmp1` | [`%tmp2 = arith.mulf %tmp0_19, %cst_2`（L56）](./bn_lif/stage_0_entry.ttir#L56) | 膜电位充电 |
| `tmp4 = tmp2 >= tmp3` | [`%tmp4 = arith.cmpf oge, %tmp2, %cst_1`（L57）](./bn_lif/stage_0_entry.ttir#L57) | 时间步 0 Heaviside 发放 |
| `tmp5 = tmp4.to(tl.float32)` | [`%tmp5 = arith.uitofp %tmp4`（L58）](./bn_lif/stage_0_entry.ttir#L58) | 脉冲 0/1 转 float |

Triton 前端把 **Python 变量名带进了 TTIR**——`tmp2`、`tmp4`、`tmp11`、`tmp22`、`tmp33`
在 TTIR 里仍叫 `%tmp2`、`%tmp4`、`%tmp11…`（`_NN` 后缀只是 MLIR 的 SSA 去重）。对应关系
因此一目了然：**TTIR 就是这一个 kernel 的 MLIR 写法，没有「组合」任何东西。**

> 这份代码 / TTIR 里时间步为何是 4 份展开而非循环，见
> [`Optimization-Insights.md` §1.6](./Optimization-Insights.md)。

