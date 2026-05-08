# AOT ID: ['0_backward']
# ============================================================
# Level 4: PyTorch Inductor Wrapper (CPU Orchestrator)
# ============================================================
# 这是真实运行时由 PyTorch Inductor 生成、存放于
# /tmp/torchinductor_charlley/md/cmd257vas74ipq2tjgauqes7ilmrpkyrijcaqa6hoxp6g3xpvypa.py
# 的完整 Python Wrapper 代码（精简版，保留关键执行调度逻辑）。
#
# 【这一层的职责】
# CPU 上的总调度器：
#   1. 接收 PyTorch runtime 传来的 Tensor 参数
#   2. 执行形状/步长 Guard 验证（确保与追踪时的形状完全一致）
#   3. 通过 empty_strided_cuda 在 GPU 上申请中间缓存
#   4. 顺序调度多个 Triton Kernel 和外部 BLAS 算子（如 cuBLAS mm）
#   5. 管理显存复用（buf = old_tensor; del old_tensor  # reuse）
#
# 【Kernel 执行顺序（反向传播）】
# 步骤 0: triton_poi_..._sub_0      <- LIF2 (fc 层之后) 的 ATan 梯度
# 步骤 1: extern_kernels.mm        <- 线性层反向: dX = grad @ W (cuBLAS)
# 步骤 2: triton_poi_...view_1      <- MaxPool 反向 + BatchNorm 前向项
# 步骤 3: triton_red_..._sub_2      <- BatchNorm 反向 Reduction (sum_2, sum_3)
# 步骤 4: triton_poi_..._sub_3      <- LIF1 + BatchNorm Elementwise 反向  ← 这是 Level 3 的 Kernel
# 步骤 5: aten.convolution_backward <- Conv2d 反向 (使用 cuDNN/CUDA 内置算子)
# 步骤 6: triton_poi_fused_sum_4    <- bias 梯度的 sum
# 步骤 7: extern_kernels.mm        <- 线性层权重梯度: dW = grad.T @ x (cuBLAS)
# 步骤 8: triton_red_..._backward_5 <- Conv bias 梯度的 Reduction
#
# ============================================================

from ctypes import c_void_p, c_long, c_int
import torch
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.guards import assert_size_stride

empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor

class Runner:
    def call(self, args):
        # ---- 参数解包 ----
        # 所有反向传播所需的保存张量和上游梯度都在这里接收
        primals_1, primals_3, primals_7, primals_8, primals_9, \
        convolution, getitem_1, rsqrt, convert_element_type, getitem_3, \
        view, addmm, convert_element_type_1, tangents_1, tangents_2, tangents_3 = args
        args.clear()

        # ---- Shape Guard 验证 ----
        # TorchDynamo 在追踪时记录了所有 Tensor 的精确形状和内存步长 (stride)。
        # 在每次实际执行时，都会先验证输入是否与追踪时完全一致。
        # 若不一致 (如 batch_size 改变)，会触发重新追踪 (re-tracing)。
        assert_size_stride(primals_1, (16, 1, 3, 3), (9, 9, 3, 1))    # Conv weight
        assert_size_stride(primals_3, (4, 1, 28, 28), (784, 784, 28, 1)) # Input x
        assert_size_stride(primals_7, (16, ), (1, ))                   # BN gamma
        assert_size_stride(primals_8, (16, ), (1, ))                   # BN beta
        assert_size_stride(primals_9, (10, 3136), (3136, 1))           # FC weight
        assert_size_stride(convolution, (4, 16, 28, 28), (12544, 784, 28, 1))  # Conv output
        assert_size_stride(getitem_1, (1, 16, 1, 1), (16, 1, 1, 1))   # BN mean
        assert_size_stride(rsqrt, (1, 16, 1, 1), (16, 1, 1, 1))       # BN rsqrt(var)
        assert_size_stride(convert_element_type, (4, 16, 28, 28), (12544, 784, 28, 1)) # LIF1 spike
        assert_size_stride(getitem_3, (4, 16, 14, 14), (3200, 196, 14, 1)) # MaxPool indices
        assert_size_stride(view, (4, 3136), (3136, 1))                 # Flatten output
        assert_size_stride(addmm, (4, 10), (10, 1))                    # FC output (LIF2 input)
        assert_size_stride(convert_element_type_1, (4, 10), (10, 1))  # LIF2 spike
        assert_size_stride(tangents_1, (4, 10), (10, 1))               # upstream grad 1
        assert_size_stride(tangents_2, (4, 16, 28, 28), (12544, 784, 28, 1)) # upstream grad 2
        assert_size_stride(tangents_3, (4, 10), (10, 1))               # upstream grad 3

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)

            # ================================================================
            # [Step 0] LIF2 (fc层后的第二个LIF节点) 的 ATan 替代梯度
            # ================================================================
            # buf0 复用了 addmm 的显存（就地操作，节省一次 malloc）
            buf0 = addmm; del addmm  # reuse
            stream0 = get_raw_stream(0)
            # 调用 Kernel #0: 处理形状 [4, 10] 的小张量 (FC 层输出)
            triton_poi_fused_add_div_mul_neg_pow_reciprocal_rsub_sub_0.run(
                buf0, tangents_3, convert_element_type_1, tangents_1,
                40,  # xnumel = 4 * 10
                stream=stream0)
            del convert_element_type_1; del tangents_1; del tangents_3

            # ================================================================
            # [Step 1] 线性层反向传播: dX = grad_output @ W^T  (cuBLAS GEMM)
            # ================================================================
            buf1 = empty_strided_cuda((4, 3136), (3136, 1), torch.float32)
            extern_kernels.mm(buf0, primals_9, out=buf1)  # [4,10] @ [10,3136] = [4, 3136]
            del primals_9

            # ================================================================
            # [Step 2] MaxPool2d 反向 + BatchNorm 前向项 + LIF1 输入预处理
            # ================================================================
            buf4 = empty_strided_cuda((4, 16, 28, 28), (12544, 784, 28, 1), torch.float32)
            buf5 = empty_strided_cuda((4, 16, 28, 28), (12544, 784, 28, 1), torch.float32)
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_mul_neg_sub_view_1.run(
                convolution, getitem_1, rsqrt, primals_7, primals_8,
                getitem_3, buf1, tangents_2, buf4, buf5,
                50176,  # xnumel = 4 * 16 * 28 * 28
                stream=stream0)
            del buf1; del getitem_3; del primals_8

            # ================================================================
            # [Step 3] BatchNorm 反向: Reduction 统计量 (sum_2, sum_3, mul_31)
            # 这是一个 Reduction Kernel，需要跨 N*H*W 维度求和
            # ================================================================
            buf6 = empty_strided_cuda((16, ), (1, ), torch.float32)  # sum_2 [per-channel]
            buf7 = empty_strided_cuda((16, ), (1, ), torch.float32)  # sum_3 [per-channel]
            buf10 = empty_strided_cuda((16, ), (1, ), torch.float32) # mul_31 = sum_3 * rsqrt
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_add_div_mul_native_batch_norm_backward_pow_reciprocal_rsub_sub_2.run(
                tangents_2, convert_element_type, buf4, buf5, convolution,
                getitem_1, rsqrt, buf6, buf7, buf10,
                16, 3136,  # xnumel=16 (channels), r_numel=3136 (N*H*W per channel)
                stream=stream0)

            # ================================================================
            # [Step 4] ← 这就是 Level 3 分析的那个 Triton Kernel！
            # LIF1 ATan 替代梯度 + BatchNorm Elementwise 反向
            # ================================================================
            # buf8/buf9 复用 convert_element_type 的显存 (in-place 优化)
            buf8 = convert_element_type; del convert_element_type  # reuse
            buf9 = buf8; del buf8  # reuse
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_add_div_mul_native_batch_norm_backward_pow_reciprocal_rsub_sub_3.run(
                buf9,           # in_out_ptr0: 读入 spike, 写出梯度
                tangents_2,     # in_ptr0: 上游梯度
                buf4,           # in_ptr1: BN 输出 (LIF 输入 v)
                buf5,           # in_ptr2: 另一中间量
                convolution,    # in_ptr3: Conv 原始输出 (用于 BN backward)
                getitem_1,      # in_ptr4: BN 均值 [16]
                buf7,           # in_ptr5: sum_2 [16]
                rsqrt,          # in_ptr6: BN rsqrt [16]
                buf6,           # in_ptr7: sum_3 [16]
                primals_7,      # in_ptr8: BN gamma [16]
                50176,          # xnumel = 4 * 16 * 28 * 28
                stream=stream0)
            del buf4; del buf5; del convolution; del getitem_1
            del primals_7; del rsqrt; del tangents_2

            # ================================================================
            # [Step 5] Conv2d 反向传播 (使用 cuDNN/CUDA 内置算子)
            # 计算 dW_conv (卷积权重梯度) 和 dX (到输入的梯度)
            # ================================================================
            buf12 = torch.ops.aten.convolution_backward.default(
                buf9, primals_3, primals_1, [16],
                [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
                [False, True, False])  # 只需要 dW, 不需要 dX (输入无参数)
            del primals_1; del primals_3
            buf13 = buf12[1]  # dW_conv: shape [16, 1, 3, 3]
            del buf12

            # ================================================================
            # [Step 6, 7, 8] FC 层相关梯度 (bias, weight) + Conv bias 梯度
            # ================================================================
            buf3 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
            triton_poi_fused_sum_4.run(buf0, buf3, 10, stream=stream0)  # FC bias 梯度

            buf2 = empty_strided_cuda((10, 3136), (3136, 1), torch.float32)
            extern_kernels.mm(reinterpret_tensor(buf0, (10, 4), (1, 10), 0),
                              view, out=buf2)  # FC weight 梯度: grad.T @ x
            del buf0; del view

            buf11 = buf7; del buf7  # reuse
            triton_red_fused_convolution_backward_5.run(
                buf9, buf11, 16, 3136, stream=stream0)  # Conv bias 梯度
            del buf9

        # ---- 返回所有梯度 ----
        # 顺序对应: d_conv_weight, d_bn_gamma_related, None*5, d_bn_bias, d_fc_weight, d_fc_bias
        return (buf13, buf11, None, None, None, None, buf10, buf6, buf2,
                reinterpret_tensor(buf3, (10, ), (1, ), 0), )
