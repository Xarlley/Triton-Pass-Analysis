# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/ka/ckatjdwch3qex52h6marxcqjlgll2w7oyt2fvymmfdc3xnup3hvv.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type_1
# Graph fragment:
#   %arg1_1 : Tensor "f32[8, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type_1 : Tensor "bf16[8, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4816896, 'x': 4816896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
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
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/fq/cfqp2ncwsskhxr7nfhbokns43g2hbj67ro5rdliuiy6f24rhycfm.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 37632, 'x': 37632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x2 + 147*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/bv/cbvvxhplq5fxbv4xqkcrv5gvoodyl7djzotaq3fweygpxtwu7bbx.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   x => convert_element_type, convert_element_type_1, convolution
# Graph fragment:
#   %convert_element_type_1 : Tensor "bf16[8, 3, 224, 224][150528, 1, 672, 3]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %convert_element_type : Tensor "bf16[64, 3, 7, 7][147, 1, 21, 3]cuda:0" = PlaceHolder[target=convert_element_type]
#   %convert_element_type_1 : Tensor "bf16[8, 3, 224, 224][150528, 50176, 224, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
#   %convert_element_type : Tensor "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_1, %convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_tem_fused__to_copy_convolution_2 = async_compile.triton('triton_tem_fused__to_copy_convolution_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=4,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_2', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 7, 'KERNEL_W': 7, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 3, 'PADDING_W': 3, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_2(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 7
    KERNEL_W : tl.constexpr = 7
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 3
    PADDING_W : tl.constexpr = 3
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 8
    IN_C = 3
    IN_H = 224
    IN_W = 224
    OUT_C = 64
    OUT_H = 112
    OUT_W = 112

    # Strides:
    stride_xn = 150528
    stride_xc = 1
    stride_xh = 672
    stride_xw = 3
    stride_wc_out = 147
    stride_wc_in = 1
    stride_wh = 21
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
    xindex = idx_w + 112*idx_h + 12544*idx_c + 802816*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/gw/cgwstgytebr6gpcoficev5xcotpw2bzjg5hocmenadi4gtjt4re7.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_1 => add, add_1, convert_element_type_4, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
# Graph fragment:
#   %convolution : Tensor "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_4
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 38535168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 12544) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 1e-05, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = tmp9 * tmp8
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/62/c62alvwaf7cccmvdte2ldsb7xmeoo2tca6itcuys7skz5tvoxffe.py
# Topologically Sorted Source Nodes: [x_1, unsqueeze_, x_2, full_like, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.unsqueeze, aten.repeat, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_35
#   full_like => full_default
#   unsqueeze_ => unsqueeze_8
#   x_1 => add, add_1, convert_element_type_4, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => repeat
# Graph fragment:
#   %convert_element_type_4 : Tensor "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = PlaceHolder[target=convert_element_type_4]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_8 : Tensor "bf16[1, 8, 64, 112, 112][6422528, 802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_4, 0), kwargs = {})
#   %repeat : Tensor "bf16[4, 8, 64, 112, 112][6422528, 802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze_8, [4, 1, 1, 1, 1]), kwargs = {})
#   %full_default : Tensor "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 64, 112, 112], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_35 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 36, grid: [(50176, 1, 1), (25088, 1, 1), (25088, 1, 1), (12544, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %repeat, v_init_ptr: %full_default, s_seq_ptr: %empty_1, h_seq_ptr: %empty_2, v_seq_ptr: %empty_2, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 6422528, soft_reset: False, save_intermediates: False}})
#   return %buf6
triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 115605504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 6422528)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/hp/chpamr5nqrxuza7z6m7lamgj2tmbfs763pyapqmyvjkzdud7vret.py
# Topologically Sorted Source Nodes: [x_1, unsqueeze_, x_2, full_like, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.unsqueeze, aten.repeat, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_35
#   full_like => full_default
#   unsqueeze_ => unsqueeze_8
#   x_1 => add, add_1, convert_element_type_4, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => repeat
# Graph fragment:
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_8 : Tensor "bf16[1, 8, 64, 112, 112][6422528, 802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_4, 0), kwargs = {})
#   %repeat : Tensor "bf16[4, 8, 64, 112, 112][6422528, 802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze_8, [4, 1, 1, 1, 1]), kwargs = {})
#   %full_default : Tensor "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 64, 112, 112], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_35 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 36, grid: [(50176, 1, 1), (25088, 1, 1), (25088, 1, 1), (12544, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %repeat, v_init_ptr: %full_default, s_seq_ptr: %empty_1, h_seq_ptr: %empty_2, v_seq_ptr: %empty_2, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 6422528, soft_reset: False, save_intermediates: False}})
#   return %buf7
triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 25690112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py:25
_multistep_if_forward_kernel_0 = async_compile.triton('_multistep_if_forward_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['50176', '1', '1'], 'cpp': ['50176L', '1L', '1L'], 'python_slow': ['50176', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['25088', '1', '1'], 'cpp': ['25088L', '1L', '1L'], 'python_slow': ['25088', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['25088', '1', '1'], 'cpp': ['25088L', '1L', '1L'], 'python_slow': ['25088', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['12544', '1', '1'], 'cpp': ['12544L', '1L', '1L'], 'python_slow': ['12544', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_if_forward_kernel_0', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 4, 'NCL': 6422528, 'dtype': triton.language.bfloat16, 'soft_reset': False, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _multistep_if_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v_init_ptrs = tl.make_block_ptr(
        v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        h = v + x
        s = (h >= v_threshold).to(dtype)
        if soft_reset:
            v = h - s * v_threshold
        else:
            v = s * v_reset + (1.0 - s) * h

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(s_ptrs, s, boundary_check=(1,))
        v_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(h_ptrs, h, boundary_check=(1,))

@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/fy/cfyo5mguo724xk7figulf34hdp7aaoo3jhs7z3nfdhy3swck3kov.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.view, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => _low_memory_max_pool_with_offsets, getitem_2, view_1
# Graph fragment:
#   %buf8 : Tensor  = PlaceHolder[target=buf8]
#   %view_1 : Tensor "bf16[32, 64, 112, 112][802816, 12544, 112, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%empty_1, [32, 64, 112, 112]), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%view_1, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_2 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 0), kwargs = {})
#   return %getitem_2
triton_poi_fused_max_pool2d_with_indices_view_6 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 25690112, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_view_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex // 56
    x2 = (xindex % 56)
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = (-1) + 2*x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + 2*x2 + 224*x3 + 12544*y4), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp12 = 2*x2
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-112) + 2*x2 + 224*x3 + 12544*y4), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x2
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-111) + 2*x2 + 224*x3 + 12544*y4), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x3
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x2 + 224*x3 + 12544*y4), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x2 + 224*x3 + 12544*y4), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x2 + 224*x3 + 12544*y4), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x3
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (111 + 2*x2 + 224*x3 + 12544*y4), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (112 + 2*x2 + 224*x3 + 12544*y4), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (113 + 2*x2 + 224*x3 + 12544*y4), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (y0 + 64*x5 + 200704*y1), tmp51, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/gc/cgc534muj24mgfhwfsfi6nootpkrhg5rf3fub6jtxlas42s4lwqf.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_5
# Graph fragment:
#   %arg6_1 : Tensor "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convert_element_type_5 : Tensor "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_5
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 147456, 'x': 147456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/5c/c5cvviq7s7dn7e544j7eszifkmpwenqjo3dwuwr2htmykncxc6a5.py
# Topologically Sorted Source Nodes: [x_3, y_1, input_2, input_3, view_1, full_like_1, ], Original ATen: [aten.view, aten._to_copy, aten.convolution, aten._native_batch_norm_legit_no_training, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_34
#   full_like_1 => full_default_1
#   input_2 => convert_element_type_5, convolution_1
#   input_3 => add_2, add_3, convert_element_type_8, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_16, unsqueeze_9
#   view_1 => view_4
#   x_3 => view_2
#   y_1 => view_3
# Graph fragment:
#   %getitem_2 : Tensor "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=getitem_2]
#   %convert_element_type_5 : Tensor "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = PlaceHolder[target=convert_element_type_5]
#   %convolution_1 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = PlaceHolder[target=convolution_1]
#   %arg7_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %view_2 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [4, 8, 64, 56, 56]), kwargs = {})
#   %view_3 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_2, [32, 64, 56, 56]), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_3, %convert_element_type_5, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, -1), kwargs = {})
#   %sub_1 : Tensor "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_10), kwargs = {})
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, -1), kwargs = {})
#   %mul_4 : Tensor "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_12), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, -1), kwargs = {})
#   %mul_5 : Tensor "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_14), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_15, -1), kwargs = {})
#   %add_3 : Tensor "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_16), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %view_4 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_8, [4, 8, 64, 56, 56]), kwargs = {})
#   %full_default_1 : Tensor "bf16[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 64, 56, 56], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_34 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 37, grid: [(12544, 1, 1), (6272, 1, 1), (6272, 1, 1), (3136, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %view_4, v_init_ptr: %full_default_1, s_seq_ptr: %empty_4, h_seq_ptr: %empty_5, v_seq_ptr: %empty_5, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 1605632, soft_reset: False, save_intermediates: False}})
#   return %convolution_1,%buf15
triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8 = async_compile.triton('triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8(arg_X, arg_W, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1):
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
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 64
    IN_H = 56
    IN_W = 56
    OUT_C = 64
    OUT_H = 56
    OUT_W = 56

    # Strides:
    stride_xn = 200704
    stride_xc = 1
    stride_xh = 3584
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
    xindex = idx_w + 56*idx_h + 3136*idx_c + 200704*idx_n
    x5 = (xindex % 3136)
    tmp1 = tl.load(in_ptr2 + (tl.broadcast_to(idx_c, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (tl.broadcast_to(idx_c, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(idx_c, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (tl.broadcast_to(idx_c, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
    tmp0 = acc.to(tl.float32)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr1 + (x5 + 3136*idx_c + 200704*idx_n), tmp15, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/vh/cvhnhvjbwg75ftintbqc2lrdrgssdhu5t54vdfk4qyrrpmzgirkp.py
# Topologically Sorted Source Nodes: [input_3, view_1, full_like_1, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_34
#   full_like_1 => full_default_1
#   input_3 => add_2, add_3, convert_element_type_8, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_16, unsqueeze_9
#   view_1 => view_4
# Graph fragment:
#   %unsqueeze_9 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, -1), kwargs = {})
#   %sub_1 : Tensor "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_10), kwargs = {})
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, -1), kwargs = {})
#   %mul_4 : Tensor "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_12), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, -1), kwargs = {})
#   %mul_5 : Tensor "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_14), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_15, -1), kwargs = {})
#   %add_3 : Tensor "f32[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_16), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %view_4 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_8, [4, 8, 64, 56, 56]), kwargs = {})
#   %full_default_1 : Tensor "bf16[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 64, 56, 56], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_34 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 37, grid: [(12544, 1, 1), (6272, 1, 1), (6272, 1, 1), (3136, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %view_4, v_init_ptr: %full_default_1, s_seq_ptr: %empty_4, h_seq_ptr: %empty_5, v_seq_ptr: %empty_5, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 1605632, soft_reset: False, save_intermediates: False}})
#   return %buf16
triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6422528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py:25
_multistep_if_forward_kernel_1 = async_compile.triton('_multistep_if_forward_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['12544', '1', '1'], 'cpp': ['12544L', '1L', '1L'], 'python_slow': ['12544', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['6272', '1', '1'], 'cpp': ['6272L', '1L', '1L'], 'python_slow': ['6272', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['6272', '1', '1'], 'cpp': ['6272L', '1L', '1L'], 'python_slow': ['6272', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['3136', '1', '1'], 'cpp': ['3136L', '1L', '1L'], 'python_slow': ['3136', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_if_forward_kernel_1', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 4, 'NCL': 1605632, 'dtype': triton.language.bfloat16, 'soft_reset': False, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _multistep_if_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v_init_ptrs = tl.make_block_ptr(
        v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        h = v + x
        s = (h >= v_threshold).to(dtype)
        if soft_reset:
            v = h - s * v_threshold
        else:
            v = s * v_reset + (1.0 - s) * h

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(s_ptrs, s, boundary_check=(1,))
        v_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(h_ptrs, h, boundary_check=(1,))

@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/ld/cldhhl5kahoaqyi7vdgbyh6djlinxqmmyn6khxrfkfbmazknwuhn.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_9, convolution_2, view_6
# Graph fragment:
#   %buf17 : Tensor  = PlaceHolder[target=buf17]
#   %view_6 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%empty_4, [32, 64, 56, 56]), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg11_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_6, %convert_element_type_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf20
triton_poi_fused__to_copy_convolution_view_10 = async_compile.triton('triton_poi_fused__to_copy_convolution_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 25690112, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_view_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 64*x2 + 200704*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/u3/cu34rm5cauuqhjbumkzs7rgpeguinj7xj6aeyru46ueq7vtkzpmp.py
# Topologically Sorted Source Nodes: [x_3, out, input_6], Original ATen: [aten.view, aten.add, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_6 => convert_element_type_13, convolution_3, view_9
#   out => add_6
#   x_3 => view_2
# Graph fragment:
#   %buf26 : Tensor  = PlaceHolder[target=buf26]
#   %getitem_2 : Tensor "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=getitem_2]
#   %add_6 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0" = PlaceHolder[target=add_6]
#   %view_2 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [4, 8, 64, 56, 56]), kwargs = {})
#   %add_6 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_7, %view_2), kwargs = {})
#   %view_9 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [32, 64, 56, 56]), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg16_1, torch.bfloat16), kwargs = {})
#   %convolution_3 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_9, %convert_element_type_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_6,%buf30
triton_poi_fused__to_copy_add_convolution_view_11 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 38535168, 'x': 38535168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_11(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0 + 64*x2 + 200704*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp2, xmask)
    tl.store(out_ptr1 + (y0 + 64*x2 + 200704*y1), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/oo/coo5noxwr3tbv22pgtuh2ggi63qnvfwicqc7rqbwph6yyzcctvbq.py
# Topologically Sorted Source Nodes: [out_1, input_10], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_10 => convert_element_type_21, convolution_5, view_15
#   out_1 => add_11
# Graph fragment:
#   %buf45 : Tensor  = PlaceHolder[target=buf45]
#   %add_6 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0" = PlaceHolder[target=add_6]
#   %add_11 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0" = PlaceHolder[target=add_11]
#   %add_11 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_13, %add_6), kwargs = {})
#   %view_15 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_11, [32, 64, 56, 56]), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg26_1, torch.bfloat16), kwargs = {})
#   %convolution_5 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_15, %convert_element_type_21, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_11,%buf49
triton_poi_fused__to_copy_add_convolution_view_12 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 51380224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_12(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 3136)
    x2 = ((xindex // 3136) % 64)
    x3 = xindex // 200704
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr0 + (x2 + 64*x1 + 200704*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/zb/czbixajmhhrv45ibhs5l53kuuec3uaxw2espn3dl7p4hkmyecdkl.py
# Topologically Sorted Source Nodes: [out_2, input_14, input_18], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_14 => convert_element_type_29, convolution_7, view_21
#   input_18 => convert_element_type_37, convolution_9, view_27
#   out_2 => add_16
# Graph fragment:
#   %buf64 : Tensor  = PlaceHolder[target=buf64]
#   %add_11 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0" = PlaceHolder[target=add_11]
#   %add_16 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0" = PlaceHolder[target=add_16]
#   %add_16 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_19, %add_11), kwargs = {})
#   %view_21 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_16, [32, 64, 56, 56]), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg36_1, torch.bfloat16), kwargs = {})
#   %convolution_7 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_21, %convert_element_type_29, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_27 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_16, [32, 64, 56, 56]), kwargs = {})
#   %convert_element_type_37 : Tensor "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg46_1, torch.bfloat16), kwargs = {})
#   %convolution_9 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_27, %convert_element_type_37, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_16,%buf68,%buf86
triton_poi_fused__to_copy_add_convolution_view_13 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 12845056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_13(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 3136)
    x2 = ((xindex // 3136) % 64)
    x3 = xindex // 200704
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 64*x1 + 200704*x3), tmp2, None)
    tl.store(out_ptr1 + (x2 + 64*x1 + 200704*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/ke/cke6ebzi537sq3jvi3e7uidjpuy22zfhmf5pz2jlg7cgubjdyr4r.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_14 => convert_element_type_29
# Graph fragment:
#   %arg36_1 : Tensor "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg36_1]
#   %convert_element_type_29 : Tensor "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg36_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_29
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 294912, 'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/bx/cbx6krcl7xneaptm7aaj6xdepgaci2qbqvjvcsnmgsi6yaq6jmag.py
# Topologically Sorted Source Nodes: [out_2, input_14], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_14 => convert_element_type_29, convolution_7, view_21
#   out_2 => add_16
# Graph fragment:
#   %buf68 : Tensor "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=buf68]
#   %convert_element_type_29 : Tensor "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = PlaceHolder[target=convert_element_type_29]
#   %add_16 : Tensor "bf16[4, 8, 64, 56, 56][1605632, 200704, 3136, 56, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_19, %add_11), kwargs = {})
#   %view_21 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_16, [32, 64, 56, 56]), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg36_1, torch.bfloat16), kwargs = {})
#   %convolution_7 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_21, %convert_element_type_29, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_7
triton_tem_fused__to_copy_add_convolution_view_15 = async_compile.triton('triton_tem_fused__to_copy_add_convolution_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_convolution_view_15', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_add_convolution_view_15(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 64
    IN_H = 56
    IN_W = 56
    OUT_C = 128
    OUT_H = 28
    OUT_W = 28

    # Strides:
    stride_xn = 200704
    stride_xc = 1
    stride_xh = 3584
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
    xindex = idx_w + 28*idx_h + 784*idx_c + 100352*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/be/cbex224ngmhw2u5rqx7ayh46smwkybcnko7kq7diqajyftrgnzrv.py
# Topologically Sorted Source Nodes: [input_15, view_7, full_like_7, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_28
#   full_like_7 => full_default_7
#   input_15 => add_17, add_18, convert_element_type_32, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63, unsqueeze_64
#   view_7 => view_22
# Graph fragment:
#   %convolution_7 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0" = PlaceHolder[target=convolution_7]
#   %arg37_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %unsqueeze_57 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_57, -1), kwargs = {})
#   %sub_7 : Tensor "f32[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_58), kwargs = {})
#   %add_17 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_17,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_59, -1), kwargs = {})
#   %mul_22 : Tensor "f32[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_60), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_61, -1), kwargs = {})
#   %mul_23 : Tensor "f32[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_62), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_64 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_63, -1), kwargs = {})
#   %add_18 : Tensor "f32[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_64), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_18, torch.bfloat16), kwargs = {})
#   %view_22 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_32, [4, 8, 128, 28, 28]), kwargs = {})
#   %full_default_7 : Tensor "bf16[8, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 128, 28, 28], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_28 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 43, grid: [(6272, 1, 1), (3136, 1, 1), (3136, 1, 1), (1568, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %view_22, v_init_ptr: %full_default_7, s_seq_ptr: %empty_22, h_seq_ptr: %empty_23, v_seq_ptr: %empty_23, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 802816, soft_reset: False, save_intermediates: False}})
#   return %buf72
triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 19267584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 784) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 1e-05, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = tmp9 * tmp8
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/bg/cbgdwqdcvcqq7m4ntfrazg7kwyujt2t74rhivwa42khxsrufrtmt.py
# Topologically Sorted Source Nodes: [input_15, view_7, full_like_7, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_28
#   full_like_7 => full_default_7
#   input_15 => add_17, add_18, convert_element_type_32, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63, unsqueeze_64
#   view_7 => view_22
# Graph fragment:
#   %unsqueeze_57 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_57, -1), kwargs = {})
#   %sub_7 : Tensor "f32[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_58), kwargs = {})
#   %add_17 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_17,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_59, -1), kwargs = {})
#   %mul_22 : Tensor "f32[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_60), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_61, -1), kwargs = {})
#   %mul_23 : Tensor "f32[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_62), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_64 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_63, -1), kwargs = {})
#   %add_18 : Tensor "f32[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_64), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_18, torch.bfloat16), kwargs = {})
#   %view_22 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_32, [4, 8, 128, 28, 28]), kwargs = {})
#   %full_default_7 : Tensor "bf16[8, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 128, 28, 28], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_28 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 43, grid: [(6272, 1, 1), (3136, 1, 1), (3136, 1, 1), (1568, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %view_22, v_init_ptr: %full_default_7, s_seq_ptr: %empty_22, h_seq_ptr: %empty_23, v_seq_ptr: %empty_23, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 802816, soft_reset: False, save_intermediates: False}})
#   return %buf73
triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3211264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py:25
_multistep_if_forward_kernel_2 = async_compile.triton('_multistep_if_forward_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['6272', '1', '1'], 'cpp': ['6272L', '1L', '1L'], 'python_slow': ['6272', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['3136', '1', '1'], 'cpp': ['3136L', '1L', '1L'], 'python_slow': ['3136', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['3136', '1', '1'], 'cpp': ['3136L', '1L', '1L'], 'python_slow': ['3136', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['1568', '1', '1'], 'cpp': ['1568L', '1L', '1L'], 'python_slow': ['1568', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_if_forward_kernel_2', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 4, 'NCL': 802816, 'dtype': triton.language.bfloat16, 'soft_reset': False, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _multistep_if_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v_init_ptrs = tl.make_block_ptr(
        v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        h = v + x
        s = (h >= v_threshold).to(dtype)
        if soft_reset:
            v = h - s * v_threshold
        else:
            v = s * v_reset + (1.0 - s) * h

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(s_ptrs, s, boundary_check=(1,))
        v_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(h_ptrs, h, boundary_check=(1,))

@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/rs/crsqcgqypxaqwgfnkzo56djvwkcph7rvtnwuav4e7kdsyub4xmpo.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_16 => convert_element_type_33
# Graph fragment:
#   %arg41_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg41_1]
#   %convert_element_type_33 : Tensor "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_33
triton_poi_fused__to_copy_18 = async_compile.triton('triton_poi_fused__to_copy_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 589824, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/gi/cgidpvpcq35fljjfbnbaetnehc26hslmkoe44xyj2e65yzirvh3e.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_16 => convert_element_type_33, convolution_8, view_24
# Graph fragment:
#   %buf74 : Tensor  = PlaceHolder[target=buf74]
#   %view_24 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%empty_22, [32, 128, 28, 28]), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %convolution_8 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_24, %convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf77
triton_poi_fused__to_copy_convolution_view_19 = async_compile.triton('triton_poi_fused__to_copy_convolution_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 12845056, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_view_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + 784*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 100352*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/qx/cqxlwknsmwhv5azba4zpjumuuiap7udk6cbwefzu6aafqsfdw7nq.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_16 => convert_element_type_33, convolution_8, view_24
# Graph fragment:
#   %buf77 : Tensor "bf16[32, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf77]
#   %convert_element_type_33 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=convert_element_type_33]
#   %view_24 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%empty_22, [32, 128, 28, 28]), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg41_1, torch.bfloat16), kwargs = {})
#   %convolution_8 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_24, %convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_8
triton_tem_fused__to_copy_convolution_view_20 = async_compile.triton('triton_tem_fused__to_copy_convolution_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_view_20', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_view_20(arg_X, arg_W, out_ptr0):
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
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 128
    IN_H = 28
    IN_W = 28
    OUT_C = 128
    OUT_H = 28
    OUT_W = 28

    # Strides:
    stride_xn = 100352
    stride_xc = 1
    stride_xh = 3584
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
    xindex = idx_w + 28*idx_h + 784*idx_c + 100352*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/ig/cigautiky6ob5hao2stfrzeptkaseqnjnso2im75ptbdg2zlgen7.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_18 => convert_element_type_37
# Graph fragment:
#   %arg46_1 : Tensor "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=arg46_1]
#   %convert_element_type_37 : Tensor "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg46_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_37
triton_poi_fused__to_copy_21 = async_compile.triton('triton_poi_fused__to_copy_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/bd/cbdrjhweurnhwe6gq4gyekejy2epdpxjrlg7rhyuq3ufgron5of2.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_18 => convert_element_type_37, convolution_9, view_27
# Graph fragment:
#   %buf86 : Tensor "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=buf86]
#   %convert_element_type_37 : Tensor "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=convert_element_type_37]
#   %view_27 : Tensor "bf16[32, 64, 56, 56][200704, 3136, 56, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_16, [32, 64, 56, 56]), kwargs = {})
#   %convert_element_type_37 : Tensor "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg46_1, torch.bfloat16), kwargs = {})
#   %convolution_9 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_27, %convert_element_type_37, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_9
triton_tem_fused__to_copy_convolution_view_22 = async_compile.triton('triton_tem_fused__to_copy_convolution_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_view_22', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 1, 'UNROLL': True, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_view_22(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 64
    IN_H = 56
    IN_W = 56
    OUT_C = 128
    OUT_H = 28
    OUT_W = 28

    # Strides:
    stride_xn = 200704
    stride_xc = 1
    stride_xh = 3584
    stride_xw = 64
    stride_wc_out = 64
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

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




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

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
    xindex = idx_w + 28*idx_h + 784*idx_c + 100352*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/4g/c4g7geuz25rr6ftcpq5srsp46mttsdxaue7qcihqqiypk7puprcm.py
# Topologically Sorted Source Nodes: [out_3, input_21], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_21 => convert_element_type_41, convolution_10, view_30
#   out_3 => add_23
# Graph fragment:
#   %buf83 : Tensor  = PlaceHolder[target=buf83]
#   %buf92 : Tensor  = PlaceHolder[target=buf92]
#   %add_23 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0" = PlaceHolder[target=add_23]
#   %add_23 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_25, %empty_28), kwargs = {})
#   %view_30 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_23, [32, 128, 28, 28]), kwargs = {})
#   %convert_element_type_41 : Tensor "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg51_1, torch.bfloat16), kwargs = {})
#   %convolution_10 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_30, %convert_element_type_41, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_23,%buf96
triton_poi_fused__to_copy_add_convolution_view_23 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 19267584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_23(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 784)
    x2 = ((xindex // 784) % 128)
    x3 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x2 + 128*x1 + 100352*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/vh/cvhhm3rqtkl7uvop3p6dsws3ymth7qvmwbjb23moedgddhnxeltm.py
# Topologically Sorted Source Nodes: [out_4, input_25], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_25 => convert_element_type_49, convolution_12, view_36
#   out_4 => add_28
# Graph fragment:
#   %buf111 : Tensor  = PlaceHolder[target=buf111]
#   %add_23 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0" = PlaceHolder[target=add_23]
#   %add_28 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0" = PlaceHolder[target=add_28]
#   %add_28 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_34, %add_23), kwargs = {})
#   %view_36 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_28, [32, 128, 28, 28]), kwargs = {})
#   %convert_element_type_49 : Tensor "bf16[128, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg61_1, torch.bfloat16), kwargs = {})
#   %convolution_12 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_36, %convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_28,%buf115
triton_poi_fused__to_copy_add_convolution_view_24 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 25690112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_24(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 784)
    x2 = ((xindex // 784) % 128)
    x3 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr0 + (x2 + 128*x1 + 100352*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/z2/cz2mcr7uho4axmjx3rtrx2evlnorujvor4nga6ipcsftpidxbtuk.py
# Topologically Sorted Source Nodes: [out_6, input_33, input_37], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_33 => convert_element_type_65, convolution_16, view_48
#   input_37 => convert_element_type_73, convolution_18, view_54
#   out_6 => add_38
# Graph fragment:
#   %buf149 : Tensor  = PlaceHolder[target=buf149]
#   %add_33 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0" = PlaceHolder[target=add_33]
#   %add_38 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0" = PlaceHolder[target=add_38]
#   %add_38 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_46, %add_33), kwargs = {})
#   %view_48 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_38, [32, 128, 28, 28]), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg81_1, torch.bfloat16), kwargs = {})
#   %convolution_16 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_48, %convert_element_type_65, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_54 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_38, [32, 128, 28, 28]), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg91_1, torch.bfloat16), kwargs = {})
#   %convolution_18 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_54, %convert_element_type_73, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_38,%buf153,%buf171
triton_poi_fused__to_copy_add_convolution_view_25 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6422528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_25(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 784)
    x2 = ((xindex // 784) % 128)
    x3 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 128*x1 + 100352*x3), tmp2, None)
    tl.store(out_ptr1 + (x2 + 128*x1 + 100352*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/25/c25tt6owba7oaao2a2oopj4mv3c3vhhfvwimipeagbv6qdiiz4fg.py
# Topologically Sorted Source Nodes: [input_33], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_33 => convert_element_type_65
# Graph fragment:
#   %arg81_1 : Tensor "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg81_1]
#   %convert_element_type_65 : Tensor "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg81_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_65
triton_poi_fused__to_copy_26 = async_compile.triton('triton_poi_fused__to_copy_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 1179648, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/af/cafryjbws7wgjx5utk5d26ebtimja7ltoo7dghq7nufbff5qmytp.py
# Topologically Sorted Source Nodes: [out_6, input_33], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_33 => convert_element_type_65, convolution_16, view_48
#   out_6 => add_38
# Graph fragment:
#   %buf153 : Tensor "bf16[32, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf153]
#   %convert_element_type_65 : Tensor "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=convert_element_type_65]
#   %add_38 : Tensor "bf16[4, 8, 128, 28, 28][802816, 100352, 784, 28, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_46, %add_33), kwargs = {})
#   %view_48 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_38, [32, 128, 28, 28]), kwargs = {})
#   %convert_element_type_65 : Tensor "bf16[256, 128, 3, 3][1152, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg81_1, torch.bfloat16), kwargs = {})
#   %convolution_16 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_48, %convert_element_type_65, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_16
triton_tem_fused__to_copy_add_convolution_view_27 = async_compile.triton('triton_tem_fused__to_copy_add_convolution_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_convolution_view_27', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_add_convolution_view_27(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 128
    IN_H = 28
    IN_W = 28
    OUT_C = 256
    OUT_H = 14
    OUT_W = 14

    # Strides:
    stride_xn = 100352
    stride_xc = 1
    stride_xh = 3584
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
    xindex = idx_w + 14*idx_h + 196*idx_c + 50176*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/fk/cfkm2vtcfpknl6tn3hnith2c3tedu2xm4biet42jluwny4nb2lp7.py
# Topologically Sorted Source Nodes: [input_34, view_16, full_like_16, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_19
#   full_like_16 => full_default_16
#   input_34 => add_39, add_40, convert_element_type_68, mul_48, mul_49, mul_50, reciprocal_16, sqrt_16, sub_16, unsqueeze_129, unsqueeze_130, unsqueeze_131, unsqueeze_132, unsqueeze_133, unsqueeze_134, unsqueeze_135, unsqueeze_136
#   view_16 => view_49
# Graph fragment:
#   %convolution_16 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0" = PlaceHolder[target=convolution_16]
#   %arg82_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg82_1]
#   %arg83_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg83_1]
#   %arg84_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg84_1]
#   %arg85_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg85_1]
#   %unsqueeze_129 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg82_1, -1), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_129, -1), kwargs = {})
#   %sub_16 : Tensor "f32[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_130), kwargs = {})
#   %add_39 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg83_1, 1e-05), kwargs = {})
#   %sqrt_16 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_39,), kwargs = {})
#   %reciprocal_16 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_16,), kwargs = {})
#   %mul_48 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 1), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_48, -1), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_131, -1), kwargs = {})
#   %mul_49 : Tensor "f32[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_132), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg84_1, -1), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_133, -1), kwargs = {})
#   %mul_50 : Tensor "f32[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_134), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg85_1, -1), kwargs = {})
#   %unsqueeze_136 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_135, -1), kwargs = {})
#   %add_40 : Tensor "f32[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_136), kwargs = {})
#   %convert_element_type_68 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_40, torch.bfloat16), kwargs = {})
#   %view_49 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_68, [4, 8, 256, 14, 14]), kwargs = {})
#   %full_default_16 : Tensor "bf16[8, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 256, 14, 14], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_19 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 52, grid: [(3136, 1, 1), (1568, 1, 1), (1568, 1, 1), (784, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %view_49, v_init_ptr: %full_default_16, s_seq_ptr: %empty_49, h_seq_ptr: %empty_50, v_seq_ptr: %empty_50, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 401408, soft_reset: False, save_intermediates: False}})
#   return %buf157
triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 9633792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 196) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 1e-05, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = tmp9 * tmp8
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/7l/c7lpvacaiqsbyaka3y5wm5en5bab4zr2peihylg24vst5v4vdqrm.py
# Topologically Sorted Source Nodes: [input_34, view_16, full_like_16, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_19
#   full_like_16 => full_default_16
#   input_34 => add_39, add_40, convert_element_type_68, mul_48, mul_49, mul_50, reciprocal_16, sqrt_16, sub_16, unsqueeze_129, unsqueeze_130, unsqueeze_131, unsqueeze_132, unsqueeze_133, unsqueeze_134, unsqueeze_135, unsqueeze_136
#   view_16 => view_49
# Graph fragment:
#   %unsqueeze_129 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg82_1, -1), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_129, -1), kwargs = {})
#   %sub_16 : Tensor "f32[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_130), kwargs = {})
#   %add_39 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg83_1, 1e-05), kwargs = {})
#   %sqrt_16 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_39,), kwargs = {})
#   %reciprocal_16 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_16,), kwargs = {})
#   %mul_48 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 1), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_48, -1), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_131, -1), kwargs = {})
#   %mul_49 : Tensor "f32[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_132), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg84_1, -1), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_133, -1), kwargs = {})
#   %mul_50 : Tensor "f32[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_134), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg85_1, -1), kwargs = {})
#   %unsqueeze_136 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_135, -1), kwargs = {})
#   %add_40 : Tensor "f32[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_136), kwargs = {})
#   %convert_element_type_68 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_40, torch.bfloat16), kwargs = {})
#   %view_49 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_68, [4, 8, 256, 14, 14]), kwargs = {})
#   %full_default_16 : Tensor "bf16[8, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 256, 14, 14], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_19 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 52, grid: [(3136, 1, 1), (1568, 1, 1), (1568, 1, 1), (784, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %view_49, v_init_ptr: %full_default_16, s_seq_ptr: %empty_49, h_seq_ptr: %empty_50, v_seq_ptr: %empty_50, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 401408, soft_reset: False, save_intermediates: False}})
#   return %buf158
triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1605632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py:25
_multistep_if_forward_kernel_3 = async_compile.triton('_multistep_if_forward_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['3136', '1', '1'], 'cpp': ['3136L', '1L', '1L'], 'python_slow': ['3136', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['1568', '1', '1'], 'cpp': ['1568L', '1L', '1L'], 'python_slow': ['1568', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['1568', '1', '1'], 'cpp': ['1568L', '1L', '1L'], 'python_slow': ['1568', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['784', '1', '1'], 'cpp': ['784L', '1L', '1L'], 'python_slow': ['784', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_if_forward_kernel_3', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 4, 'NCL': 401408, 'dtype': triton.language.bfloat16, 'soft_reset': False, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _multistep_if_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v_init_ptrs = tl.make_block_ptr(
        v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        h = v + x
        s = (h >= v_threshold).to(dtype)
        if soft_reset:
            v = h - s * v_threshold
        else:
            v = s * v_reset + (1.0 - s) * h

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(s_ptrs, s, boundary_check=(1,))
        v_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(h_ptrs, h, boundary_check=(1,))

@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/xt/cxtoc2chrwzaiw4wajw2wxkrpprnj4milorfhzjxkpz7ai5hz64y.py
# Topologically Sorted Source Nodes: [input_35], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_35 => convert_element_type_69
# Graph fragment:
#   %arg86_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg86_1]
#   %convert_element_type_69 : Tensor "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg86_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_69
triton_poi_fused__to_copy_30 = async_compile.triton('triton_poi_fused__to_copy_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 2359296, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/rp/crppdrtyxv5odxwkbsh7drftky4xhrixw6nizo7kyehnh2t3pb65.py
# Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_35 => convert_element_type_69, convolution_17, view_51
# Graph fragment:
#   %buf159 : Tensor  = PlaceHolder[target=buf159]
#   %view_51 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%empty_49, [32, 256, 14, 14]), kwargs = {})
#   %convert_element_type_69 : Tensor "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg86_1, torch.bfloat16), kwargs = {})
#   %convolution_17 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_51, %convert_element_type_69, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf162
triton_poi_fused__to_copy_convolution_view_31 = async_compile.triton('triton_poi_fused__to_copy_convolution_view_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_view_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 6422528, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_view_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 196*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 50176*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/3v/c3vsgksv2mwgcuzjwafdv5ovzngejhii4hfjzb6ovdn74ihcbjzu.py
# Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_35 => convert_element_type_69, convolution_17, view_51
# Graph fragment:
#   %buf162 : Tensor "bf16[32, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = PlaceHolder[target=buf162]
#   %convert_element_type_69 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=convert_element_type_69]
#   %view_51 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%empty_49, [32, 256, 14, 14]), kwargs = {})
#   %convert_element_type_69 : Tensor "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg86_1, torch.bfloat16), kwargs = {})
#   %convolution_17 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_51, %convert_element_type_69, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_17
triton_tem_fused__to_copy_convolution_view_32 = async_compile.triton('triton_tem_fused__to_copy_convolution_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_view_32', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_view_32(arg_X, arg_W, out_ptr0):
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
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 256
    IN_H = 14
    IN_W = 14
    OUT_C = 256
    OUT_H = 14
    OUT_W = 14

    # Strides:
    stride_xn = 50176
    stride_xc = 1
    stride_xh = 3584
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
    xindex = idx_w + 14*idx_h + 196*idx_c + 50176*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/fi/cfiwhkfntoqvlzid4nmip7o6dcdm77nbraj5ksb7zuqr35bhwaz4.py
# Topologically Sorted Source Nodes: [input_37], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_37 => convert_element_type_73
# Graph fragment:
#   %arg91_1 : Tensor "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=arg91_1]
#   %convert_element_type_73 : Tensor "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg91_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_73
triton_poi_fused__to_copy_33 = async_compile.triton('triton_poi_fused__to_copy_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/sd/csdrbh5vvbsbrqypg7zqemvztycttd3b476ksfxrtanncqawaold.py
# Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_37 => convert_element_type_73, convolution_18, view_54
# Graph fragment:
#   %buf171 : Tensor "bf16[32, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf171]
#   %convert_element_type_73 : Tensor "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=convert_element_type_73]
#   %view_54 : Tensor "bf16[32, 128, 28, 28][100352, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_38, [32, 128, 28, 28]), kwargs = {})
#   %convert_element_type_73 : Tensor "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg91_1, torch.bfloat16), kwargs = {})
#   %convolution_18 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_54, %convert_element_type_73, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_18
triton_tem_fused__to_copy_convolution_view_34 = async_compile.triton('triton_tem_fused__to_copy_convolution_view_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=4,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_view_34', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 1, 'UNROLL': True, 'ALLOW_TF32': False, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_view_34(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 128
    IN_H = 28
    IN_W = 28
    OUT_C = 256
    OUT_H = 14
    OUT_W = 14

    # Strides:
    stride_xn = 100352
    stride_xc = 1
    stride_xh = 3584
    stride_xw = 128
    stride_wc_out = 128
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

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




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

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
    xindex = idx_w + 14*idx_h + 196*idx_c + 50176*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/w5/cw5crnfdwioccdi66bx7dpby44ukc3vbbv74bq3qqvday3tk5lqd.py
# Topologically Sorted Source Nodes: [out_7, input_40], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_40 => convert_element_type_77, convolution_19, view_57
#   out_7 => add_45
# Graph fragment:
#   %buf168 : Tensor  = PlaceHolder[target=buf168]
#   %buf177 : Tensor  = PlaceHolder[target=buf177]
#   %add_45 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0" = PlaceHolder[target=add_45]
#   %add_45 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_52, %empty_55), kwargs = {})
#   %view_57 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_45, [32, 256, 14, 14]), kwargs = {})
#   %convert_element_type_77 : Tensor "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg96_1, torch.bfloat16), kwargs = {})
#   %convolution_19 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_57, %convert_element_type_77, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_45,%buf181
triton_poi_fused__to_copy_add_convolution_view_35 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 9633792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_35(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 196)
    x2 = ((xindex // 196) % 256)
    x3 = xindex // 50176
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x2 + 256*x1 + 50176*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/ne/cne6n3ewyqrqpv33srsouzw6d5rqi6mxnlpwvnwexm46wxqxzmdm.py
# Topologically Sorted Source Nodes: [out_8, input_44], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_44 => convert_element_type_85, convolution_21, view_63
#   out_8 => add_50
# Graph fragment:
#   %buf196 : Tensor  = PlaceHolder[target=buf196]
#   %add_45 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0" = PlaceHolder[target=add_45]
#   %add_50 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0" = PlaceHolder[target=add_50]
#   %add_50 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_61, %add_45), kwargs = {})
#   %view_63 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_50, [32, 256, 14, 14]), kwargs = {})
#   %convert_element_type_85 : Tensor "bf16[256, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg106_1, torch.bfloat16), kwargs = {})
#   %convolution_21 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_63, %convert_element_type_85, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_50,%buf200
triton_poi_fused__to_copy_add_convolution_view_36 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 12845056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_36(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 196)
    x2 = ((xindex // 196) % 256)
    x3 = xindex // 50176
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr0 + (x2 + 256*x1 + 50176*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/i3/ci3st72sjfkje4nqicnr7adojjken2n3zsnpblygsbghhgb52ptu.py
# Topologically Sorted Source Nodes: [out_12, input_60, input_64], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_60 => convert_element_type_117, convolution_29, view_87
#   input_64 => convert_element_type_125, convolution_31, view_93
#   out_12 => add_70
# Graph fragment:
#   %buf272 : Tensor  = PlaceHolder[target=buf272]
#   %add_65 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0" = PlaceHolder[target=add_65]
#   %add_70 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0" = PlaceHolder[target=add_70]
#   %add_70 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_85, %add_65), kwargs = {})
#   %view_87 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [32, 256, 14, 14]), kwargs = {})
#   %convert_element_type_117 : Tensor "bf16[512, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg146_1, torch.bfloat16), kwargs = {})
#   %convolution_29 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_87, %convert_element_type_117, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_93 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [32, 256, 14, 14]), kwargs = {})
#   %convert_element_type_125 : Tensor "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg156_1, torch.bfloat16), kwargs = {})
#   %convolution_31 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_93, %convert_element_type_125, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_70,%buf276,%buf294
triton_poi_fused__to_copy_add_convolution_view_37 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3211264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_37(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 196)
    x2 = ((xindex // 196) % 256)
    x3 = xindex // 50176
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 256*x1 + 50176*x3), tmp2, None)
    tl.store(out_ptr1 + (x2 + 256*x1 + 50176*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/ds/cdsqcybuwqu5vkvaombkk4s6wynhcgxjcqirvunod5hekd6zrzgs.py
# Topologically Sorted Source Nodes: [input_60], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_60 => convert_element_type_117
# Graph fragment:
#   %arg146_1 : Tensor "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg146_1]
#   %convert_element_type_117 : Tensor "bf16[512, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg146_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_117
triton_poi_fused__to_copy_38 = async_compile.triton('triton_poi_fused__to_copy_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4718592, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/kf/ckf5izr6pohn5wrhrmu3on6rt6xz7xyg32ysl64n64nwevv6wqvw.py
# Topologically Sorted Source Nodes: [out_12, input_60], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_60 => convert_element_type_117, convolution_29, view_87
#   out_12 => add_70
# Graph fragment:
#   %buf276 : Tensor "bf16[32, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = PlaceHolder[target=buf276]
#   %convert_element_type_117 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=convert_element_type_117]
#   %add_70 : Tensor "bf16[4, 8, 256, 14, 14][401408, 50176, 196, 14, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_85, %add_65), kwargs = {})
#   %view_87 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [32, 256, 14, 14]), kwargs = {})
#   %convert_element_type_117 : Tensor "bf16[512, 256, 3, 3][2304, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg146_1, torch.bfloat16), kwargs = {})
#   %convolution_29 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_87, %convert_element_type_117, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_29
triton_tem_fused__to_copy_add_convolution_view_39 = async_compile.triton('triton_tem_fused__to_copy_add_convolution_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=4,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_convolution_view_39', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_add_convolution_view_39(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 256
    IN_H = 14
    IN_W = 14
    OUT_C = 512
    OUT_H = 7
    OUT_W = 7

    # Strides:
    stride_xn = 50176
    stride_xc = 1
    stride_xh = 3584
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
    xindex = idx_w + 7*idx_h + 49*idx_c + 25088*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/vn/cvn4qvf6oqjrlmotyrpfc6l5mimrg6fia7qplfo6ayvzchlbvlhn.py
# Topologically Sorted Source Nodes: [input_61, view_29, full_like_29, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_6
#   full_like_29 => full_default_29
#   input_61 => add_71, add_72, convert_element_type_120, mul_87, mul_88, mul_89, reciprocal_29, sqrt_29, sub_29, unsqueeze_233, unsqueeze_234, unsqueeze_235, unsqueeze_236, unsqueeze_237, unsqueeze_238, unsqueeze_239, unsqueeze_240
#   view_29 => view_88
# Graph fragment:
#   %convolution_29 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0" = PlaceHolder[target=convolution_29]
#   %arg147_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg147_1]
#   %arg148_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg148_1]
#   %arg149_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %arg150_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg150_1]
#   %unsqueeze_233 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg147_1, -1), kwargs = {})
#   %unsqueeze_234 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_233, -1), kwargs = {})
#   %sub_29 : Tensor "f32[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_234), kwargs = {})
#   %add_71 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg148_1, 1e-05), kwargs = {})
#   %sqrt_29 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_71,), kwargs = {})
#   %reciprocal_29 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_29,), kwargs = {})
#   %mul_87 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_29, 1), kwargs = {})
#   %unsqueeze_235 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_87, -1), kwargs = {})
#   %unsqueeze_236 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_235, -1), kwargs = {})
#   %mul_88 : Tensor "f32[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_236), kwargs = {})
#   %unsqueeze_237 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg149_1, -1), kwargs = {})
#   %unsqueeze_238 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_237, -1), kwargs = {})
#   %mul_89 : Tensor "f32[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_238), kwargs = {})
#   %unsqueeze_239 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, -1), kwargs = {})
#   %unsqueeze_240 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_239, -1), kwargs = {})
#   %add_72 : Tensor "f32[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_240), kwargs = {})
#   %convert_element_type_120 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_72, torch.bfloat16), kwargs = {})
#   %view_88 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_120, [4, 8, 512, 7, 7]), kwargs = {})
#   %full_default_29 : Tensor "bf16[8, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 512, 7, 7], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_6 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 65, grid: [(1568, 1, 1), (784, 1, 1), (784, 1, 1), (392, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %view_88, v_init_ptr: %full_default_29, s_seq_ptr: %empty_88, h_seq_ptr: %empty_89, v_seq_ptr: %empty_89, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 200704, soft_reset: False, save_intermediates: False}})
#   return %buf280
triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4816896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 49) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 1e-05, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.sqrt_rn(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = tmp9 * tmp8
    tmp11 = tmp3 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/yz/cyzmgm5fgimaegvgynus2kbmdsehjc25w6px7xgkl7an5nleheuj.py
# Topologically Sorted Source Nodes: [input_61, view_29, full_like_29, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_6
#   full_like_29 => full_default_29
#   input_61 => add_71, add_72, convert_element_type_120, mul_87, mul_88, mul_89, reciprocal_29, sqrt_29, sub_29, unsqueeze_233, unsqueeze_234, unsqueeze_235, unsqueeze_236, unsqueeze_237, unsqueeze_238, unsqueeze_239, unsqueeze_240
#   view_29 => view_88
# Graph fragment:
#   %unsqueeze_233 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg147_1, -1), kwargs = {})
#   %unsqueeze_234 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_233, -1), kwargs = {})
#   %sub_29 : Tensor "f32[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_234), kwargs = {})
#   %add_71 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg148_1, 1e-05), kwargs = {})
#   %sqrt_29 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_71,), kwargs = {})
#   %reciprocal_29 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_29,), kwargs = {})
#   %mul_87 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_29, 1), kwargs = {})
#   %unsqueeze_235 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_87, -1), kwargs = {})
#   %unsqueeze_236 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_235, -1), kwargs = {})
#   %mul_88 : Tensor "f32[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_236), kwargs = {})
#   %unsqueeze_237 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg149_1, -1), kwargs = {})
#   %unsqueeze_238 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_237, -1), kwargs = {})
#   %mul_89 : Tensor "f32[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_238), kwargs = {})
#   %unsqueeze_239 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg150_1, -1), kwargs = {})
#   %unsqueeze_240 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_239, -1), kwargs = {})
#   %add_72 : Tensor "f32[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_240), kwargs = {})
#   %convert_element_type_120 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_72, torch.bfloat16), kwargs = {})
#   %view_88 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_120, [4, 8, 512, 7, 7]), kwargs = {})
#   %full_default_29 : Tensor "bf16[8, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 512, 7, 7], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %triton_kernel_wrapper_mutation_6 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 65, grid: [(1568, 1, 1), (784, 1, 1), (784, 1, 1), (392, 1, 1)], tma_descriptor_metadata: {}, kwargs: {x_seq_ptr: %view_88, v_init_ptr: %full_default_29, s_seq_ptr: %empty_88, h_seq_ptr: %empty_89, v_seq_ptr: %empty_89, v_threshold: 1.0, v_reset: 0.0, T: 4, NCL: 200704, soft_reset: False, save_intermediates: False}})
#   return %buf281
triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 802816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py:25
_multistep_if_forward_kernel_4 = async_compile.triton('_multistep_if_forward_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['1568', '1', '1'], 'cpp': ['1568L', '1L', '1L'], 'python_slow': ['1568', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['784', '1', '1'], 'cpp': ['784L', '1L', '1L'], 'python_slow': ['784', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['784', '1', '1'], 'cpp': ['784L', '1L', '1L'], 'python_slow': ['784', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['392', '1', '1'], 'cpp': ['392L', '1L', '1L'], 'python_slow': ['392', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_if_forward_kernel_4', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 4, 'NCL': 200704, 'dtype': triton.language.bfloat16, 'soft_reset': False, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _multistep_if_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    v_init_ptrs = tl.make_block_ptr(
        v_init_ptr,
        shape=(1, NCL),
        strides=(NCL, 1),
        offsets=(0, ncl_offset),
        block_shape=(1, BLOCK_NCL),
        order=(1, 0),
    )
    v = tl.load(v_init_ptrs, boundary_check=(1,), padding_option="zero")

    for t in tl.static_range(0, T, 1):
        x_ptrs = tl.make_block_ptr(
            x_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        x = tl.load(x_ptrs, boundary_check=(1,), padding_option="zero")

        h = v + x
        s = (h >= v_threshold).to(dtype)
        if soft_reset:
            v = h - s * v_threshold
        else:
            v = s * v_reset + (1.0 - s) * h

        s_ptrs = tl.make_block_ptr(
            s_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(s_ptrs, s, boundary_check=(1,))
        v_ptrs = tl.make_block_ptr(
            v_seq_ptr,
            shape=(T, NCL),
            strides=(NCL, 1),
            offsets=(t, ncl_offset),
            block_shape=(1, BLOCK_NCL),
            order=(1, 0),
        )
        convert_and_store(v_ptrs, v, boundary_check=(1,))
        if save_intermediates:
            h_ptrs = tl.make_block_ptr(
                h_seq_ptr,
                shape=(T, NCL),
                strides=(NCL, 1),
                offsets=(t, ncl_offset),
                block_shape=(1, BLOCK_NCL),
                order=(1, 0),
            )
            convert_and_store(h_ptrs, h, boundary_check=(1,))

@triton.jit
def convert_and_store(pointer, value, boundary_check):
    # For block pointers created by tl.make_block_pointer(),
    # implicit type casting is not supported when calling tl.store().
    # This function manually converts dtype and then stores the data.
    value = value.to(pointer.dtype.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/5z/c5z4gu4ne4fnl2dbz3pl5rl7je55omw5fziqp3tyx73guepultxl.py
# Topologically Sorted Source Nodes: [input_62], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_62 => convert_element_type_121
# Graph fragment:
#   %arg151_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg151_1]
#   %convert_element_type_121 : Tensor "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg151_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_121
triton_poi_fused__to_copy_42 = async_compile.triton('triton_poi_fused__to_copy_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 9437184, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_42(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/x6/cx6vrx46wzm2mj2stl2nwijyqiiqgveudpx5trxqgalrzxhzwyk5.py
# Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_62 => convert_element_type_121, convolution_30, view_90
# Graph fragment:
#   %buf282 : Tensor  = PlaceHolder[target=buf282]
#   %view_90 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%empty_88, [32, 512, 7, 7]), kwargs = {})
#   %convert_element_type_121 : Tensor "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg151_1, torch.bfloat16), kwargs = {})
#   %convolution_30 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_90, %convert_element_type_121, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf285
triton_poi_fused__to_copy_convolution_view_43 = async_compile.triton('triton_poi_fused__to_copy_convolution_view_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_view_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 3211264, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_view_43(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 25088*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/k5/ck5h72v3un6k6e3jnxzknwpdq3oowen2uh45ewy2jutkfmf4gnsh.py
# Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_62 => convert_element_type_121, convolution_30, view_90
# Graph fragment:
#   %buf285 : Tensor "bf16[32, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = PlaceHolder[target=buf285]
#   %convert_element_type_121 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = PlaceHolder[target=convert_element_type_121]
#   %view_90 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%empty_88, [32, 512, 7, 7]), kwargs = {})
#   %convert_element_type_121 : Tensor "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg151_1, torch.bfloat16), kwargs = {})
#   %convolution_30 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_90, %convert_element_type_121, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_30
triton_tem_fused__to_copy_convolution_view_44 = async_compile.triton('triton_tem_fused__to_copy_convolution_view_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=4,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_view_44', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': False, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_view_44(arg_X, arg_W, out_ptr0):
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
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 512
    IN_H = 7
    IN_W = 7
    OUT_C = 512
    OUT_H = 7
    OUT_W = 7

    # Strides:
    stride_xn = 25088
    stride_xc = 1
    stride_xh = 3584
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
    xindex = idx_w + 7*idx_h + 49*idx_c + 25088*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/kw/ckwhc5k4kyu5rno47h6wbv5lw2ue7mrtsdzz6x3m6ypzmuzwkncy.py
# Topologically Sorted Source Nodes: [input_64], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_64 => convert_element_type_125
# Graph fragment:
#   %arg156_1 : Tensor "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=arg156_1]
#   %convert_element_type_125 : Tensor "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg156_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_125
triton_poi_fused__to_copy_45 = async_compile.triton('triton_poi_fused__to_copy_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/od/codibg3qlyy3b5pypdyo5oltotvajfndwo57ak3zgvmser4i7laf.py
# Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_64 => convert_element_type_125, convolution_31, view_93
# Graph fragment:
#   %buf294 : Tensor "bf16[32, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = PlaceHolder[target=buf294]
#   %convert_element_type_125 : Tensor "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=convert_element_type_125]
#   %view_93 : Tensor "bf16[32, 256, 14, 14][50176, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_70, [32, 256, 14, 14]), kwargs = {})
#   %convert_element_type_125 : Tensor "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg156_1, torch.bfloat16), kwargs = {})
#   %convolution_31 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_93, %convert_element_type_125, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_31
triton_tem_fused__to_copy_convolution_view_46 = async_compile.triton('triton_tem_fused__to_copy_convolution_view_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=4,
triton_meta={'signature': {'arg_X': '*bf16', 'arg_W': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_convolution_view_46', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 1, 'KERNEL_W': 1, 'STRIDE_H': 2, 'STRIDE_W': 2, 'PADDING_H': 0, 'PADDING_W': 0, 'GROUPS': 1, 'UNROLL': True, 'ALLOW_TF32': False, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}},

)
@triton.jit
def triton_tem_fused__to_copy_convolution_view_46(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 1
    KERNEL_W : tl.constexpr = 1
    STRIDE_H : tl.constexpr = 2
    STRIDE_W : tl.constexpr = 2
    PADDING_H : tl.constexpr = 0
    PADDING_W : tl.constexpr = 0
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 32
    IN_C = 256
    IN_H = 14
    IN_W = 14
    OUT_C = 512
    OUT_H = 7
    OUT_W = 7

    # Strides:
    stride_xn = 50176
    stride_xc = 1
    stride_xh = 3584
    stride_xw = 256
    stride_wc_out = 256
    stride_wc_in = 1
    stride_wh = 1
    stride_ww = 1

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




    i = 0
    j = 0
    for k in range(0, GROUP_IN_C, BLOCK_K):

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
    xindex = idx_w + 7*idx_h + 49*idx_c + 25088*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/7b/c7bxztow367wwrfktnrfphljyunud25qwsvoarvtownorvrmu3yx.py
# Topologically Sorted Source Nodes: [out_13, input_67], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_67 => convert_element_type_129, convolution_32, view_96
#   out_13 => add_77
# Graph fragment:
#   %buf291 : Tensor  = PlaceHolder[target=buf291]
#   %buf300 : Tensor  = PlaceHolder[target=buf300]
#   %add_77 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0" = PlaceHolder[target=add_77]
#   %add_77 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_91, %empty_94), kwargs = {})
#   %view_96 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_77, [32, 512, 7, 7]), kwargs = {})
#   %convert_element_type_129 : Tensor "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg161_1, torch.bfloat16), kwargs = {})
#   %convolution_32 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_96, %convert_element_type_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_77,%buf304
triton_poi_fused__to_copy_add_convolution_view_47 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4816896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_47(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 49)
    x2 = ((xindex // 49) % 512)
    x3 = xindex // 25088
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x2 + 512*x1 + 25088*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/we/cweryxlwmf5bcbxbfmgx2uqodqy4qw3qld3nrbc6kfvqrdvdlvoj.py
# Topologically Sorted Source Nodes: [out_14, input_71], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   input_71 => convert_element_type_137, convolution_34, view_102
#   out_14 => add_82
# Graph fragment:
#   %buf319 : Tensor  = PlaceHolder[target=buf319]
#   %add_77 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0" = PlaceHolder[target=add_77]
#   %add_82 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0" = PlaceHolder[target=add_82]
#   %add_82 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_100, %add_77), kwargs = {})
#   %view_102 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_82, [32, 512, 7, 7]), kwargs = {})
#   %convert_element_type_137 : Tensor "bf16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg171_1, torch.bfloat16), kwargs = {})
#   %convolution_34 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_102, %convert_element_type_137, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_82,%buf323
triton_poi_fused__to_copy_add_convolution_view_48 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_view_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_view_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6422528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_view_48(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 49)
    x2 = ((xindex // 49) % 512)
    x3 = xindex // 25088
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr0 + (x2 + 512*x1 + 25088*x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/db/cdb6mjkzs6ikp6mhqtklqfvynvch4xjscs63f5x3xqpvy7unqt53.py
# Topologically Sorted Source Nodes: [out_15, input_75], Original ATen: [aten.add, aten.view, aten.mean]
# Source node to ATen node mapping:
#   input_75 => mean, view_108
#   out_15 => add_87
# Graph fragment:
#   %buf338 : Tensor  = PlaceHolder[target=buf338]
#   %add_82 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0" = PlaceHolder[target=add_82]
#   %add_87 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_106, %add_82), kwargs = {})
#   %view_108 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_87, [32, 512, 7, 7]), kwargs = {})
#   %mean : Tensor "bf16[32, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_108, [-1, -2], True), kwargs = {})
#   return %buf340
triton_per_fused_add_mean_view_49 = async_compile.triton('triton_per_fused_add_mean_view_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_view_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 131072, 'r0_': 1605632}}
)
@triton.jit
def triton_per_fused_add_mean_view_49(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 49*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 49*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp7, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/ou/couyyylraqpfyzdbaskrv564k2vsnjikwkqsdd2gjlfefwrbs6ln.py
# Topologically Sorted Source Nodes: [out_15, input_75, x_4, x_5, mean], Original ATen: [aten.add, aten.view, aten.mean]
# Source node to ATen node mapping:
#   input_75 => mean, view_108
#   mean => mean_1
#   out_15 => add_87
#   x_4 => view_109
#   x_5 => view_110
# Graph fragment:
#   %buf340 : Tensor "f32[32, 512, 1, 1][512, 1, 16384, 16384]cuda:0" = PlaceHolder[target=buf340]
#   %add_87 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_106, %add_82), kwargs = {})
#   %view_108 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_87, [32, 512, 7, 7]), kwargs = {})
#   %mean : Tensor "bf16[32, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_108, [-1, -2], True), kwargs = {})
#   %view_109 : Tensor "bf16[4, 8, 512, 1, 1][4096, 512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mean, [4, 8, 512, 1, 1]), kwargs = {})
#   %view_110 : Tensor "bf16[4, 8, 512][4096, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_109, [4, 8, 512]), kwargs = {})
#   %mean_1 : Tensor "bf16[8, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_110, [0]), kwargs = {})
#   return %mean_1
triton_poi_fused_add_mean_view_50 = async_compile.triton('triton_poi_fused_add_mean_view_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_view_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 81920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_view_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr0 + (4096 + x0), None)
    tmp10 = tl.load(in_ptr0 + (8192 + x0), None)
    tmp15 = tl.load(in_ptr0 + (12288 + x0), None)
    tmp1 = tl.full([1], 49.0, tl.float32)
    tmp2 = (tmp0 / tmp1)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = (tmp5 / tmp1)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp4 + tmp8
    tmp11 = (tmp10 / tmp1)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp9 + tmp13
    tmp16 = (tmp15 / tmp1)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp14 + tmp18
    tmp20 = tl.full([1], 4.0, tl.float32)
    tmp21 = (tmp19 / tmp20)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp22, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/rx/crx4obwurdbw6i5nbzepi4jo3gfarucgefrqlephznx4vwarkp4i.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_146
# Graph fragment:
#   %arg181_1 : Tensor "f32[1000, 512][512, 1]cuda:0" = PlaceHolder[target=arg181_1]
#   %convert_element_type_146 : Tensor "bf16[1000, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg181_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_146
triton_poi_fused__to_copy_51 = async_compile.triton('triton_poi_fused__to_copy_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4096000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/jq/cjqoq2nppdwygydzgesyk3wt4gdcw36bmdewdxu4xpcoz2dtt63s.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_145
# Graph fragment:
#   %arg182_1 : Tensor "f32[1000][1]cuda:0" = PlaceHolder[target=arg182_1]
#   %convert_element_type_145 : Tensor "bf16[1000][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg182_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_145
triton_poi_fused__to_copy_52 = async_compile.triton('triton_poi_fused__to_copy_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/liushifeng/charlley/snn_infer_triton/capture/sew/inductor_cache/gd/cgdkr6dzh6wykilypdb5tvmcrpoaawwtxvfdm775tkchnvckgemo.py
# Topologically Sorted Source Nodes: [linear, out_15, input_75, x_4, x_5, mean], Original ATen: [aten._to_copy, aten.add, aten.view, aten.mean, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   input_75 => mean, view_108
#   linear => addmm, convert_element_type_145, convert_element_type_146, permute
#   mean => mean_1
#   out_15 => add_87
#   x_4 => view_109
#   x_5 => view_110
# Graph fragment:
#   %convert_element_type_145 : Tensor "bf16[1000][1]cuda:0" = PlaceHolder[target=convert_element_type_145]
#   %mean_1 : Tensor "bf16[8, 512][512, 1]cuda:0" = PlaceHolder[target=mean_1]
#   %convert_element_type_146 : Tensor "bf16[1000, 512][512, 1]cuda:0" = PlaceHolder[target=convert_element_type_146]
#   %convert_element_type_145 : Tensor "bf16[1000][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg182_1, torch.bfloat16), kwargs = {})
#   %add_87 : Tensor "bf16[4, 8, 512, 7, 7][200704, 25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%empty_106, %add_82), kwargs = {})
#   %view_108 : Tensor "bf16[32, 512, 7, 7][25088, 49, 7, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_87, [32, 512, 7, 7]), kwargs = {})
#   %mean : Tensor "bf16[32, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_108, [-1, -2], True), kwargs = {})
#   %view_109 : Tensor "bf16[4, 8, 512, 1, 1][4096, 512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mean, [4, 8, 512, 1, 1]), kwargs = {})
#   %view_110 : Tensor "bf16[4, 8, 512][4096, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_109, [4, 8, 512]), kwargs = {})
#   %mean_1 : Tensor "bf16[8, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_110, [0]), kwargs = {})
#   %convert_element_type_146 : Tensor "bf16[1000, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg181_1, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[512, 1000][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_146, [1, 0]), kwargs = {})
#   %addmm : Tensor "bf16[8, 1000][1000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%convert_element_type_145, %mean_1, %permute), kwargs = {})
#   return %addmm
triton_tem_fused__to_copy_add_addmm_mean_t_view_53 = async_compile.triton('triton_tem_fused__to_copy_add_addmm_mean_t_view_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=2,
triton_meta={'signature': {'in_ptr0': '*bf16', 'arg_A': '*bf16', 'arg_B': '*bf16', 'out_ptr0': '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_addmm_mean_t_view_53', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'OUT_DTYPE': 'tl.bfloat16', 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'GROUP_M': 8, 'ALLOW_TF32': False}},

)
@triton.jit
def triton_tem_fused__to_copy_add_addmm_mean_t_view_53(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    OUT_DTYPE : tl.constexpr = tl.bfloat16
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 8
    N = 1000
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

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

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K).to(INDEX_DTYPE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1000*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(INDEX_DTYPE)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(INDEX_DTYPE)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1000*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1 = args
        args.clear()
        assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_misaligned(arg1_1)
            buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg1_1, buf0, 24, 50176, stream=raw_stream0)
            del arg1_1
            assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
            buf1 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_1:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg0_1, buf1, 192, 49, stream=raw_stream0)
            del arg0_1
            buf2 = empty_strided_cuda((8, 64, 112, 112), (802816, 12544, 112, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_2:3
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_2.run(buf0, buf1, buf2, 1568, 1, 1, stream=raw_stream0)
            del buf0
            del buf1
            assert_size_stride(arg2_1, (64, ), (1, ))
            assert_size_stride(arg3_1, (64, ), (1, ))
            assert_size_stride(arg4_1, (64, ), (1, ))
            assert_size_stride(arg5_1, (64, ), (1, ))
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_no_training]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_3:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 6422528, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf4 = empty_strided_cuda((4, 8, 64, 112, 112), (6422528, 802816, 12544, 112, 1), torch.bfloat16)
            buf5 = empty_strided_cuda((4, 8, 64, 112, 112), (6422528, 802816, 12544, 112, 1), torch.bfloat16)
            buf6 = empty_strided_cuda((4, 8, 64, 112, 112), (6422528, 802816, 12544, 112, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, unsqueeze_, x_2, full_like, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.unsqueeze, aten.repeat, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_4:5
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_4.run(buf3, buf6, 25690112, stream=raw_stream0)
            buf7 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [x_1, unsqueeze_, x_2, full_like, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.unsqueeze, aten.repeat, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_5:6
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_repeat_unsqueeze_5.run(buf7, 6422528, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_1, unsqueeze_, x_2, full_like, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.unsqueeze, aten.repeat, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_0:7
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_0.run(buf6, buf7, buf4, buf5, buf5, 1.0, 0.0, stream=raw_stream0)
            del buf6
            buf10 = reinterpret_tensor(buf7, (32, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.view, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_max_pool2d_with_indices_view_6:8
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_view_6.run(buf4, buf10, 2048, 3136, stream=raw_stream0)
            del buf4
            assert_size_stride(arg6_1, (64, 64, 3, 3), (576, 9, 3, 1))
            buf11 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:9
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg6_1, buf11, 4096, 9, stream=raw_stream0)
            del arg6_1
            assert_size_stride(arg7_1, (64, ), (1, ))
            assert_size_stride(arg8_1, (64, ), (1, ))
            assert_size_stride(arg9_1, (64, ), (1, ))
            assert_size_stride(arg10_1, (64, ), (1, ))
            buf15 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_3, y_1, input_2, input_3, view_1, full_like_1, ], Original ATen: [aten.view, aten._to_copy, aten.convolution, aten._native_batch_norm_legit_no_training, aten.full_like]
            # [Provenance debug handles] triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8:10
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8.run(buf10, buf11, arg7_1, arg8_1, arg9_1, arg10_1, buf15, 784, 1, 1, stream=raw_stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            buf13 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            buf14 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            buf16 = empty_strided_cuda((8, 64, 56, 56), (200704, 3136, 56, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_3, view_1, full_like_1, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9:11
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9.run(buf16, 1605632, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_3, view_1, full_like_1, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_1:12
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_1.run(buf15, buf16, buf13, buf14, buf14, 1.0, 0.0, stream=raw_stream0)
            del buf15
            assert_size_stride(arg11_1, (64, 64, 3, 3), (576, 9, 3, 1))
            buf19 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:13
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg11_1, buf19, 4096, 9, stream=raw_stream0)
            del arg11_1
            buf20 = empty_strided_cuda((32, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_10:14
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_10.run(buf13, buf20, 2048, 3136, stream=raw_stream0)
            assert_size_stride(arg12_1, (64, ), (1, ))
            assert_size_stride(arg13_1, (64, ), (1, ))
            assert_size_stride(arg14_1, (64, ), (1, ))
            assert_size_stride(arg15_1, (64, ), (1, ))
            buf24 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [input_4, input_5, view_2, full_like_2, ], Original ATen: [aten.view, aten._to_copy, aten.convolution, aten._native_batch_norm_legit_no_training, aten.full_like]
            # [Provenance debug handles] triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8:15
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8.run(buf20, buf19, arg12_1, arg13_1, arg14_1, arg15_1, buf24, 784, 1, 1, stream=raw_stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            buf22 = reinterpret_tensor(buf20, (4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), 0); del buf20  # reuse
            buf23 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            buf25 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [input_5, view_2, full_like_2, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9:16
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9.run(buf25, 1605632, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_5, view_2, full_like_2, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_1:17
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_1.run(buf24, buf25, buf22, buf23, buf23, 1.0, 0.0, stream=raw_stream0)
            buf28 = buf24; del buf24  # reuse
            buf30 = empty_strided_cuda((32, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_3, out, input_6], Original ATen: [aten.view, aten.add, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_11:18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_11.run(buf22, buf10, buf28, buf30, 2048, 3136, stream=raw_stream0)
            del buf10
            del buf22
            assert_size_stride(arg16_1, (64, 64, 3, 3), (576, 9, 3, 1))
            buf29 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:19
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg16_1, buf29, 4096, 9, stream=raw_stream0)
            del arg16_1
            assert_size_stride(arg17_1, (64, ), (1, ))
            assert_size_stride(arg18_1, (64, ), (1, ))
            assert_size_stride(arg19_1, (64, ), (1, ))
            assert_size_stride(arg20_1, (64, ), (1, ))
            buf34 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_3, out, input_6, input_7, view_3, full_like_3, ], Original ATen: [aten.view, aten.add, aten._to_copy, aten.convolution, aten._native_batch_norm_legit_no_training, aten.full_like]
            # [Provenance debug handles] triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8:20
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8.run(buf30, buf29, arg17_1, arg18_1, arg19_1, arg20_1, buf34, 784, 1, 1, stream=raw_stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            buf32 = reinterpret_tensor(buf30, (4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), 0); del buf30  # reuse
            buf33 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            buf35 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [input_7, view_3, full_like_3, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9:21
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9.run(buf35, 1605632, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_7, view_3, full_like_3, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_1:22
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_1.run(buf34, buf35, buf32, buf33, buf33, 1.0, 0.0, stream=raw_stream0)
            del buf34
            assert_size_stride(arg21_1, (64, 64, 3, 3), (576, 9, 3, 1))
            buf38 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:23
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg21_1, buf38, 4096, 9, stream=raw_stream0)
            del arg21_1
            buf39 = empty_strided_cuda((32, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_10:24
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_10.run(buf32, buf39, 2048, 3136, stream=raw_stream0)
            assert_size_stride(arg22_1, (64, ), (1, ))
            assert_size_stride(arg23_1, (64, ), (1, ))
            assert_size_stride(arg24_1, (64, ), (1, ))
            assert_size_stride(arg25_1, (64, ), (1, ))
            buf43 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [input_8, input_9, view_4, full_like_4, ], Original ATen: [aten.view, aten._to_copy, aten.convolution, aten._native_batch_norm_legit_no_training, aten.full_like]
            # [Provenance debug handles] triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8:25
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8.run(buf39, buf38, arg22_1, arg23_1, arg24_1, arg25_1, buf43, 784, 1, 1, stream=raw_stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            buf41 = reinterpret_tensor(buf39, (4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), 0); del buf39  # reuse
            buf42 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            buf44 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [input_9, view_4, full_like_4, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9:26
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9.run(buf44, 1605632, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_9, view_4, full_like_4, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_1:27
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_1.run(buf43, buf44, buf41, buf42, buf42, 1.0, 0.0, stream=raw_stream0)
            buf47 = buf28; del buf28  # reuse
            buf49 = reinterpret_tensor(buf43, (32, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [out_1, input_10], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_12:28
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_12.run(buf47, buf41, buf49, 6422528, stream=raw_stream0)
            del buf41
            assert_size_stride(arg26_1, (64, 64, 3, 3), (576, 9, 3, 1))
            buf48 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:29
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg26_1, buf48, 4096, 9, stream=raw_stream0)
            del arg26_1
            assert_size_stride(arg27_1, (64, ), (1, ))
            assert_size_stride(arg28_1, (64, ), (1, ))
            assert_size_stride(arg29_1, (64, ), (1, ))
            assert_size_stride(arg30_1, (64, ), (1, ))
            buf53 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_1, input_10, input_11, view_5, full_like_5, ], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution, aten._native_batch_norm_legit_no_training, aten.full_like]
            # [Provenance debug handles] triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8:30
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8.run(buf49, buf48, arg27_1, arg28_1, arg29_1, arg30_1, buf53, 784, 1, 1, stream=raw_stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            buf51 = reinterpret_tensor(buf49, (4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), 0); del buf49  # reuse
            buf52 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            buf54 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [input_11, view_5, full_like_5, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9:31
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9.run(buf54, 1605632, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_11, view_5, full_like_5, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_1:32
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_1.run(buf53, buf54, buf51, buf52, buf52, 1.0, 0.0, stream=raw_stream0)
            del buf53
            assert_size_stride(arg31_1, (64, 64, 3, 3), (576, 9, 3, 1))
            buf57 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_7:33
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg31_1, buf57, 4096, 9, stream=raw_stream0)
            del arg31_1
            buf58 = empty_strided_cuda((32, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_10:34
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_10.run(buf51, buf58, 2048, 3136, stream=raw_stream0)
            assert_size_stride(arg32_1, (64, ), (1, ))
            assert_size_stride(arg33_1, (64, ), (1, ))
            assert_size_stride(arg34_1, (64, ), (1, ))
            assert_size_stride(arg35_1, (64, ), (1, ))
            buf62 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [input_12, input_13, view_6, full_like_6, ], Original ATen: [aten.view, aten._to_copy, aten.convolution, aten._native_batch_norm_legit_no_training, aten.full_like]
            # [Provenance debug handles] triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8:35
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__native_batch_norm_legit_no_training__to_copy_convolution_full_like_view_8.run(buf58, buf57, arg32_1, arg33_1, arg34_1, arg35_1, buf62, 784, 1, 1, stream=raw_stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            del buf57
            buf60 = reinterpret_tensor(buf58, (4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), 0); del buf58  # reuse
            buf61 = empty_strided_cuda((4, 8, 64, 56, 56), (1605632, 200704, 3136, 56, 1), torch.bfloat16)
            buf63 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [input_13, view_6, full_like_6, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9:36
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_9.run(buf63, 1605632, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_13, view_6, full_like_6, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_1:37
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_1.run(buf62, buf63, buf60, buf61, buf61, 1.0, 0.0, stream=raw_stream0)
            buf66 = buf47; del buf47  # reuse
            buf68 = reinterpret_tensor(buf62, (32, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf62  # reuse
            buf86 = empty_strided_cuda((32, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_2, input_14, input_18], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_13:38
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_13.run(buf66, buf60, buf68, buf86, 6422528, stream=raw_stream0)
            del buf60
            del buf66
            assert_size_stride(arg36_1, (128, 64, 3, 3), (576, 9, 3, 1))
            buf67 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_14:39
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_14.run(arg36_1, buf67, 8192, 9, stream=raw_stream0)
            del arg36_1
            buf69 = empty_strided_cuda((32, 128, 28, 28), (100352, 784, 28, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_2, input_14], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_add_convolution_view_15:40
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_convolution_view_15.run(buf68, buf67, buf69, 196, 1, 1, stream=raw_stream0)
            del buf67
            del buf68
            buf70 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            buf71 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            assert_size_stride(arg37_1, (128, ), (1, ))
            assert_size_stride(arg38_1, (128, ), (1, ))
            assert_size_stride(arg39_1, (128, ), (1, ))
            assert_size_stride(arg40_1, (128, ), (1, ))
            buf72 = reinterpret_tensor(buf69, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf69  # reuse
            # Topologically Sorted Source Nodes: [input_15, view_7, full_like_7, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:41
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf72, arg37_1, arg38_1, arg39_1, arg40_1, 3211264, stream=raw_stream0)
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            buf73 = empty_strided_cuda((8, 128, 28, 28), (100352, 784, 28, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15, view_7, full_like_7, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:42
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf73, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_15, view_7, full_like_7, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:43
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf72, buf73, buf70, buf71, buf71, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg41_1, (128, 128, 3, 3), (1152, 9, 3, 1))
            buf76 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_18:44
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg41_1, buf76, 16384, 9, stream=raw_stream0)
            del arg41_1
            buf77 = reinterpret_tensor(buf72, (32, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_19:45
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_19.run(buf70, buf77, 4096, 784, stream=raw_stream0)
            buf78 = reinterpret_tensor(buf70, (32, 128, 28, 28), (100352, 784, 28, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_20:46
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_20.run(buf77, buf76, buf78, 196, 1, 1, stream=raw_stream0)
            buf79 = reinterpret_tensor(buf77, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf77  # reuse
            buf80 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            assert_size_stride(arg42_1, (128, ), (1, ))
            assert_size_stride(arg43_1, (128, ), (1, ))
            assert_size_stride(arg44_1, (128, ), (1, ))
            assert_size_stride(arg45_1, (128, ), (1, ))
            buf81 = reinterpret_tensor(buf78, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [input_17, view_8, full_like_8, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:47
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf81, arg42_1, arg43_1, arg44_1, arg45_1, 3211264, stream=raw_stream0)
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            buf82 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [input_17, view_8, full_like_8, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:48
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf82, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_17, view_8, full_like_8, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:49
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf81, buf82, buf79, buf80, buf80, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg46_1, (128, 64, 1, 1), (64, 1, 1, 1))
            buf85 = empty_strided_cuda((128, 64, 1, 1), (64, 1, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_21:50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_21.run(arg46_1, buf85, 8192, stream=raw_stream0)
            del arg46_1
            buf87 = reinterpret_tensor(buf81, (32, 128, 28, 28), (100352, 784, 28, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_22:51
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_22.run(buf86, buf85, buf87, 196, 1, 1, stream=raw_stream0)
            del buf85
            del buf86
            buf88 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            buf89 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            assert_size_stride(arg47_1, (128, ), (1, ))
            assert_size_stride(arg48_1, (128, ), (1, ))
            assert_size_stride(arg49_1, (128, ), (1, ))
            assert_size_stride(arg50_1, (128, ), (1, ))
            buf90 = reinterpret_tensor(buf87, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [input_19, input_20, full_like_9, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:52
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf90, arg47_1, arg48_1, arg49_1, arg50_1, 3211264, stream=raw_stream0)
            del arg47_1
            del arg48_1
            del arg49_1
            del arg50_1
            buf91 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [input_19, input_20, full_like_9, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:53
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf91, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_19, input_20, full_like_9, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:54
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf90, buf91, buf88, buf89, buf89, 1.0, 0.0, stream=raw_stream0)
            buf94 = buf90; del buf90  # reuse
            buf96 = empty_strided_cuda((32, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_3, input_21], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_23:55
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_23.run(buf79, buf88, buf94, buf96, 3211264, stream=raw_stream0)
            assert_size_stride(arg51_1, (128, 128, 3, 3), (1152, 9, 3, 1))
            buf95 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_18:56
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg51_1, buf95, 16384, 9, stream=raw_stream0)
            del arg51_1
            buf97 = reinterpret_tensor(buf88, (32, 128, 28, 28), (100352, 784, 28, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [out_3, input_21], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_20:57
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_20.run(buf96, buf95, buf97, 196, 1, 1, stream=raw_stream0)
            buf98 = reinterpret_tensor(buf96, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf96  # reuse
            buf99 = buf79; del buf79  # reuse
            assert_size_stride(arg52_1, (128, ), (1, ))
            assert_size_stride(arg53_1, (128, ), (1, ))
            assert_size_stride(arg54_1, (128, ), (1, ))
            assert_size_stride(arg55_1, (128, ), (1, ))
            buf100 = reinterpret_tensor(buf97, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [input_22, view_10, full_like_10, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:58
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf100, arg52_1, arg53_1, arg54_1, arg55_1, 3211264, stream=raw_stream0)
            del arg52_1
            del arg53_1
            del arg54_1
            del arg55_1
            buf101 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [input_22, view_10, full_like_10, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:59
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf101, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_22, view_10, full_like_10, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:60
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf100, buf101, buf98, buf99, buf99, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg56_1, (128, 128, 3, 3), (1152, 9, 3, 1))
            buf104 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_18:61
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg56_1, buf104, 16384, 9, stream=raw_stream0)
            del arg56_1
            buf105 = reinterpret_tensor(buf100, (32, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf100  # reuse
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_19:62
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_19.run(buf98, buf105, 4096, 784, stream=raw_stream0)
            buf106 = reinterpret_tensor(buf98, (32, 128, 28, 28), (100352, 784, 28, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_20:63
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_20.run(buf105, buf104, buf106, 196, 1, 1, stream=raw_stream0)
            buf107 = reinterpret_tensor(buf105, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf105  # reuse
            buf108 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            assert_size_stride(arg57_1, (128, ), (1, ))
            assert_size_stride(arg58_1, (128, ), (1, ))
            assert_size_stride(arg59_1, (128, ), (1, ))
            assert_size_stride(arg60_1, (128, ), (1, ))
            buf109 = reinterpret_tensor(buf106, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [input_24, view_11, full_like_11, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:64
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf109, arg57_1, arg58_1, arg59_1, arg60_1, 3211264, stream=raw_stream0)
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            buf110 = buf101; del buf101  # reuse
            # Topologically Sorted Source Nodes: [input_24, view_11, full_like_11, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:65
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf110, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_24, view_11, full_like_11, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:66
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf109, buf110, buf107, buf108, buf108, 1.0, 0.0, stream=raw_stream0)
            buf113 = buf94; del buf94  # reuse
            buf115 = reinterpret_tensor(buf109, (32, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [out_4, input_25], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_24:67
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_24.run(buf113, buf107, buf115, 3211264, stream=raw_stream0)
            assert_size_stride(arg61_1, (128, 128, 3, 3), (1152, 9, 3, 1))
            buf114 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_18:68
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg61_1, buf114, 16384, 9, stream=raw_stream0)
            del arg61_1
            buf116 = reinterpret_tensor(buf107, (32, 128, 28, 28), (100352, 784, 28, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [out_4, input_25], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_20:69
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_20.run(buf115, buf114, buf116, 196, 1, 1, stream=raw_stream0)
            buf117 = reinterpret_tensor(buf115, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf115  # reuse
            buf118 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            assert_size_stride(arg62_1, (128, ), (1, ))
            assert_size_stride(arg63_1, (128, ), (1, ))
            assert_size_stride(arg64_1, (128, ), (1, ))
            assert_size_stride(arg65_1, (128, ), (1, ))
            buf119 = reinterpret_tensor(buf116, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [input_26, view_12, full_like_12, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:70
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf119, arg62_1, arg63_1, arg64_1, arg65_1, 3211264, stream=raw_stream0)
            del arg62_1
            del arg63_1
            del arg64_1
            del arg65_1
            buf120 = buf110; del buf110  # reuse
            # Topologically Sorted Source Nodes: [input_26, view_12, full_like_12, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:71
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf120, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_26, view_12, full_like_12, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:72
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf119, buf120, buf117, buf118, buf118, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg66_1, (128, 128, 3, 3), (1152, 9, 3, 1))
            buf123 = buf114; del buf114  # reuse
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_18:73
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg66_1, buf123, 16384, 9, stream=raw_stream0)
            del arg66_1
            buf124 = reinterpret_tensor(buf119, (32, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_19:74
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_19.run(buf117, buf124, 4096, 784, stream=raw_stream0)
            buf125 = reinterpret_tensor(buf117, (32, 128, 28, 28), (100352, 784, 28, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_20:75
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_20.run(buf124, buf123, buf125, 196, 1, 1, stream=raw_stream0)
            buf126 = reinterpret_tensor(buf124, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf124  # reuse
            buf127 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            assert_size_stride(arg67_1, (128, ), (1, ))
            assert_size_stride(arg68_1, (128, ), (1, ))
            assert_size_stride(arg69_1, (128, ), (1, ))
            assert_size_stride(arg70_1, (128, ), (1, ))
            buf128 = reinterpret_tensor(buf125, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [input_28, view_13, full_like_13, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:76
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf128, arg67_1, arg68_1, arg69_1, arg70_1, 3211264, stream=raw_stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            del arg70_1
            buf129 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [input_28, view_13, full_like_13, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:77
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf129, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_28, view_13, full_like_13, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:78
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf128, buf129, buf126, buf127, buf127, 1.0, 0.0, stream=raw_stream0)
            buf132 = buf113; del buf113  # reuse
            buf134 = reinterpret_tensor(buf128, (32, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf128  # reuse
            # Topologically Sorted Source Nodes: [out_5, input_29], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_24:79
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_24.run(buf132, buf126, buf134, 3211264, stream=raw_stream0)
            assert_size_stride(arg71_1, (128, 128, 3, 3), (1152, 9, 3, 1))
            buf133 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_18:80
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg71_1, buf133, 16384, 9, stream=raw_stream0)
            del arg71_1
            buf135 = reinterpret_tensor(buf126, (32, 128, 28, 28), (100352, 784, 28, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [out_5, input_29], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_20:81
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_20.run(buf134, buf133, buf135, 196, 1, 1, stream=raw_stream0)
            buf136 = reinterpret_tensor(buf134, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf134  # reuse
            buf137 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            assert_size_stride(arg72_1, (128, ), (1, ))
            assert_size_stride(arg73_1, (128, ), (1, ))
            assert_size_stride(arg74_1, (128, ), (1, ))
            assert_size_stride(arg75_1, (128, ), (1, ))
            buf138 = reinterpret_tensor(buf135, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [input_30, view_14, full_like_14, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:82
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf138, arg72_1, arg73_1, arg74_1, arg75_1, 3211264, stream=raw_stream0)
            del arg72_1
            del arg73_1
            del arg74_1
            del arg75_1
            buf139 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [input_30, view_14, full_like_14, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:83
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf139, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_30, view_14, full_like_14, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:84
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf138, buf139, buf136, buf137, buf137, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg76_1, (128, 128, 3, 3), (1152, 9, 3, 1))
            buf142 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_18:85
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(arg76_1, buf142, 16384, 9, stream=raw_stream0)
            del arg76_1
            buf143 = reinterpret_tensor(buf138, (32, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_19:86
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_19.run(buf136, buf143, 4096, 784, stream=raw_stream0)
            buf144 = reinterpret_tensor(buf136, (32, 128, 28, 28), (100352, 784, 28, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_20:87
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_20.run(buf143, buf142, buf144, 196, 1, 1, stream=raw_stream0)
            del buf142
            buf145 = reinterpret_tensor(buf143, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf143  # reuse
            buf146 = empty_strided_cuda((4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), torch.bfloat16)
            assert_size_stride(arg77_1, (128, ), (1, ))
            assert_size_stride(arg78_1, (128, ), (1, ))
            assert_size_stride(arg79_1, (128, ), (1, ))
            assert_size_stride(arg80_1, (128, ), (1, ))
            buf147 = reinterpret_tensor(buf144, (4, 8, 128, 28, 28), (802816, 100352, 784, 28, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [input_32, view_15, full_like_15, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16:88
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_16.run(buf147, arg77_1, arg78_1, arg79_1, arg80_1, 3211264, stream=raw_stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            del arg80_1
            buf148 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [input_32, view_15, full_like_15, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17:89
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_17.run(buf148, 802816, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_32, view_15, full_like_15, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_2:90
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_2.run(buf147, buf148, buf145, buf146, buf146, 1.0, 0.0, stream=raw_stream0)
            buf151 = buf132; del buf132  # reuse
            buf153 = reinterpret_tensor(buf147, (32, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf147  # reuse
            buf171 = empty_strided_cuda((32, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_6, input_33, input_37], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_25:91
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_25.run(buf151, buf145, buf153, buf171, 3211264, stream=raw_stream0)
            del buf145
            del buf151
            assert_size_stride(arg81_1, (256, 128, 3, 3), (1152, 9, 3, 1))
            buf152 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_26:92
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_26.run(arg81_1, buf152, 32768, 9, stream=raw_stream0)
            del arg81_1
            buf154 = reinterpret_tensor(buf63, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [out_6, input_33], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_add_convolution_view_27:93
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_convolution_view_27.run(buf153, buf152, buf154, 49, 2, 1, stream=raw_stream0)
            del buf152
            del buf153
            buf155 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            buf156 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg82_1, (256, ), (1, ))
            assert_size_stride(arg83_1, (256, ), (1, ))
            assert_size_stride(arg84_1, (256, ), (1, ))
            assert_size_stride(arg85_1, (256, ), (1, ))
            buf157 = reinterpret_tensor(buf154, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf154  # reuse
            # Topologically Sorted Source Nodes: [input_34, view_16, full_like_16, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:94
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf157, arg82_1, arg83_1, arg84_1, arg85_1, 1605632, stream=raw_stream0)
            del arg82_1
            del arg83_1
            del arg84_1
            del arg85_1
            buf158 = empty_strided_cuda((8, 256, 14, 14), (50176, 196, 14, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_34, view_16, full_like_16, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:95
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf158, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_34, view_16, full_like_16, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:96
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf157, buf158, buf155, buf156, buf156, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg86_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf161 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:97
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg86_1, buf161, 65536, 9, stream=raw_stream0)
            del arg86_1
            buf162 = reinterpret_tensor(buf157, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_31:98
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_31.run(buf155, buf162, 8192, 196, stream=raw_stream0)
            buf163 = reinterpret_tensor(buf155, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf155  # reuse
            # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:99
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf162, buf161, buf163, 49, 2, 1, stream=raw_stream0)
            buf164 = reinterpret_tensor(buf162, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf162  # reuse
            buf165 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg87_1, (256, ), (1, ))
            assert_size_stride(arg88_1, (256, ), (1, ))
            assert_size_stride(arg89_1, (256, ), (1, ))
            assert_size_stride(arg90_1, (256, ), (1, ))
            buf166 = reinterpret_tensor(buf163, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf163  # reuse
            # Topologically Sorted Source Nodes: [input_36, view_17, full_like_17, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:100
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf166, arg87_1, arg88_1, arg89_1, arg90_1, 1605632, stream=raw_stream0)
            del arg87_1
            del arg88_1
            del arg89_1
            del arg90_1
            buf167 = buf158; del buf158  # reuse
            # Topologically Sorted Source Nodes: [input_36, view_17, full_like_17, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:101
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf167, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_36, view_17, full_like_17, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:102
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf166, buf167, buf164, buf165, buf165, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg91_1, (256, 128, 1, 1), (128, 1, 1, 1))
            buf170 = empty_strided_cuda((256, 128, 1, 1), (128, 1, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_33:103
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_33.run(arg91_1, buf170, 32768, stream=raw_stream0)
            del arg91_1
            buf172 = reinterpret_tensor(buf166, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf166  # reuse
            # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_34:104
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_34.run(buf171, buf170, buf172, 98, 2, 1, stream=raw_stream0)
            del buf170
            del buf171
            buf173 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            buf174 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg92_1, (256, ), (1, ))
            assert_size_stride(arg93_1, (256, ), (1, ))
            assert_size_stride(arg94_1, (256, ), (1, ))
            assert_size_stride(arg95_1, (256, ), (1, ))
            buf175 = reinterpret_tensor(buf172, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [input_38, input_39, full_like_18, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:105
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf175, arg92_1, arg93_1, arg94_1, arg95_1, 1605632, stream=raw_stream0)
            del arg92_1
            del arg93_1
            del arg94_1
            del arg95_1
            buf176 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [input_38, input_39, full_like_18, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:106
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf176, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_38, input_39, full_like_18, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:107
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf175, buf176, buf173, buf174, buf174, 1.0, 0.0, stream=raw_stream0)
            buf179 = buf175; del buf175  # reuse
            buf181 = empty_strided_cuda((32, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_7, input_40], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_35:108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_35.run(buf164, buf173, buf179, buf181, 1605632, stream=raw_stream0)
            assert_size_stride(arg96_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf180 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:109
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg96_1, buf180, 65536, 9, stream=raw_stream0)
            del arg96_1
            buf182 = reinterpret_tensor(buf173, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [out_7, input_40], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:110
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf181, buf180, buf182, 49, 2, 1, stream=raw_stream0)
            buf183 = reinterpret_tensor(buf181, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf181  # reuse
            buf184 = buf164; del buf164  # reuse
            assert_size_stride(arg97_1, (256, ), (1, ))
            assert_size_stride(arg98_1, (256, ), (1, ))
            assert_size_stride(arg99_1, (256, ), (1, ))
            assert_size_stride(arg100_1, (256, ), (1, ))
            buf185 = reinterpret_tensor(buf182, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [input_41, view_19, full_like_19, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:111
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf185, arg97_1, arg98_1, arg99_1, arg100_1, 1605632, stream=raw_stream0)
            del arg100_1
            del arg97_1
            del arg98_1
            del arg99_1
            buf186 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [input_41, view_19, full_like_19, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:112
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf186, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_41, view_19, full_like_19, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:113
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf185, buf186, buf183, buf184, buf184, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg101_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf189 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:114
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg101_1, buf189, 65536, 9, stream=raw_stream0)
            del arg101_1
            buf190 = reinterpret_tensor(buf185, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf185  # reuse
            # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_31:115
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_31.run(buf183, buf190, 8192, 196, stream=raw_stream0)
            buf191 = reinterpret_tensor(buf183, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:116
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf190, buf189, buf191, 49, 2, 1, stream=raw_stream0)
            buf192 = reinterpret_tensor(buf190, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf190  # reuse
            buf193 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg102_1, (256, ), (1, ))
            assert_size_stride(arg103_1, (256, ), (1, ))
            assert_size_stride(arg104_1, (256, ), (1, ))
            assert_size_stride(arg105_1, (256, ), (1, ))
            buf194 = reinterpret_tensor(buf191, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [input_43, view_20, full_like_20, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:117
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf194, arg102_1, arg103_1, arg104_1, arg105_1, 1605632, stream=raw_stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            del arg105_1
            buf195 = buf186; del buf186  # reuse
            # Topologically Sorted Source Nodes: [input_43, view_20, full_like_20, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:118
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf195, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_43, view_20, full_like_20, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:119
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf194, buf195, buf192, buf193, buf193, 1.0, 0.0, stream=raw_stream0)
            buf198 = buf179; del buf179  # reuse
            buf200 = reinterpret_tensor(buf194, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [out_8, input_44], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_36:120
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_36.run(buf198, buf192, buf200, 1605632, stream=raw_stream0)
            assert_size_stride(arg106_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf199 = buf189; del buf189  # reuse
            # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:121
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg106_1, buf199, 65536, 9, stream=raw_stream0)
            del arg106_1
            buf201 = reinterpret_tensor(buf192, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [out_8, input_44], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:122
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf200, buf199, buf201, 49, 2, 1, stream=raw_stream0)
            buf202 = reinterpret_tensor(buf200, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf200  # reuse
            buf203 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg107_1, (256, ), (1, ))
            assert_size_stride(arg108_1, (256, ), (1, ))
            assert_size_stride(arg109_1, (256, ), (1, ))
            assert_size_stride(arg110_1, (256, ), (1, ))
            buf204 = reinterpret_tensor(buf201, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf201  # reuse
            # Topologically Sorted Source Nodes: [input_45, view_21, full_like_21, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:123
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf204, arg107_1, arg108_1, arg109_1, arg110_1, 1605632, stream=raw_stream0)
            del arg107_1
            del arg108_1
            del arg109_1
            del arg110_1
            buf205 = buf195; del buf195  # reuse
            # Topologically Sorted Source Nodes: [input_45, view_21, full_like_21, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:124
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf205, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_45, view_21, full_like_21, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:125
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf204, buf205, buf202, buf203, buf203, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg111_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf208 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:126
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg111_1, buf208, 65536, 9, stream=raw_stream0)
            del arg111_1
            buf209 = reinterpret_tensor(buf204, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_31:127
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_31.run(buf202, buf209, 8192, 196, stream=raw_stream0)
            buf210 = reinterpret_tensor(buf202, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf202  # reuse
            # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:128
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf209, buf208, buf210, 49, 2, 1, stream=raw_stream0)
            buf211 = reinterpret_tensor(buf209, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf209  # reuse
            buf212 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg112_1, (256, ), (1, ))
            assert_size_stride(arg113_1, (256, ), (1, ))
            assert_size_stride(arg114_1, (256, ), (1, ))
            assert_size_stride(arg115_1, (256, ), (1, ))
            buf213 = reinterpret_tensor(buf210, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [input_47, view_22, full_like_22, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:129
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf213, arg112_1, arg113_1, arg114_1, arg115_1, 1605632, stream=raw_stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del arg115_1
            buf214 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [input_47, view_22, full_like_22, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:130
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf214, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_47, view_22, full_like_22, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:131
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf213, buf214, buf211, buf212, buf212, 1.0, 0.0, stream=raw_stream0)
            buf217 = buf198; del buf198  # reuse
            buf219 = reinterpret_tensor(buf213, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf213  # reuse
            # Topologically Sorted Source Nodes: [out_9, input_48], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_36:132
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_36.run(buf217, buf211, buf219, 1605632, stream=raw_stream0)
            assert_size_stride(arg116_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf218 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:133
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg116_1, buf218, 65536, 9, stream=raw_stream0)
            del arg116_1
            buf220 = reinterpret_tensor(buf211, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [out_9, input_48], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:134
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf219, buf218, buf220, 49, 2, 1, stream=raw_stream0)
            buf221 = reinterpret_tensor(buf219, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf219  # reuse
            buf222 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg117_1, (256, ), (1, ))
            assert_size_stride(arg118_1, (256, ), (1, ))
            assert_size_stride(arg119_1, (256, ), (1, ))
            assert_size_stride(arg120_1, (256, ), (1, ))
            buf223 = reinterpret_tensor(buf220, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf220  # reuse
            # Topologically Sorted Source Nodes: [input_49, view_23, full_like_23, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:135
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf223, arg117_1, arg118_1, arg119_1, arg120_1, 1605632, stream=raw_stream0)
            del arg117_1
            del arg118_1
            del arg119_1
            del arg120_1
            buf224 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [input_49, view_23, full_like_23, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:136
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf224, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_49, view_23, full_like_23, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:137
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf223, buf224, buf221, buf222, buf222, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg121_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf227 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:138
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg121_1, buf227, 65536, 9, stream=raw_stream0)
            del arg121_1
            buf228 = reinterpret_tensor(buf223, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_31:139
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_31.run(buf221, buf228, 8192, 196, stream=raw_stream0)
            buf229 = reinterpret_tensor(buf221, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:140
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf228, buf227, buf229, 49, 2, 1, stream=raw_stream0)
            buf230 = reinterpret_tensor(buf228, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf228  # reuse
            buf231 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg122_1, (256, ), (1, ))
            assert_size_stride(arg123_1, (256, ), (1, ))
            assert_size_stride(arg124_1, (256, ), (1, ))
            assert_size_stride(arg125_1, (256, ), (1, ))
            buf232 = reinterpret_tensor(buf229, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [input_51, view_24, full_like_24, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:141
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf232, arg122_1, arg123_1, arg124_1, arg125_1, 1605632, stream=raw_stream0)
            del arg122_1
            del arg123_1
            del arg124_1
            del arg125_1
            buf233 = buf224; del buf224  # reuse
            # Topologically Sorted Source Nodes: [input_51, view_24, full_like_24, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:142
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf233, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_51, view_24, full_like_24, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:143
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf232, buf233, buf230, buf231, buf231, 1.0, 0.0, stream=raw_stream0)
            buf236 = buf217; del buf217  # reuse
            buf238 = reinterpret_tensor(buf232, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [out_10, input_52], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_36:144
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_36.run(buf236, buf230, buf238, 1605632, stream=raw_stream0)
            assert_size_stride(arg126_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf237 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:145
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg126_1, buf237, 65536, 9, stream=raw_stream0)
            del arg126_1
            buf239 = reinterpret_tensor(buf230, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [out_10, input_52], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:146
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf238, buf237, buf239, 49, 2, 1, stream=raw_stream0)
            buf240 = reinterpret_tensor(buf238, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf238  # reuse
            buf241 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg127_1, (256, ), (1, ))
            assert_size_stride(arg128_1, (256, ), (1, ))
            assert_size_stride(arg129_1, (256, ), (1, ))
            assert_size_stride(arg130_1, (256, ), (1, ))
            buf242 = reinterpret_tensor(buf239, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf239  # reuse
            # Topologically Sorted Source Nodes: [input_53, view_25, full_like_25, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:147
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf242, arg127_1, arg128_1, arg129_1, arg130_1, 1605632, stream=raw_stream0)
            del arg127_1
            del arg128_1
            del arg129_1
            del arg130_1
            buf243 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [input_53, view_25, full_like_25, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:148
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf243, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_53, view_25, full_like_25, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:149
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf242, buf243, buf240, buf241, buf241, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg131_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf246 = buf237; del buf237  # reuse
            # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:150
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg131_1, buf246, 65536, 9, stream=raw_stream0)
            del arg131_1
            buf247 = reinterpret_tensor(buf242, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_31:151
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_31.run(buf240, buf247, 8192, 196, stream=raw_stream0)
            buf248 = reinterpret_tensor(buf240, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:152
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf247, buf246, buf248, 49, 2, 1, stream=raw_stream0)
            buf249 = reinterpret_tensor(buf247, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf247  # reuse
            buf250 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg132_1, (256, ), (1, ))
            assert_size_stride(arg133_1, (256, ), (1, ))
            assert_size_stride(arg134_1, (256, ), (1, ))
            assert_size_stride(arg135_1, (256, ), (1, ))
            buf251 = reinterpret_tensor(buf248, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf248  # reuse
            # Topologically Sorted Source Nodes: [input_55, view_26, full_like_26, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:153
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf251, arg132_1, arg133_1, arg134_1, arg135_1, 1605632, stream=raw_stream0)
            del arg132_1
            del arg133_1
            del arg134_1
            del arg135_1
            buf252 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [input_55, view_26, full_like_26, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:154
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf252, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_55, view_26, full_like_26, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:155
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf251, buf252, buf249, buf250, buf250, 1.0, 0.0, stream=raw_stream0)
            buf255 = buf236; del buf236  # reuse
            buf257 = reinterpret_tensor(buf251, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf251  # reuse
            # Topologically Sorted Source Nodes: [out_11, input_56], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_36:156
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_36.run(buf255, buf249, buf257, 1605632, stream=raw_stream0)
            assert_size_stride(arg136_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf256 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:157
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg136_1, buf256, 65536, 9, stream=raw_stream0)
            del arg136_1
            buf258 = reinterpret_tensor(buf249, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf249  # reuse
            # Topologically Sorted Source Nodes: [out_11, input_56], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:158
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf257, buf256, buf258, 49, 2, 1, stream=raw_stream0)
            buf259 = reinterpret_tensor(buf257, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf257  # reuse
            buf260 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg137_1, (256, ), (1, ))
            assert_size_stride(arg138_1, (256, ), (1, ))
            assert_size_stride(arg139_1, (256, ), (1, ))
            assert_size_stride(arg140_1, (256, ), (1, ))
            buf261 = reinterpret_tensor(buf258, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [input_57, view_27, full_like_27, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:159
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf261, arg137_1, arg138_1, arg139_1, arg140_1, 1605632, stream=raw_stream0)
            del arg137_1
            del arg138_1
            del arg139_1
            del arg140_1
            buf262 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [input_57, view_27, full_like_27, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:160
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf262, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_57, view_27, full_like_27, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:161
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf261, buf262, buf259, buf260, buf260, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg141_1, (256, 256, 3, 3), (2304, 9, 3, 1))
            buf265 = buf256; del buf256  # reuse
            # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_30:162
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_30.run(arg141_1, buf265, 65536, 9, stream=raw_stream0)
            del arg141_1
            buf266 = reinterpret_tensor(buf261, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_31:163
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_31.run(buf259, buf266, 8192, 196, stream=raw_stream0)
            buf267 = reinterpret_tensor(buf259, (32, 256, 14, 14), (50176, 196, 14, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_32:164
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_32.run(buf266, buf265, buf267, 49, 2, 1, stream=raw_stream0)
            del buf265
            buf268 = reinterpret_tensor(buf266, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf266  # reuse
            buf269 = empty_strided_cuda((4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), torch.bfloat16)
            assert_size_stride(arg142_1, (256, ), (1, ))
            assert_size_stride(arg143_1, (256, ), (1, ))
            assert_size_stride(arg144_1, (256, ), (1, ))
            assert_size_stride(arg145_1, (256, ), (1, ))
            buf270 = reinterpret_tensor(buf267, (4, 8, 256, 14, 14), (401408, 50176, 196, 14, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [input_59, view_28, full_like_28, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28:165
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_28.run(buf270, arg142_1, arg143_1, arg144_1, arg145_1, 1605632, stream=raw_stream0)
            del arg142_1
            del arg143_1
            del arg144_1
            del arg145_1
            buf271 = buf262; del buf262  # reuse
            # Topologically Sorted Source Nodes: [input_59, view_28, full_like_28, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29:166
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_29.run(buf271, 401408, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_59, view_28, full_like_28, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_3:167
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_3.run(buf270, buf271, buf268, buf269, buf269, 1.0, 0.0, stream=raw_stream0)
            del buf271
            buf274 = buf255; del buf255  # reuse
            buf276 = reinterpret_tensor(buf270, (32, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf270  # reuse
            buf294 = empty_strided_cuda((32, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_12, input_60, input_64], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_37:168
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_37.run(buf274, buf268, buf276, buf294, 1605632, stream=raw_stream0)
            del buf268
            del buf274
            assert_size_stride(arg146_1, (512, 256, 3, 3), (2304, 9, 3, 1))
            buf275 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_38:169
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(arg146_1, buf275, 131072, 9, stream=raw_stream0)
            del arg146_1
            buf277 = reinterpret_tensor(buf148, (32, 512, 7, 7), (25088, 49, 7, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [out_12, input_60], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_add_convolution_view_39:170
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_convolution_view_39.run(buf276, buf275, buf277, 25, 4, 1, stream=raw_stream0)
            del buf275
            del buf276
            buf278 = empty_strided_cuda((4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), torch.bfloat16)
            buf279 = empty_strided_cuda((4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), torch.bfloat16)
            assert_size_stride(arg147_1, (512, ), (1, ))
            assert_size_stride(arg148_1, (512, ), (1, ))
            assert_size_stride(arg149_1, (512, ), (1, ))
            assert_size_stride(arg150_1, (512, ), (1, ))
            buf280 = reinterpret_tensor(buf277, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [input_61, view_29, full_like_29, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40:171
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40.run(buf280, arg147_1, arg148_1, arg149_1, arg150_1, 802816, stream=raw_stream0)
            del arg147_1
            del arg148_1
            del arg149_1
            del arg150_1
            buf281 = empty_strided_cuda((8, 512, 7, 7), (25088, 49, 7, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_61, view_29, full_like_29, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41:172
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41.run(buf281, 200704, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_61, view_29, full_like_29, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_4:173
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_4.run(buf280, buf281, buf278, buf279, buf279, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg151_1, (512, 512, 3, 3), (4608, 9, 3, 1))
            buf284 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_42:174
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(arg151_1, buf284, 262144, 9, stream=raw_stream0)
            del arg151_1
            buf285 = reinterpret_tensor(buf280, (32, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf280  # reuse
            # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_43:175
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_43.run(buf278, buf285, 16384, 49, stream=raw_stream0)
            buf286 = reinterpret_tensor(buf278, (32, 512, 7, 7), (25088, 49, 7, 1), 0); del buf278  # reuse
            # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_44:176
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_44.run(buf285, buf284, buf286, 25, 4, 1, stream=raw_stream0)
            buf287 = reinterpret_tensor(buf285, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf285  # reuse
            buf288 = empty_strided_cuda((4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), torch.bfloat16)
            assert_size_stride(arg152_1, (512, ), (1, ))
            assert_size_stride(arg153_1, (512, ), (1, ))
            assert_size_stride(arg154_1, (512, ), (1, ))
            assert_size_stride(arg155_1, (512, ), (1, ))
            buf289 = reinterpret_tensor(buf286, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf286  # reuse
            # Topologically Sorted Source Nodes: [input_63, view_30, full_like_30, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40:177
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40.run(buf289, arg152_1, arg153_1, arg154_1, arg155_1, 802816, stream=raw_stream0)
            del arg152_1
            del arg153_1
            del arg154_1
            del arg155_1
            buf290 = buf281; del buf281  # reuse
            # Topologically Sorted Source Nodes: [input_63, view_30, full_like_30, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41:178
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41.run(buf290, 200704, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_63, view_30, full_like_30, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_4:179
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_4.run(buf289, buf290, buf287, buf288, buf288, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg156_1, (512, 256, 1, 1), (256, 1, 1, 1))
            buf293 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_45:180
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_45.run(arg156_1, buf293, 131072, stream=raw_stream0)
            del arg156_1
            buf295 = reinterpret_tensor(buf289, (32, 512, 7, 7), (25088, 49, 7, 1), 0); del buf289  # reuse
            # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_46:181
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_46.run(buf294, buf293, buf295, 25, 4, 1, stream=raw_stream0)
            del buf293
            del buf294
            buf296 = empty_strided_cuda((4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), torch.bfloat16)
            buf297 = empty_strided_cuda((4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), torch.bfloat16)
            assert_size_stride(arg157_1, (512, ), (1, ))
            assert_size_stride(arg158_1, (512, ), (1, ))
            assert_size_stride(arg159_1, (512, ), (1, ))
            assert_size_stride(arg160_1, (512, ), (1, ))
            buf298 = reinterpret_tensor(buf295, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf295  # reuse
            # Topologically Sorted Source Nodes: [input_65, input_66, full_like_31, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40:182
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40.run(buf298, arg157_1, arg158_1, arg159_1, arg160_1, 802816, stream=raw_stream0)
            del arg157_1
            del arg158_1
            del arg159_1
            del arg160_1
            buf299 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [input_65, input_66, full_like_31, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41:183
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41.run(buf299, 200704, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_65, input_66, full_like_31, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_4:184
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_4.run(buf298, buf299, buf296, buf297, buf297, 1.0, 0.0, stream=raw_stream0)
            buf302 = buf298; del buf298  # reuse
            buf304 = empty_strided_cuda((32, 512, 7, 7), (25088, 1, 3584, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_13, input_67], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_47:185
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_47.run(buf287, buf296, buf302, buf304, 802816, stream=raw_stream0)
            assert_size_stride(arg161_1, (512, 512, 3, 3), (4608, 9, 3, 1))
            buf303 = buf284; del buf284  # reuse
            # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_42:186
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(arg161_1, buf303, 262144, 9, stream=raw_stream0)
            del arg161_1
            buf305 = reinterpret_tensor(buf296, (32, 512, 7, 7), (25088, 49, 7, 1), 0); del buf296  # reuse
            # Topologically Sorted Source Nodes: [out_13, input_67], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_44:187
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_44.run(buf304, buf303, buf305, 25, 4, 1, stream=raw_stream0)
            buf306 = reinterpret_tensor(buf304, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf304  # reuse
            buf307 = buf287; del buf287  # reuse
            assert_size_stride(arg162_1, (512, ), (1, ))
            assert_size_stride(arg163_1, (512, ), (1, ))
            assert_size_stride(arg164_1, (512, ), (1, ))
            assert_size_stride(arg165_1, (512, ), (1, ))
            buf308 = reinterpret_tensor(buf305, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf305  # reuse
            # Topologically Sorted Source Nodes: [input_68, view_32, full_like_32, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40:188
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40.run(buf308, arg162_1, arg163_1, arg164_1, arg165_1, 802816, stream=raw_stream0)
            del arg162_1
            del arg163_1
            del arg164_1
            del arg165_1
            buf309 = buf299; del buf299  # reuse
            # Topologically Sorted Source Nodes: [input_68, view_32, full_like_32, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41:189
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41.run(buf309, 200704, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_68, view_32, full_like_32, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_4:190
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_4.run(buf308, buf309, buf306, buf307, buf307, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg166_1, (512, 512, 3, 3), (4608, 9, 3, 1))
            buf312 = buf303; del buf303  # reuse
            # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_42:191
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(arg166_1, buf312, 262144, 9, stream=raw_stream0)
            del arg166_1
            buf313 = reinterpret_tensor(buf308, (32, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf308  # reuse
            # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_43:192
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_43.run(buf306, buf313, 16384, 49, stream=raw_stream0)
            buf314 = reinterpret_tensor(buf306, (32, 512, 7, 7), (25088, 49, 7, 1), 0); del buf306  # reuse
            # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_44:193
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_44.run(buf313, buf312, buf314, 25, 4, 1, stream=raw_stream0)
            del buf312
            buf315 = reinterpret_tensor(buf313, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf313  # reuse
            buf316 = empty_strided_cuda((4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), torch.bfloat16)
            assert_size_stride(arg167_1, (512, ), (1, ))
            assert_size_stride(arg168_1, (512, ), (1, ))
            assert_size_stride(arg169_1, (512, ), (1, ))
            assert_size_stride(arg170_1, (512, ), (1, ))
            buf317 = reinterpret_tensor(buf314, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf314  # reuse
            # Topologically Sorted Source Nodes: [input_70, view_33, full_like_33, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40:194
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40.run(buf317, arg167_1, arg168_1, arg169_1, arg170_1, 802816, stream=raw_stream0)
            del arg167_1
            del arg168_1
            del arg169_1
            del arg170_1
            buf318 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [input_70, view_33, full_like_33, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41:195
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41.run(buf318, 200704, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_70, view_33, full_like_33, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_4:196
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_4.run(buf317, buf318, buf315, buf316, buf316, 1.0, 0.0, stream=raw_stream0)
            buf321 = buf302; del buf302  # reuse
            buf323 = reinterpret_tensor(buf317, (32, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf317  # reuse
            # Topologically Sorted Source Nodes: [out_14, input_71], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_view_48:197
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_view_48.run(buf321, buf315, buf323, 802816, stream=raw_stream0)
            assert_size_stride(arg171_1, (512, 512, 3, 3), (4608, 9, 3, 1))
            buf322 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_42:198
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(arg171_1, buf322, 262144, 9, stream=raw_stream0)
            del arg171_1
            buf324 = reinterpret_tensor(buf315, (32, 512, 7, 7), (25088, 49, 7, 1), 0); del buf315  # reuse
            # Topologically Sorted Source Nodes: [out_14, input_71], Original ATen: [aten.add, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_44:199
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_44.run(buf323, buf322, buf324, 25, 4, 1, stream=raw_stream0)
            del buf322
            buf325 = reinterpret_tensor(buf323, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf323  # reuse
            buf326 = empty_strided_cuda((4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), torch.bfloat16)
            assert_size_stride(arg172_1, (512, ), (1, ))
            assert_size_stride(arg173_1, (512, ), (1, ))
            assert_size_stride(arg174_1, (512, ), (1, ))
            assert_size_stride(arg175_1, (512, ), (1, ))
            buf327 = reinterpret_tensor(buf324, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf324  # reuse
            # Topologically Sorted Source Nodes: [input_72, view_34, full_like_34, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40:200
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40.run(buf327, arg172_1, arg173_1, arg174_1, arg175_1, 802816, stream=raw_stream0)
            del arg172_1
            del arg173_1
            del arg174_1
            del arg175_1
            buf328 = buf318; del buf318  # reuse
            # Topologically Sorted Source Nodes: [input_72, view_34, full_like_34, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41:201
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41.run(buf328, 200704, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_72, view_34, full_like_34, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_4:202
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_4.run(buf327, buf328, buf325, buf326, buf326, 1.0, 0.0, stream=raw_stream0)
            del buf328
            assert_size_stride(arg176_1, (512, 512, 3, 3), (4608, 9, 3, 1))
            buf331 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_42:203
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(arg176_1, buf331, 262144, 9, stream=raw_stream0)
            del arg176_1
            buf332 = reinterpret_tensor(buf327, (32, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf327  # reuse
            # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__to_copy_convolution_view_43:204
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_convolution_view_43.run(buf325, buf332, 16384, 49, stream=raw_stream0)
            buf333 = reinterpret_tensor(buf325, (32, 512, 7, 7), (25088, 49, 7, 1), 0); del buf325  # reuse
            # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_view_44:205
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_view_44.run(buf332, buf331, buf333, 25, 4, 1, stream=raw_stream0)
            del buf331
            buf334 = reinterpret_tensor(buf332, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf332  # reuse
            buf335 = empty_strided_cuda((4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), torch.bfloat16)
            assert_size_stride(arg177_1, (512, ), (1, ))
            assert_size_stride(arg178_1, (512, ), (1, ))
            assert_size_stride(arg179_1, (512, ), (1, ))
            assert_size_stride(arg180_1, (512, ), (1, ))
            buf336 = reinterpret_tensor(buf333, (4, 8, 512, 7, 7), (200704, 25088, 49, 7, 1), 0); del buf333  # reuse
            # Topologically Sorted Source Nodes: [input_74, view_35, full_like_35, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40:206
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_40.run(buf336, arg177_1, arg178_1, arg179_1, arg180_1, 802816, stream=raw_stream0)
            del arg177_1
            del arg178_1
            del arg179_1
            del arg180_1
            buf337 = empty_strided_cuda((8, 512, 7, 7), (25088, 49, 7, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_74, view_35, full_like_35, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41:207
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_41.run(buf337, 200704, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_74, view_35, full_like_35, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_if_forward_kernel_4:208
            raw_stream0 = get_raw_stream(0)
            _multistep_if_forward_kernel_4.run(buf336, buf337, buf334, buf335, buf335, 1.0, 0.0, stream=raw_stream0)
            del buf336
            del buf337
            buf340 = empty_strided_cuda((32, 512, 1, 1), (512, 1, 16384, 16384), torch.float32)
            # Topologically Sorted Source Nodes: [out_15, input_75], Original ATen: [aten.add, aten.view, aten.mean]
            # [Provenance debug handles] triton_per_fused_add_mean_view_49:209
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_mean_view_49.run(buf334, buf321, buf340, 16384, 49, stream=raw_stream0)
            del buf321
            del buf334
            buf341 = empty_strided_cuda((8, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [out_15, input_75, x_4, x_5, mean], Original ATen: [aten.add, aten.view, aten.mean]
            # [Provenance debug handles] triton_poi_fused_add_mean_view_50:210
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mean_view_50.run(buf340, buf341, 4096, stream=raw_stream0)
            del buf340
            assert_size_stride(arg181_1, (1000, 512), (512, 1))
            buf342 = empty_strided_cuda((1000, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_51:211
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_51.run(arg181_1, buf342, 512000, stream=raw_stream0)
            del arg181_1
            assert_size_stride(arg182_1, (1000, ), (1, ))
            buf343 = empty_strided_cuda((1000, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_52:212
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_52.run(arg182_1, buf343, 1000, stream=raw_stream0)
            del arg182_1
            buf344 = empty_strided_cuda((8, 1000), (1000, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, out_15, input_75, x_4, x_5, mean], Original ATen: [aten._to_copy, aten.add, aten.view, aten.mean, aten.t, aten.addmm]
            # [Provenance debug handles] triton_tem_fused__to_copy_add_addmm_mean_t_view_53:213
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_addmm_mean_t_view_53.run(buf343, buf341, buf342, buf344, 32, 1, 1, stream=raw_stream0)
            del buf341
            del buf342
            del buf343
        return (buf344, reinterpret_tensor(buf5, (8, 64, 112, 112), (802816, 12544, 112, 1), 19267584), reinterpret_tensor(buf14, (8, 64, 56, 56), (200704, 3136, 56, 1), 4816896), reinterpret_tensor(buf23, (8, 64, 56, 56), (200704, 3136, 56, 1), 4816896), reinterpret_tensor(buf33, (8, 64, 56, 56), (200704, 3136, 56, 1), 4816896), reinterpret_tensor(buf42, (8, 64, 56, 56), (200704, 3136, 56, 1), 4816896), reinterpret_tensor(buf52, (8, 64, 56, 56), (200704, 3136, 56, 1), 4816896), reinterpret_tensor(buf61, (8, 64, 56, 56), (200704, 3136, 56, 1), 4816896), reinterpret_tensor(buf71, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf80, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf89, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf99, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf108, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf118, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf127, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf137, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf146, (8, 128, 28, 28), (100352, 784, 28, 1), 2408448), reinterpret_tensor(buf156, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf165, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf174, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf184, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf193, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf203, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf212, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf222, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf231, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf241, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf250, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf260, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf269, (8, 256, 14, 14), (50176, 196, 14, 1), 1204224), reinterpret_tensor(buf279, (8, 512, 7, 7), (25088, 49, 7, 1), 602112), reinterpret_tensor(buf288, (8, 512, 7, 7), (25088, 49, 7, 1), 602112), reinterpret_tensor(buf297, (8, 512, 7, 7), (25088, 49, 7, 1), 602112), reinterpret_tensor(buf307, (8, 512, 7, 7), (25088, 49, 7, 1), 602112), reinterpret_tensor(buf316, (8, 512, 7, 7), (25088, 49, 7, 1), 602112), reinterpret_tensor(buf326, (8, 512, 7, 7), (25088, 49, 7, 1), 602112), reinterpret_tensor(buf335, (8, 512, 7, 7), (25088, 49, 7, 1), 602112), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
