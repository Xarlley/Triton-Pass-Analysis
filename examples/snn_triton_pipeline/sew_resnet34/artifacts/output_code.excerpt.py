# ===== EXCERPT of inductor output_code.py (full 6215 lines on a100) =====

## --- (A) the spikingjelly neuron Triton kernel, embedded + autotuned by inductor ---
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

## --- (B) a fused pointwise kernel: conv-bias + SEW residual add + dtype cast ---
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

## --- (C) call(): the real GPU kernel launch sequence (head) ---
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
