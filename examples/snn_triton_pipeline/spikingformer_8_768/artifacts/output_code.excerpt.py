# ===== EXCERPT of inductor output_code.py (full 7534 lines on a100) =====

## --- (A) spikingjelly LIF Triton kernel embedded+autotuned by inductor (def at line 432) ---
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['301056', '1', '1'], 'cpp': ['301056L', '1L', '1L'], 'python_slow': ['301056', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['150528', '1', '1'], 'cpp': ['150528L', '1L', '1L'], 'python_slow': ['150528', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['150528', '1', '1'], 'cpp': ['150528L', '1L', '1L'], 'python_slow': ['150528', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['75264', '1', '1'], 'cpp': ['75264L', '1L', '1L'], 'python_slow': ['75264', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'decay_input', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_lif_forward_kernel_0', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'tau': 'fp64', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'decay_input': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 4, 'NCL': 38535168, 'dtype': triton.language.bfloat16, 'decay_input': True, 'soft_reset': False, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _multistep_lif_forward_kernel(
    x_seq_ptr,  # [T, NCL]
    v_init_ptr,  # [1, NCL]
    s_seq_ptr,
    h_seq_ptr,
    v_seq_ptr,
    tau,
    v_threshold,
    v_reset,
    T: tl.constexpr,
    NCL: tl.constexpr,
    BLOCK_NCL: tl.constexpr,
    dtype: tl.constexpr,
    decay_input: tl.constexpr,
    soft_reset: tl.constexpr,
    save_intermediates: tl.constexpr,
):
    pid_ncl = tl.program_id(0)
    ncl_offset = pid_ncl * BLOCK_NCL

    r_tau = tl.full([1], 1.0 / tau, dtype=dtype)

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

        if decay_input:
            h = v + r_tau * (v_reset - v + x)
        else:
            h = v + r_tau * (v_reset - v) + x
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

## --- (B) call(): GPU launch sequence (head, def at line 4564) ---
    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_misaligned(arg0_1)
            buf0 = empty_strided_cuda((32, 3, 224, 224), (150528, 1, 672, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [unsqueeze, x, flatten, x_1], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_repeat_unsqueeze_view_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_repeat_unsqueeze_view_0.run(arg0_1, buf0, 96, 50176, stream=raw_stream0)
            del arg0_1
            assert_size_stride(arg1_1, (96, 3, 3, 3), (27, 9, 3, 1))
            buf1 = empty_strided_cuda((96, 3, 3, 3), (27, 1, 9, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_1:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg1_1, buf1, 288, 9, stream=raw_stream0)
            del arg1_1
            buf2 = empty_strided_cuda((32, 96, 224, 224), (4816896, 50176, 224, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [unsqueeze, x, flatten, x_1], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_repeat_unsqueeze_view_2:3
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_repeat_unsqueeze_view_2.run(buf0, buf1, buf2, 25088, 1, 1, stream=raw_stream0)
            del buf1
            buf3 = empty_strided_cuda((4, 8, 96, 224, 224), (38535168, 4816896, 50176, 224, 1), torch.bfloat16)
            buf4 = empty_strided_cuda((4, 8, 96, 224, 224), (38535168, 4816896, 50176, 224, 1), torch.bfloat16)
            assert_size_stride(arg2_1, (96, ), (1, ))
            assert_size_stride(arg3_1, (96, ), (1, ))
            assert_size_stride(arg4_1, (96, ), (1, ))
            assert_size_stride(arg5_1, (96, ), (1, ))
            buf5 = reinterpret_tensor(buf2, (4, 8, 96, 224, 224), (38535168, 4816896, 50176, 224, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [batch_norm, reshape, full_like, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_3:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_3.run(buf5, arg2_1, arg3_1, arg4_1, arg5_1, 154140672, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf6 = empty_strided_cuda((8, 96, 224, 224), (4816896, 50176, 224, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [batch_norm, reshape, full_like, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_4:5
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_full_like_view_4.run(buf6, 38535168, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [batch_norm, reshape, full_like, ], Original ATen: [aten._native_batch_norm_legit_no_training, aten.view, aten.full_like]
            # [Provenance debug handles] _multistep_lif_forward_kernel_0:6
            raw_stream0 = get_raw_stream(0)
            _multistep_lif_forward_kernel_0.run(buf5, buf6, buf3, buf4, buf4, 2.0, 1.0, 0.0, stream=raw_stream0)
            del buf5
            buf9 = reinterpret_tensor(buf6, (32, 96, 112, 112), (1204224, 1, 10752, 96), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.view, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused_max_pool2d_with_indices_view_5:7
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_view_5.run(buf3, buf9, 3072, 12544, stream=raw_stream0)
            del buf3
            assert_size_stride(arg6_1, (192, 96, 3, 3), (864, 9, 3, 1))
            buf10 = empty_strided_cuda((192, 96, 3, 3), (864, 1, 288, 96), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_6:8
