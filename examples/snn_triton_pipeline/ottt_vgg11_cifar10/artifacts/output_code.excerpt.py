# ===== EXCERPT of inductor output_code.py (full 4503 lines on a100) =====

## --- (A) spikingjelly LIF Triton kernel (def line 588) ---
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['4096', '1', '1'], 'cpp': ['4096L', '1L', '1L'], 'python_slow': ['4096', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['2048', '1', '1'], 'cpp': ['2048L', '1L', '1L'], 'python_slow': ['2048', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['2048', '1', '1'], 'cpp': ['2048L', '1L', '1L'], 'python_slow': ['2048', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['1024', '1', '1'], 'cpp': ['1024L', '1L', '1L'], 'python_slow': ['1024', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'decay_input', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_lif_forward_kernel_0', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'tau': 'fp64', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'decay_input': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 6, 'NCL': 524288, 'dtype': triton.language.bfloat16, 'decay_input': False, 'soft_reset': True, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
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

## --- (B) call(): real GPU launch sequence (WS mean/var -> WS conv(triton) -> LIF -> Scale -> ... -> head mm). extern_kernels=0 ---
    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1 = args
        args.clear()
        assert_size_stride(arg1_1, (64, 3, 3, 3), (27, 9, 3, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((64, 1, 1, 1), (1, 64, 64, 64), torch.float32)
            buf2 = empty_strided_cuda((64, 1, 1, 1), (1, 64, 64, 64), torch.float32)
            # Topologically Sorted Source Nodes: [mean, var], Original ATen: [aten.mean, aten.var]
            # [Provenance debug handles] triton_per_fused_mean_var_0:1
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mean_var_0.run(arg1_1, buf0, buf2, 64, 27, stream=raw_stream0)
        buf4 = empty_strided_cpu((), (), torch.int64)
        assert_size_stride(arg0_1, (8, 3, 32, 32), (3072, 1024, 32, 1))
        # [Provenance debug handles] cpp_fused_lift_fresh_prod_1:2
        cpp_fused_lift_fresh_prod_1(buf4)
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_misaligned(arg0_1)
            buf5 = empty_strided_cuda((48, 3, 32, 32), (3072, 1, 96, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [unsqueeze, x, y, input_1], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_repeat_unsqueeze_view_2:3
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_repeat_unsqueeze_view_2.run(arg0_1, buf5, 144, 1024, stream=raw_stream0)
            del arg0_1
            assert_size_stride(arg2_1, (64, 1, 1, 1), (1, 1, 1, 1))
            buf6 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mean, sub, var, mul, add, pow_1, weight, weight_1, input_1], Original ATen: [aten.mean, aten.sub, aten.var, aten.mul, aten.add, aten.pow, aten.div, aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_div_mean_mul_pow_sub_var_3:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_div_mean_mul_pow_sub_var_3.run(arg1_1, buf0, buf2, buf4.item(), arg2_1, buf6, 192, 9, stream=raw_stream0)
            del arg1_1
            del arg2_1
            del buf0
            del buf2
            buf7 = empty_strided_cuda((48, 64, 32, 32), (65536, 1024, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [unsqueeze, x, y, input_1, mean, sub, var, mul, add, pow_1, weight, weight_1], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy, aten.mean, aten.sub, aten.var, aten.mul, aten.add, aten.pow, aten.div, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_add_convolution_div_mean_mul_pow_repeat_sub_unsqueeze_var_view_4:5
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_add_convolution_div_mean_mul_pow_repeat_sub_unsqueeze_var_view_4.run(buf5, buf6, buf7, 192, 1, 1, stream=raw_stream0)
            del buf5
            del buf6
            buf8 = empty_strided_cuda((6, 8, 64, 32, 32), (524288, 65536, 1024, 32, 1), torch.bfloat16)
            buf9 = empty_strided_cuda((6, 8, 64, 32, 32), (524288, 65536, 1024, 32, 1), torch.bfloat16)
            assert_size_stride(arg3_1, (64, ), (1, ))
            buf10 = reinterpret_tensor(buf7, (6, 8, 64, 32, 32), (524288, 65536, 1024, 32, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [unsqueeze, x, y, input_1, mean, sub, var, mul, add, pow_1, weight, weight_1, input_2, full_like, ], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy, aten.mean, aten.sub, aten.var, aten.mul, aten.add, aten.pow, aten.div, aten.convolution, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_div_full_like_mean_mul_pow_repeat_sub_unsqueeze_var_view_5:6
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_div_full_like_mean_mul_pow_repeat_sub_unsqueeze_var_view_5.run(buf10, arg3_1, 3145728, stream=raw_stream0)
            del arg3_1
            buf11 = empty_strided_cuda((8, 64, 32, 32), (65536, 1024, 32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [unsqueeze, x, y, input_1, mean, sub, var, mul, add, pow_1, weight, weight_1, input_2, full_like, ], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy, aten.mean, aten.sub, aten.var, aten.mul, aten.add, aten.pow, aten.div, aten.convolution, aten.full_like]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_div_full_like_mean_mul_pow_repeat_sub_unsqueeze_var_view_6:7
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_div_full_like_mean_mul_pow_repeat_sub_unsqueeze_var_view_6.run(buf11, 524288, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [unsqueeze, x, y, input_1, mean, sub, var, mul, add, pow_1, weight, weight_1, input_2, full_like, ], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy, aten.mean, aten.sub, aten.var, aten.mul, aten.add, aten.pow, aten.div, aten.convolution, aten.full_like]
            # [Provenance debug handles] _multistep_lif_forward_kernel_0:8
            raw_stream0 = get_raw_stream(0)
            _multistep_lif_forward_kernel_0.run(buf10, buf11, buf8, buf9, buf9, 2.0, 1.0, 0.0, stream=raw_stream0)
            assert_size_stride(arg4_1, (128, 64, 3, 3), (576, 9, 3, 1))
            buf14 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
            buf16 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
