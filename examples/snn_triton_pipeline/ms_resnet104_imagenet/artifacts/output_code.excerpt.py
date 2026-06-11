# ===== EXCERPT of inductor output_code.py (full 11289 lines on a100) =====
# extern_kernels.{mm,bmm,convolution} count in full file = 0  (=> 0 cublas / 0 cudnn).

## --- (A) spikingjelly LIF Triton kernel def (line 851) ---
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['50176', '1', '1'], 'cpp': ['50176L', '1L', '1L'], 'python_slow': ['50176', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['25088', '1', '1'], 'cpp': ['25088L', '1L', '1L'], 'python_slow': ['25088', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['25088', '1', '1'], 'cpp': ['25088L', '1L', '1L'], 'python_slow': ['25088', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['12544', '1', '1'], 'cpp': ['12544L', '1L', '1L'], 'python_slow': ['12544', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'decay_input', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_lif_forward_kernel_0', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'tau': 'fp64', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'decay_input': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 6, 'NCL': 6422528, 'dtype': triton.language.bfloat16, 'decay_input': False, 'soft_reset': False, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
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

## --- (B) real call() region with the LIF launch (around line 6299). tau/vth/vreset are runtime fp64 args ---
            buf10 = empty_strided_cuda((6, 8, 64, 112, 112), (6422528, 802816, 12544, 112, 1), torch.bfloat16)
            buf11 = empty_strided_cuda((6, 8, 64, 112, 112), (6422528, 802816, 12544, 112, 1), torch.bfloat16)
            buf13 = empty_strided_cuda((8, 64, 112, 112), (802816, 12544, 112, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2, _generalized_scatter_3, _generalized_scatter_4, _generalized_scatter_5, y, y_1, y_2, input_4, transpose, contiguous, transpose_1, y_3, y_4, transpose_2, contiguous_3, input_5, multistep_lif_inference_default, ], Original ATen: [aten.zeros, aten.view, aten._to_copy, aten.convolution, aten.transpose, aten.clone, aten._native_batch_norm_legit_no_training]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training__to_copy_clone_convolution_transpose_view_zeros_7:10
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training__to_copy_clone_convolution_transpose_view_zeros_7.run(buf13, 6422528, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [input_1, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2, _generalized_scatter_3, _generalized_scatter_4, _generalized_scatter_5, y, y_1, y_2, input_4, transpose, contiguous, transpose_1, y_3, y_4, transpose_2, contiguous_3, input_5, multistep_lif_inference_default, ], Original ATen: [aten.zeros, aten.view, aten._to_copy, aten.convolution, aten.transpose, aten.clone, aten._native_batch_norm_legit_no_training]
            # [Provenance debug handles] _multistep_lif_forward_kernel_0:11
            raw_stream0 = get_raw_stream(0)
            _multistep_lif_forward_kernel_0.run(buf12, buf13, buf10, buf11, buf11, 1.3333333333333333, 0.5, 0.0, stream=raw_stream0)
            del buf12
            del buf13
