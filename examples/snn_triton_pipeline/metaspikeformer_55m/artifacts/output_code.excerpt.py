# ===== EXCERPT of inductor output_code.py (full 17046 lines on a100) =====

## --- (A) spikingjelly LIF Triton kernel (def line 453) ---
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}],
    inductor_meta={'grid_type': 'PrecomputedGrid', 'precomputed_grids': [{'config': {'BLOCK_NCL': 128, 'num_warps': 4, 'num_stages': 3}, 'python': ['50176', '1', '1'], 'cpp': ['50176L', '1L', '1L'], 'python_slow': ['50176', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 8, 'num_stages': 3}, 'python': ['25088', '1', '1'], 'cpp': ['25088L', '1L', '1L'], 'python_slow': ['25088', '1', '1']}, {'config': {'BLOCK_NCL': 256, 'num_warps': 4, 'num_stages': 3}, 'python': ['25088', '1', '1'], 'cpp': ['25088L', '1L', '1L'], 'python_slow': ['25088', '1', '1']}, {'config': {'BLOCK_NCL': 512, 'num_warps': 8, 'num_stages': 3}, 'python': ['12544', '1', '1'], 'cpp': ['12544L', '1L', '1L'], 'python_slow': ['12544', '1', '1']}], 'extra_launcher_args': [], 'declared_constexpr_names': ['T', 'NCL', 'BLOCK_NCL', 'dtype', 'decay_input', 'soft_reset', 'save_intermediates'], 'kernel_name': '_multistep_lif_forward_kernel_0', 'backend_hash': '6F0525D885E9BF9AAD18AC5366ACD705CC63C622283A31CDFD8E26C2B9DB046A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    triton_meta={'signature': {'x_seq_ptr': '*bf16', 'v_init_ptr': '*bf16', 's_seq_ptr': '*bf16', 'h_seq_ptr': '*bf16', 'v_seq_ptr': '*bf16', 'tau': 'fp64', 'v_threshold': 'fp64', 'v_reset': 'fp64', 'T': 'constexpr', 'NCL': 'constexpr', 'BLOCK_NCL': 'constexpr', 'dtype': 'constexpr', 'decay_input': 'constexpr', 'soft_reset': 'constexpr', 'save_intermediates': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'T': 4, 'NCL': 6422528, 'dtype': triton.language.bfloat16, 'decay_input': True, 'soft_reset': False, 'save_intermediates': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'restore_value': ('s_seq_ptr', 'h_seq_ptr', 'v_seq_ptr')},
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

## --- (B) THE residual: depthwise conv via ATEN extern_kernels (groups>1, no Triton template) ---
            triton_poi_fused__to_copy_convolution_view_11.run(buf12, buf19, 4096, 12544, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:15
            buf20 = extern_kernels.convolution(buf19, buf18, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
            assert_size_stride(buf20, (32, 128, 112, 112), (1605632, 12544, 112, 1), 'unknown_op')
            del buf18

## --- (C) call() head: Triton kernels + interleaved extern_kernels.convolution(groups=...) ---
    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1 = args
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
            assert_size_stride(arg1_1, (64, 3, 7, 7), (147, 49, 7, 1))
            buf1 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_1:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg1_1, buf1, 192, 49, stream=raw_stream0)
            del arg1_1
            buf2 = empty_strided_cuda((32, 64, 112, 112), (802816, 12544, 112, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [unsqueeze, x, flatten, x_1], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy, aten.convolution]
            # [Provenance debug handles] triton_tem_fused__to_copy_convolution_repeat_unsqueeze_view_2:3
            raw_stream0 = get_raw_stream(0)
            triton_tem_fused__to_copy_convolution_repeat_unsqueeze_view_2.run(buf0, buf1, buf2, 6272, 1, 1, stream=raw_stream0)
            del buf0
            del buf1
            assert_size_stride(arg2_1, (64, ), (1, ))
            assert_size_stride(arg3_1, (64, ), (1, ))
            assert_size_stride(arg4_1, (64, ), (1, ))
            assert_size_stride(arg5_1, (64, ), (1, ))
            assert_size_stride(arg6_1, (64, ), (1, ))
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [unsqueeze, x, flatten, x_1, batch_norm], Original ATen: [aten.unsqueeze, aten.repeat, aten.view, aten._to_copy, aten.convolution, aten._native_batch_norm_legit_no_training]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_repeat_unsqueeze_view_3:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_repeat_unsqueeze_view_3.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, 25690112, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
