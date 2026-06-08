# ===== EXCERPT of inductor fx_graph_readable.py (full 1590 lines on a100) =====
# Shows the ATen FX graph: conv -> bn -> the spikingjelly neuron as an opaque custom op

class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "f32[8, 3, 224, 224]", arg2_1: "f32[64]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64, 64, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64, 64, 3, 3]", arg12_1: "f32[64]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[64, 64, 3, 3]", arg17_1: "f32[64]", arg18_1: "f32[64]", arg19_1: "f32[64]", arg20_1: "f32[64]", arg21_1: "f32[64, 64, 3, 3]", arg22_1: "f32[64]", arg23_1: "f32[64]", arg24_1: "f32[64]", arg25_1: "f32[64]", arg26_1: "f32[64, 64, 3, 3]", arg27_1: "f32[64]", arg28_1: "f32[64]", arg29_1: "f32[64]", arg30_1: "f32[64]", arg31_1: "f32[64, 64, 3, 3]", arg32_1: "f32[64]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[128, 64, 3, 3]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[128]", arg40_1: "f32[128]", arg41_1: "f32[128, 128, 3, 3]", arg42_1: "f32[128]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[128]", arg46_1: "f32[128, 64, 1, 1]", arg47_1: "f32[128]", arg48_1: "f32[128]", arg49_1: "f32[128]", arg50_1: "f32[128]", arg51_1: "f32[128, 128, 3, 3]", arg52_1: "f32[128]", arg53_1: "f32[128]", arg54_1: "f32[128]", arg55_1: "f32[128]", arg56_1: "f32[128, 128, 3, 3]", arg57_1: "f32[128]", arg58_1: "f32[128]", arg59_1: "f32[128]", arg60_1: "f32[128]", arg61_1: "f32[128, 128, 3, 3]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[128]", arg66_1: "f32[128, 128, 3, 3]", arg67_1: "f32[128]", arg68_1: "f32[128]", arg69_1: "f32[128]", arg70_1: "f32[128]", arg71_1: "f32[128, 128, 3, 3]", arg72_1: "f32[128]", arg73_1: "f32[128]", arg74_1: "f32[128]", arg75_1: "f32[128]", arg76_1: "f32[128, 128, 3, 3]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[128]", arg80_1: "f32[128]", arg81_1: "f32[256, 128, 3, 3]", arg82_1: "f32[256]", arg83_1: "f32[256]", arg84_1: "f32[256]", arg85_1: "f32[256]", arg86_1: "f32[256, 256, 3, 3]", arg87_1: "f32[256]", arg88_1: "f32[256]", arg89_1: "f32[256]", arg90_1: "f32[256]", arg91_1: "f32[256, 128, 1, 1]", arg92_1: "f32[256]", arg93_1: "f32[256]", arg94_1: "f32[256]", arg95_1: "f32[256]", arg96_1: "f32[256, 256, 3, 3]", arg97_1: "f32[256]", arg98_1: "f32[256]", arg99_1: "f32[256]", arg100_1: "f32[256]", arg101_1: "f32[256, 256, 3, 3]", arg102_1: "f32[256]", arg103_1: "f32[256]", arg104_1: "f32[256]", arg105_1: "f32[256]", arg106_1: "f32[256, 256, 3, 3]", arg107_1: "f32[256]", arg108_1: "f32[256]", arg109_1: "f32[256]", arg110_1: "f32[256]", arg111_1: "f32[256, 256, 3, 3]", arg112_1: "f32[256]", arg113_1: "f32[256]", arg114_1: "f32[256]", arg115_1: "f32[256]", arg116_1: "f32[256, 256, 3, 3]", arg117_1: "f32[256]", arg118_1: "f32[256]", arg119_1: "f32[256]", arg120_1: "f32[256]", arg121_1: "f32[256, 256, 3, 3]", arg122_1: "f32[256]", arg123_1: "f32[256]", arg124_1: "f32[256]", arg125_1: "f32[256]", arg126_1: "f32[256, 256, 3, 3]", arg127_1: "f32[256]", arg128_1: "f32[256]", arg129_1: "f32[256]", arg130_1: "f32[256]", arg131_1: "f32[256, 256, 3, 3]", arg132_1: "f32[256]", arg133_1: "f32[256]", arg134_1: "f32[256]", arg135_1: "f32[256]", arg136_1: "f32[256, 256, 3, 3]", arg137_1: "f32[256]", arg138_1: "f32[256]", arg139_1: "f32[256]", arg140_1: "f32[256]", arg141_1: "f32[256, 256, 3, 3]", arg142_1: "f32[256]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[256]", arg146_1: "f32[512, 256, 3, 3]", arg147_1: "f32[512]", arg148_1: "f32[512]", arg149_1: "f32[512]", arg150_1: "f32[512]", arg151_1: "f32[512, 512, 3, 3]", arg152_1: "f32[512]", arg153_1: "f32[512]", arg154_1: "f32[512]", arg155_1: "f32[512]", arg156_1: "f32[512, 256, 1, 1]", arg157_1: "f32[512]", arg158_1: "f32[512]", arg159_1: "f32[512]", arg160_1: "f32[512]", arg161_1: "f32[512, 512, 3, 3]", arg162_1: "f32[512]", arg163_1: "f32[512]", arg164_1: "f32[512]", arg165_1: "f32[512]", arg166_1: "f32[512, 512, 3, 3]", arg167_1: "f32[512]", arg168_1: "f32[512]", arg169_1: "f32[512]", arg170_1: "f32[512]", arg171_1: "f32[512, 512, 3, 3]", arg172_1: "f32[512]", arg173_1: "f32[512]", arg174_1: "f32[512]", arg175_1: "f32[512]", arg176_1: "f32[512, 512, 3, 3]", arg177_1: "f32[512]", arg178_1: "f32[512]", arg179_1: "f32[512]", arg180_1: "f32[512]", arg181_1: "f32[1000, 512]", arg182_1: "f32[1000]"):
        # File: /home/liushifeng/charlley/snn_infer/repos/Spike-Element-Wise-ResNet/imagenet/sew_resnet.py:210 in _forward_impl, code: x = self.conv1(x)
        convert_element_type: "bf16[64, 3, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[8, 3, 224, 224]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convolution: "bf16[8, 64, 112, 112]" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  convert_element_type_1 = convert_element_type = None

        # File: /home/liushifeng/charlley/snn_infer/repos/Spike-Element-Wise-ResNet/imagenet/sew_resnet.py:211 in _forward_impl, code: x = self.bn1(x)
        add: "f32[64]" = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        sub: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        mul_1: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_4: "bf16[8, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None

        # File: /home/liushifeng/charlley/snn_infer/repos/Spike-Element-Wise-ResNet/imagenet/sew_resnet.py:212 in _forward_impl, code: x.unsqueeze_(0)
        unsqueeze_8: "bf16[1, 8, 64, 112, 112]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 0);  convert_element_type_4 = None

        # File: /home/liushifeng/charlley/snn_infer/repos/Spike-Element-Wise-ResNet/imagenet/sew_resnet.py:213 in _forward_impl, code: x = x.repeat(self.T, 1, 1, 1, 1)
        repeat: "bf16[4, 8, 64, 112, 112]" = torch.ops.aten.repeat.default(unsqueeze_8, [4, 1, 1, 1, 1]);  unsqueeze_8 = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/neuron/base_node.py:377 in v_float_to_tensor, code: self.v = torch.full_like(x, v_init, requires_grad=False)
        full_default: "bf16[8, 64, 112, 112]" = torch.ops.aten.full.default([8, 64, 112, 112], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/integrate_and_fire.py:453 in multistep_if, code: s_seq, v_seq = multistep_if_inference(
        empty_1: "bf16[4, 8, 64, 112, 112]" = torch.ops.aten.empty.memory_format([4, 8, 64, 112, 112], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_2: "bf16[4, 8, 64, 112, 112]" = torch.ops.aten.empty.memory_format([4, 8, 64, 112, 112], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 36, grid = [(50176, 1, 1), (25088, 1, 1), (25088, 1, 1), (12544, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'x_seq_ptr': repeat, 'v_init_ptr': full_default, 's_seq_ptr': empty_1, 'h_seq_ptr': empty_2, 'v_seq_ptr': empty_2, 'v_threshold': 1.0, 'v_reset': 0.0, 'T': 4, 'NCL': 6422528, 'soft_reset': False, 'save_intermediates': False}, tensors_to_clone = ['s_seq_ptr', 'v_seq_ptr']);  repeat = full_default = empty_1 = empty_2 = None
        getitem: "bf16[4, 8, 64, 112, 112]" = triton_kernel_wrapper_functional_proxy['s_seq_ptr']
        getitem_1: "bf16[4, 8, 64, 112, 112]" = triton_kernel_wrapper_functional_proxy['v_seq_ptr'];  triton_kernel_wrapper_functional_proxy = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/neuron/integrate_and_fire.py:467 in multi_step_forward, code: self.v = v_seq[-1].clone()
        select_3: "bf16[8, 64, 112, 112]" = torch.ops.aten.select.int(getitem_1, 0, -1);  getitem_1 = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/functional/forward.py:288 in seq_to_ann_forward, code: y = x_seq.flatten(0, 1)
        view_1: "bf16[32, 64, 112, 112]" = torch.ops.aten.view.default(getitem, [32, 64, 112, 112]);  getitem = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/functional/forward.py:293 in seq_to_ann_forward, code: y = stateless_module(y)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(view_1, [3, 3], [2, 2], [1, 1], [1, 1], False);  view_1 = None
        getitem_2: "bf16[32, 64, 56, 56]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/functional/forward.py:295 in seq_to_ann_forward, code: return y.view(y_shape)
        view_2: "bf16[4, 8, 64, 56, 56]" = torch.ops.aten.view.default(getitem_2, [4, 8, 64, 56, 56]);  getitem_2 = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/functional/forward.py:288 in seq_to_ann_forward, code: y = x_seq.flatten(0, 1)
        view_3: "bf16[32, 64, 56, 56]" = torch.ops.aten.view.default(view_2, [32, 64, 56, 56])

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/functional/forward.py:293 in seq_to_ann_forward, code: y = stateless_module(y)
        convert_element_type_5: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
