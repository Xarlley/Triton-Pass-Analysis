# ===== EXCERPT of inductor fx_graph_readable.py (full 423 lines on a100) =====
# Note: ScaledWSConv2d weight-standardization (aten.mean/var.correction/*gain) is traced INTO the graph,
#       then fused into the Triton conv kernel. The LIF neuron appears as the opaque multistep_lif op.

class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 3, 32, 32]", arg1_1: "f32[64, 3, 3, 3]", arg2_1: "f32[64, 1, 1, 1]", arg3_1: "f32[64]", arg4_1: "f32[128, 64, 3, 3]", arg5_1: "f32[128, 1, 1, 1]", arg6_1: "f32[128]", arg7_1: "f32[256, 128, 3, 3]", arg8_1: "f32[256, 1, 1, 1]", arg9_1: "f32[256]", arg10_1: "f32[256, 256, 3, 3]", arg11_1: "f32[256, 1, 1, 1]", arg12_1: "f32[256]", arg13_1: "f32[512, 256, 3, 3]", arg14_1: "f32[512, 1, 1, 1]", arg15_1: "f32[512]", arg16_1: "f32[512, 512, 3, 3]", arg17_1: "f32[512, 1, 1, 1]", arg18_1: "f32[512]", arg19_1: "f32[512, 512, 3, 3]", arg20_1: "f32[512, 1, 1, 1]", arg21_1: "f32[512]", arg22_1: "f32[512, 512, 3, 3]", arg23_1: "f32[512, 1, 1, 1]", arg24_1: "f32[512]", arg25_1: "f32[10, 512]", arg26_1: "f32[10]"):
        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:73 in forward, code: x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)   # [B,C,H,W] -> [T,B,C,H,W] (direct encoding)
        unsqueeze: "f32[1, 8, 3, 32, 32]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        repeat: "f32[6, 8, 3, 32, 32]" = torch.ops.aten.repeat.default(unsqueeze, [6, 1, 1, 1, 1]);  unsqueeze = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/functional/forward.py:288 in seq_to_ann_forward, code: y = x_seq.flatten(0, 1)
        view: "f32[48, 3, 32, 32]" = torch.ops.aten.view.default(repeat, [48, 3, 32, 32]);  repeat = None

        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:26 in get_weight, code: fan_in = np.prod(self.weight.shape[1:])
        _tensor_constant0: "i64[3]" = self._tensor_constant0
        lift_fresh_copy: "i64[3]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        prod: "i64[]" = torch.ops.aten.prod.default(lift_fresh_copy);  lift_fresh_copy = None

        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:27 in get_weight, code: mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        mean: "f32[64, 1, 1, 1]" = torch.ops.aten.mean.dim(arg1_1, [1, 2, 3], True)

        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:28 in get_weight, code: var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        var: "f32[64, 1, 1, 1]" = torch.ops.aten.var.correction(arg1_1, [1, 2, 3], correction = 1, keepdim = True)

        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:29 in get_weight, code: weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        sub: "f32[64, 3, 3, 3]" = torch.ops.aten.sub.Tensor(arg1_1, mean);  arg1_1 = mean = None
        mul: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(var, prod);  var = prod = None
        add: "f32[64, 1, 1, 1]" = torch.ops.aten.add.Tensor(mul, 0.0001);  mul = None
        pow_1: "f32[64, 1, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(add, 0.5);  add = None
        div: "f32[64, 3, 3, 3]" = torch.ops.aten.div.Tensor(sub, pow_1);  sub = pow_1 = None

        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:31 in get_weight, code: weight = weight * self.gain
        mul_1: "f32[64, 3, 3, 3]" = torch.ops.aten.mul.Tensor(div, arg2_1);  div = arg2_1 = None

        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:35 in forward, code: return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
        convert_element_type: "bf16[64]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[64, 3, 3, 3]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        convert_element_type_2: "bf16[48, 3, 32, 32]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convolution: "bf16[48, 64, 32, 32]" = torch.ops.aten.convolution.default(convert_element_type_2, convert_element_type_1, convert_element_type, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_2 = convert_element_type_1 = convert_element_type = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/functional/forward.py:295 in seq_to_ann_forward, code: return y.view(y_shape)
        view_1: "bf16[6, 8, 64, 32, 32]" = torch.ops.aten.view.default(convolution, [6, 8, 64, 32, 32]);  convolution = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/neuron/base_node.py:377 in v_float_to_tensor, code: self.v = torch.full_like(x, v_init, requires_grad=False)
        full_default: "bf16[8, 64, 32, 32]" = torch.ops.aten.full.default([8, 64, 32, 32], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/triton_kernel/neuron_kernel/lif.py:497 in multistep_lif, code: s_seq, v_seq = multistep_lif_inference(
        empty: "bf16[6, 8, 64, 32, 32]" = torch.ops.aten.empty.memory_format([6, 8, 64, 32, 32], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        empty_1: "bf16[6, 8, 64, 32, 32]" = torch.ops.aten.empty.memory_format([6, 8, 64, 32, 32], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 8, grid = [(4096, 1, 1), (2048, 1, 1), (2048, 1, 1), (1024, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'x_seq_ptr': view_1, 'v_init_ptr': full_default, 's_seq_ptr': empty, 'h_seq_ptr': empty_1, 'v_seq_ptr': empty_1, 'tau': 2.0, 'v_threshold': 1.0, 'v_reset': 0.0, 'T': 6, 'NCL': 524288, 'decay_input': False, 'soft_reset': True, 'save_intermediates': False}, tensors_to_clone = ['s_seq_ptr', 'v_seq_ptr']);  view_1 = full_default = empty = empty_1 = None
        getitem: "bf16[6, 8, 64, 32, 32]" = triton_kernel_wrapper_functional_proxy['s_seq_ptr']
        getitem_1: "bf16[6, 8, 64, 32, 32]" = triton_kernel_wrapper_functional_proxy['v_seq_ptr'];  triton_kernel_wrapper_functional_proxy = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/neuron/lif.py:647 in multi_step_forward, code: self.v = v_seq[-1].clone()
        select_3: "bf16[8, 64, 32, 32]" = torch.ops.aten.select.int(getitem_1, 0, -1);  getitem_1 = None

        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:40 in forward, code: def forward(self, x): return x * self.scale
        mul_2: "bf16[6, 8, 64, 32, 32]" = torch.ops.aten.mul.Tensor(getitem, 2.74);  getitem = None

        # File: /home/liushifeng/charlley/spikingjelly/spikingjelly/activation_based/functional/forward.py:288 in seq_to_ann_forward, code: y = x_seq.flatten(0, 1)
        view_2: "bf16[48, 64, 32, 32]" = torch.ops.aten.view.default(mul_2, [48, 64, 32, 32]);  mul_2 = None

        # File: /home/liushifeng/charlley/snn_infer_triton/ottt_vgg_triton.py:26 in get_weight, code: fan_in = np.prod(self.weight.shape[1:])
        _tensor_constant1: "i64[3]" = self._tensor_constant1
