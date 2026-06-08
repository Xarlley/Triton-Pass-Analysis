import sys, torch
sys.path.insert(0, "/home/liushifeng/charlley/snn_compiler_test")
import triton
print("torch", torch.__version__, "triton", triton.__version__, "cuda", torch.cuda.is_available())
from snn_compiler.kernels.neurons import if_lif, naive_if_lif
# small IF correctness vs naive reference
T,B,C,H,W = 4,2,8,4,4
x = torch.randn(T,B,C,H,W, device="cuda")
sp = if_lif(x.contiguous(), neuron="if", soft_reset=False, v_threshold=1.0, v_reset=0.0, layout="NCHW")
ref = naive_if_lif(x, neuron="if", soft_reset=False, v_threshold=1.0, v_reset=0.0, layout="NCHW")
print("IF kernel vs naive maxdiff:", (sp.float()-ref.float()).abs().max().item())
from snn_compiler.nn.modules import FusedConvBNNeuron
import torch.nn as nn
conv = nn.Conv2d(8,8,3,padding=1,bias=False).cuda().eval()
bn = nn.BatchNorm2d(8).cuda().eval(); bn.running_var.fill_(0.7); bn.running_mean.normal_(0,0.1)
m = FusedConvBNNeuron(conv, bn, neuron="if", v_threshold=1.0, v_reset=0.0, layout="NCHW").cuda()
y = m(x)
print("FusedConvBNNeuron out shape", tuple(y.shape), "dtype", y.dtype, "OK")
print("SMOKE_OK")
