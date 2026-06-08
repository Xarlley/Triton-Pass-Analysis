import os, sys, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("SJ_NEURON_BACKEND","torch")
BASE="/home/liushifeng/charlley/snn_infer"; TBASE="/home/liushifeng/charlley/snn_infer_triton"
SNNC="/home/liushifeng/charlley/snn_compiler_test"
sys.path.insert(0,SNNC); sys.path.insert(0,TBASE); import sj_compat
sys.path.insert(0, os.path.join(BASE,"repos/Spike-Element-Wise-ResNet/imagenet")); import sew_resnet
from spikingjelly.activation_based import functional as AF
from snn_compiler.kernels.neurons import if_lif
from snn_compiler.kernels.fused import fold_conv_bn

CK="/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth"
sd=torch.load(CK,map_location="cpu",weights_only=False)["model"]
ref=sew_resnet.sew_resnet34(T=4,connect_f="ADD",num_classes=1000)
if not any(".module." in k for k in ref.state_dict().keys()):
    sd={k.replace(".module.","."):v for k,v in sd.items()}
ref.load_state_dict(sd,strict=False); ref=ref.eval().cuda()

cap={}
ref.sn1.register_forward_hook(lambda m,i,o: cap.__setitem__("sn1_out",o.detach()))
ref.sn1.register_forward_pre_hook(lambda m,i: cap.__setitem__("sn1_in",i[0].detach()))
torch.manual_seed(0); x=torch.randn(4,3,224,224,device="cuda")
AF.reset_net(ref)
with torch.no_grad(): ref(x)

pre=cap["sn1_in"]                     # exact pre-activation ref fed to IF (= repeated bn1(conv1(x)))
# (A) feed ref's OWN pre-activation into if_lif:
s_kr=if_lif(pre.contiguous(),neuron="if",soft_reset=False,v_threshold=1.0,v_reset=0.0,layout="NCHW")
print("(A) if_lif on REF pre-activation vs ref.sn1: max|Δ|=",(s_kr.float()-cap["sn1_out"].float()).abs().max().item())

# (B) my folded pre-activation:
w,b=fold_conv_bn(ref.conv1.weight.detach(),None,ref.bn1.weight.detach(),ref.bn1.bias.detach(),
                 ref.bn1.running_mean.detach(),ref.bn1.running_var.detach(),ref.bn1.eps)
y=F.conv2d(x,w,bias=b,stride=ref.conv1.stride,padding=ref.conv1.padding)
pre_fold=y.unsqueeze(0).repeat(4,1,1,1,1)
print("(B) folded pre-act vs ref pre-act: max|Δ|=",(pre_fold-pre).abs().max().item(),
      " mean|Δ|=",(pre_fold-pre).abs().mean().item())
near=( (pre-1.0).abs()<1e-4 ).sum().item()
print("    #elements within 1e-4 of threshold:",near," / ",pre.numel())

# (C) separate conv+BN (eager, no fold) pre-activation vs ref:
y2=ref.bn1(F.conv2d(x,ref.conv1.weight,stride=ref.conv1.stride,padding=ref.conv1.padding))
pre_sep=y2.unsqueeze(0).repeat(4,1,1,1,1)
print("(C) separate conv+BN pre-act vs ref pre-act: max|Δ|=",(pre_sep-pre).abs().max().item())
s_sep=if_lif(pre_sep.contiguous(),neuron="if",soft_reset=False,v_threshold=1.0,v_reset=0.0,layout="NCHW")
print("    if_lif on separate-conv+BN vs ref.sn1: max|Δ|=",(s_sep.float()-cap["sn1_out"].float()).abs().max().item())
