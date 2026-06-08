import os, sys, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("SJ_NEURON_BACKEND","torch")
BASE="/home/liushifeng/charlley/snn_infer"; TBASE="/home/liushifeng/charlley/snn_infer_triton"
SNNC="/home/liushifeng/charlley/snn_compiler_test"
sys.path.insert(0,SNNC); sys.path.insert(0,TBASE); import sj_compat
sys.path.insert(0, os.path.join(BASE,"repos/Spike-Element-Wise-ResNet/imagenet")); import sew_resnet
from spikingjelly.activation_based import functional as AF
from snn_compiler.kernels.neurons import if_lif

CK="/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth"
sd=torch.load(CK,map_location="cpu",weights_only=False)["model"]
m=sew_resnet.sew_resnet34(T=4,connect_f="ADD",num_classes=1000)
if not any(".module." in k for k in m.state_dict().keys()):
    sd={k.replace(".module.","."):v for k,v in sd.items()}
m.load_state_dict(sd,strict=False); m=m.eval().cuda()

# ---- 1. direct neuron equivalence: feed same pre-activation to SJ IFNode vs if_lif ----
torch.manual_seed(0)
T,B,C,H,W=4,2,4,3,3
pre=torch.randn(T,B,C,H,W,device="cuda")*1.5
sn=m.layer1[0].sn1.__class__()  # MultiStepIFNode via sj_compat
sn=sn.cuda(); AF.reset_net(sn)
with torch.no_grad():
    s_sj=sn(pre.clone())
s_kr=if_lif(pre.contiguous(),neuron="if",soft_reset=False,v_threshold=1.0,v_reset=0.0,layout="NCHW")
print("neuron eq  max|Δ|=",(s_sj.float()-s_kr.float()).abs().max().item(),
      "  sj fired frac",s_sj.float().mean().item(),"  kr",s_kr.float().mean().item())
print("sn repr:", sn)

# inspect a block structure
blk=m.layer1[0]
print("block.conv1 type:",type(blk.conv1).__name__, "children:",[type(c).__name__ for c in blk.conv1.modules()])
print("block.sn1:",blk.sn1)
print("downsample of layer2[0]:", type(m.layer2[0].downsample).__name__,
      None if m.layer2[0].downsample is None else [type(c).__name__ for c in m.layer2[0].downsample.modules()])
