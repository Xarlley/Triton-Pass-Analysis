import os, sys, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("SJ_NEURON_BACKEND","torch")
BASE="/home/liushifeng/charlley/snn_infer"; TBASE="/home/liushifeng/charlley/snn_infer_triton"
SNNC="/home/liushifeng/charlley/snn_compiler_test"
sys.path.insert(0,SNNC); sys.path.insert(0,TBASE); import sj_compat
sys.path.insert(0, os.path.join(BASE,"repos/Spike-Element-Wise-ResNet/imagenet")); import sew_resnet
from spikingjelly.activation_based import functional as AF
import sew_exp  # reuse SEWResNetFused / FusedBlock / StemFused

CK="/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth"
sd=torch.load(CK,map_location="cpu",weights_only=False)["model"]
ref=sew_resnet.sew_resnet34(T=4,connect_f="ADD",num_classes=1000)
if not any(".module." in k for k in ref.state_dict().keys()):
    sd={k.replace(".module.","."):v for k,v in sd.items()}
ref.load_state_dict(sd,strict=False); ref=ref.eval().cuda()

caps={}
def hook(name):
    def f(mod,inp,out): caps[name]=out.detach()
    return f
ref.sn1.register_forward_hook(hook("stem_sn"))
ref.maxpool.register_forward_hook(hook("maxpool"))
ref.layer1.register_forward_hook(hook("layer1"))
ref.layer2.register_forward_hook(hook("layer2"))
ref.layer3.register_forward_hook(hook("layer3"))
ref.layer4.register_forward_hook(hook("layer4"))

torch.manual_seed(0)
x=torch.randn(4,3,224,224,device="cuda")
AF.reset_net(ref)
with torch.no_grad(): yref=ref(x).float()

fused=sew_exp.SEWResNetFused(ref,T=4,layout="NCHW").cuda().eval()
# manual stepwise forward capturing same points
with torch.no_grad():
    s=fused.stem(x)
    print("stem  max|Δ|=",(s-caps["stem_sn"]).abs().max().item(), "fused-fire",s.float().mean().item(),"ref-fire",caps["stem_sn"].float().mean().item())
    T,B=s.shape[0],s.shape[1]
    x4=s.reshape(T*B,*s.shape[2:]); x4=fused.maxpool(x4); s=x4.view(T,B,*x4.shape[1:])
    print("mpool max|Δ|=",(s-caps["maxpool"]).abs().max().item())
    names=["layer1","layer2","layer3","layer4"]
    for nm,stage in zip(names,fused.stages):
        for blk in stage: s=blk(s)
        print(f"{nm} max|Δ|=",(s-caps[nm]).abs().max().item(),"fused-fire",s.float().mean().item(),"ref-fire",caps[nm].float().mean().item())
