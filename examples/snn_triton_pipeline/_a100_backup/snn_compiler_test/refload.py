import os, sys, torch
os.environ.setdefault("SJ_NEURON_BACKEND","torch")  # eager baseline reference
BASE="/home/liushifeng/charlley/snn_infer"
TBASE="/home/liushifeng/charlley/snn_infer_triton"
sys.path.insert(0, TBASE)
import sj_compat  # maps old API -> activation_based
sys.path.insert(0, os.path.join(BASE,"repos/Spike-Element-Wise-ResNet/imagenet"))
import sew_resnet
CK="/home/liushifeng/lsf/checkpoints/sew_resnet34_imagenet/sew34_checkpoint_319.pth"
sd=torch.load(CK,map_location="cpu",weights_only=False)["model"]
m=sew_resnet.sew_resnet34(T=4,connect_f="ADD",num_classes=1000,zero_init_residual=False)
mk=list(m.state_dict().keys())
if not any(".module." in k for k in mk):
    sd={k.replace(".module.","."):v for k,v in sd.items()}
miss,unexp=m.load_state_dict(sd,strict=False)
print("missing",len(miss),"unexpected",len(unexp))
m=m.cuda().eval()
x=torch.randn(2,3,224,224,device="cuda")
from spikingjelly.activation_based import functional as AF
with torch.no_grad():
    y=m(x)
print("ref out", tuple(y.shape), y.dtype, "argmax", y.argmax(1).tolist())
AF.reset_net(m)
print("REF_OK")
