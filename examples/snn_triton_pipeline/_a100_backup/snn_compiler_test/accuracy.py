import os, sys, time, argparse, torch
os.environ.setdefault("SJ_NEURON_BACKEND","torch")
BASE="/home/liushifeng/charlley/snn_infer"; TBASE="/home/liushifeng/charlley/snn_infer_triton"
SNNC="/home/liushifeng/charlley/snn_compiler_test"
sys.path.insert(0,SNNC); sys.path.insert(0,TBASE); import sj_compat
import sew_exp
from snn_eval_lib_triton import ValSet, make_transform
from spikingjelly.activation_based import functional as AF
from torch.utils.data import DataLoader

ap=argparse.ArgumentParser(); ap.add_argument("--num",type=int,default=2000); ap.add_argument("--bs",type=int,default=50)
a=ap.parse_args()
ref=sew_exp.load_reference().cuda()
exact=sew_exp.SEWExact(ref).cuda().eval()
fold=sew_exp.SEWFold(ref,layout="NHWC").cuda().eval().to(torch.bfloat16)

ds=ValSet(a.num, make_transform("r256_bilinear"))
dl=DataLoader(ds,batch_size=a.bs,shuffle=False,num_workers=8,pin_memory=True)
print(f"evaluating {len(ds)} images")

stats={k:[0,0] for k in ["ref","exact","fold"]}  # top1,top5
agree={"exact":0,"fold":0}; tot=0
@torch.no_grad()
def topk(out,y):
    _,p=out.topk(5,1,True,True); p=p.t(); c=p.eq(y.view(1,-1))
    return c[:1].reshape(-1).float().sum().item(), c[:5].reshape(-1).float().sum().item(), p[0]
st=time.time()
with torch.no_grad():
    for x,y in dl:
        x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
        AF.reset_net(ref); o_ref=ref(x).float()
        o_ex=exact(x).float()
        xb=x.to(torch.bfloat16).contiguous(memory_format=torch.channels_last); o_fo=fold(xb).float()
        for nm,o in [("ref",o_ref),("exact",o_ex),("fold",o_fo)]:
            t1,t5,pred=topk(o,y); stats[nm][0]+=t1; stats[nm][1]+=t5
        agree["exact"]+=(o_ref.argmax(1)==o_ex.argmax(1)).sum().item()
        agree["fold"]+=(o_ref.argmax(1)==o_fo.argmax(1)).sum().item()
        tot+=y.numel()
dt=time.time()-st
print(f"N={tot}  ({dt:.1f}s)")
for nm in ["ref","exact","fold"]:
    print(f"  {nm:6}: top1={stats[nm][0]/tot*100:.2f}%  top5={stats[nm][1]/tot*100:.2f}%")
print(f"  argmax-agree vs ref:  EXACT={agree['exact']/tot*100:.2f}%   FOLD-bf16={agree['fold']/tot*100:.2f}%")
