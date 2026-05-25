"""测量已训练 VGG16-SNN 在 ImageNet 样本上的推理耗时。

加载 finetune_snn.py 训出的 T=4 VGG16-SNN（vgg16_snn_imagenet.pth），在 ImageNet
验证集上跑 N_SAMPLES 张，分别报告：
  - 纯 GPU 前向计算耗时（CUDA 同步前后计时，已预热）；
  - 含 JPEG 解码 / 数据加载的总墙钟耗时；
  - 吞吐与 top-1（健全性检查）。

两种模式：
  - 默认：eager fp32 推理。
  - COMPILE=1：走 torch.compile 的全 Triton 路径（复用 vgg16_test.py 的
    configure_full_triton_compilation——max_autotune、conv/gemm 后端=TRITON）。
    首次调用会触发编译 + 自动调优（数分钟），单独计时、不计入推理耗时。

用法：python benchmark_inference.py [N_SAMPLES=10000] [BATCH=50]
      COMPILE=1 python benchmark_inference.py ...
"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vgg16_test import (VGG16SNN, NUM_CLASSES, T,
                        configure_full_triton_compilation,
                        patch_spikingjelly_for_full_graph)
from spikingjelly.activation_based import neuron, functional
from torchvision import transforms, datasets

IMAGENET = "/home/charlley/Code/Dataset/imagenet"
CKPT = os.path.join(HERE, "vgg16_snn_imagenet.pth")
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 50
COMPILE = bool(os.environ.get("COMPILE"))
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

if COMPILE:
    print("[模式] torch.compile 全 Triton 路径", flush=True)
    configure_full_triton_compilation()
    patch_spikingjelly_for_full_graph()
else:
    print("[模式] eager fp32", flush=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = VGG16SNN(NUM_CLASSES)
model.load_state_dict(ck["state_dict"])
lifs = [m for m in model.modules() if isinstance(m, neuron.LIFNode)]
if "thresholds" in ck:                       # v_threshold 不进 state_dict，单独恢复
    for lif, th in zip(lifs, ck["thresholds"]):
        lif.v_threshold = th
functional.set_step_mode(model, "m")
model = model.to(device).eval()
infer_model = torch.compile(model) if COMPILE else model
print(f"模型 vgg16_snn_imagenet.pth (val acc={ck.get('acc','?')}) | T={T} | BATCH={BATCH}",
      flush=True)

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])
ds = datasets.ImageFolder(os.path.join(IMAGENET, "val"), tf)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False,
                                     num_workers=8, pin_memory=True)


@torch.no_grad()
def infer(x):
    xseq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)   # [T,N,C,H,W]
    functional.reset_net(model)                   # 复位原始 model（与 compiled 共享模块）
    return infer_model(xseq).mean(0)              # [N,1000]


it = iter(loader)
if COMPILE:
    print("torch.compile 首次调用：编译 + 自动调优（可能数分钟）...", flush=True)
    tc0 = time.time()
    x, _ = next(it)
    infer(x.to(device, non_blocking=True))
    torch.cuda.synchronize()
    print(f"编译 + 首次前向耗时 : {time.time() - tc0:.1f} s", flush=True)

print("预热 5 个 batch ...", flush=True)
for _ in range(5):
    x, _ = next(it)
    infer(x.to(device, non_blocking=True))
torch.cuda.synchronize()
del it

print(f"开始计时：推理 {N_SAMPLES} 张 ImageNet val 样本 ...", flush=True)
compute_s = 0.0
correct = total = 0
wall0 = time.time()
for x, y in loader:
    x = x.to(device, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.time()
    out = infer(x)
    torch.cuda.synchronize()
    compute_s += time.time() - t0
    correct += (out.argmax(1).cpu() == y).sum().item()
    total += y.size(0)
    if total >= N_SAMPLES:
        break
wall_s = time.time() - wall0

print(f"\n模式                : {'torch.compile 全 Triton' if COMPILE else 'eager fp32'}")
print(f"样本数              : {total}")
print(f"纯 GPU 前向计算耗时 : {compute_s:.2f} s   ({total / compute_s:.0f} 张/秒)")
print(f"总墙钟耗时          : {wall_s:.2f} s   (含 JPEG 解码与数据加载, "
      f"{total / wall_s:.0f} 张/秒)")
print(f"单张推理            : {compute_s / total * 1000:.2f} ms (纯计算)")
print(f"top-1（健全性检查）  : {100 * correct / total:.2f}%")
