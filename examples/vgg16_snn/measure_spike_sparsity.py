"""测量已训练 VGG16-SNN 的逐层脉冲稀疏度，并判定其是否「结构化」。

脉冲神经元（LIFNode）的前向输出是二值张量 {0,1}：1 = 发放脉冲，0 = 静默。
「稀疏度」= 0 的占比。它关系到「零值检测」优化——若某层输出大量为 0，下一层
（卷积 / 全连接）消费它时，理论上可跳过零输入对应的乘加。

但 GPU 上能否真正获益，取决于零值是否「结构化」：
  - 逐元素零值随机散布  → SIMT 下同一 warp 内零/非零混杂，无法整体跳过；
  - 整块零值（连续一段归约维全为 0）→ 才可能整块跳过、不引入分支发散。
因此本脚本同时测两件事：
  1. 逐层 / 整网 发放率与稀疏度；
  2. 沿通道（归约维）切成连续 32 元素一组，统计「整块全零」的比例，并与
     「零值独立同分布」时的理论值 (1-发放率)^32 对比——实测≈理论 ⇒ 非结构化、
     块级跳过无效；实测 >> 理论 ⇒ 结构化、块级跳过才可能有意义。

用法：python measure_spike_sparsity.py  [N_IMAGES]
"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vgg16_test import VGG16SNN, NUM_CLASSES, T
from spikingjelly.activation_based import neuron, functional
from torchvision import transforms, datasets

IMAGENET = "/home/charlley/Code/Dataset/imagenet"
CKPT_BEST = os.path.join(HERE, "vgg16_snn_imagenet.pth")
CKPT_LAST = os.path.join(HERE, "vgg16_snn_imagenet_latest.pth")
N_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 512
BLK = 32                     # 连续归约块大小（≈ warp 宽度 / 常见 BLOCK_K）
device = torch.device('cuda')

ckpt_path = CKPT_BEST if os.path.exists(CKPT_BEST) else CKPT_LAST
if not os.path.exists(ckpt_path):
    sys.exit("找不到已训练的 checkpoint，请先运行 finetune_snn.py")
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f"载入 checkpoint: {os.path.basename(ckpt_path)}  "
      f"(step={ck.get('global_step','?')}, val acc={ck.get('acc','?')})")

model = VGG16SNN(NUM_CLASSES)
model.load_state_dict(ck['state_dict'])
functional.set_step_mode(model, 'm')
model = model.to(device).eval()

# 收集全部 LIF 层；按在网络中出现的顺序编号
lifs = [m for m in model.modules() if isinstance(m, neuron.LIFNode)]
# v_threshold 是 LIFNode 的普通属性、不进 state_dict，须从 checkpoint 单独恢复
if 'thresholds' in ck:
    for lif, th in zip(lifs, ck['thresholds']):
        lif.v_threshold = th
    print("已恢复逐层校准阈值:", [f"{t:.2f}" for t in ck['thresholds']])
print(f"LIF 层数: {len(lifs)}\n")
stats = [{'spikes': 0, 'total': 0, 'blk_zero': 0, 'blk_total': 0} for _ in lifs]
handles = []
for i, lif in enumerate(lifs):
    def hk(mod, inp, out, i=i):
        s = stats[i]
        s['spikes'] += (out != 0).sum().item()
        s['total'] += out.numel()
        # 沿通道维切成 BLK 个一组的连续块，统计「整块全零」数
        C = out.shape[2]
        if C % BLK == 0:
            b = out.unflatten(2, (C // BLK, BLK))      # 把通道维拆成 (块, 块内)
            has = (b != 0).any(dim=3)                  # 该块内是否存在脉冲
            s['blk_zero'] += (~has).sum().item()
            s['blk_total'] += has.numel()
    handles.append(lif.register_forward_hook(hk))

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])
ds = datasets.ImageFolder(os.path.join(IMAGENET, "val"), tf)
loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=8)

print(f"在 {N_IMAGES} 张 val 图上统计脉冲发放 ...")
correct = total = 0
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xseq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        functional.reset_net(model)
        out = model(xseq).mean(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if total >= N_IMAGES:
            break
for h in handles:
    h.remove()

print(f"\n推理 top-1（{total} 张）: {100*correct/total:.2f}%\n")
print("逐层脉冲发放率 / 稀疏度：")
print(f"{'层':>6} | {'发放率':>10} | {'稀疏度':>10}")
print("-" * 36)
tot_s = tot_t = 0
for i, st in enumerate(stats):
    rate = st['spikes'] / st['total']
    tag = "分类器" if i >= 13 else "卷积块"
    print(f"LIF{i:2d} | {rate*100:9.2f}% | {(1-rate)*100:9.2f}%   ({tag})")
    tot_s += st['spikes']
    tot_t += st['total']
overall = tot_s / tot_t
print("-" * 36)
print(f"{'整网':>6} | {overall*100:9.2f}% | {(1-overall)*100:9.2f}%")

print(f"\n零值是否结构化（连续 {BLK} 通道为一归约块）：")
print(f"{'层':>6} | {'全零块·实测':>12} | {'全零块·i.i.d.理论':>16} | {'实测/理论':>10}")
print("-" * 56)
tz = tn = 0
for i, st in enumerate(stats):
    if st['blk_total'] == 0:
        continue
    meas = st['blk_zero'] / st['blk_total']
    rate = st['spikes'] / st['total']
    iid = (1 - rate) ** BLK
    ratio = meas / iid if iid > 0 else float('inf')
    print(f"LIF{i:2d} | {meas*100:11.3f}% | {iid*100:15.4f}% | {ratio:9.1f}x")
    tz += st['blk_zero']
    tn += st['blk_total']
meas_all = tz / tn
print("-" * 56)
print(f"{'整网':>6} | {meas_all*100:11.3f}% |")

print(f"\n整网平均脉冲发放率 {overall*100:.2f}%，约 {(1-overall)*100:.1f}% 的神经元-时间步静默。")
print(f"整网连续 {BLK} 通道「整块全零」比例 {meas_all*100:.3f}%——"
      f"这是块级零值检测真正能跳过的工作量上限。")
