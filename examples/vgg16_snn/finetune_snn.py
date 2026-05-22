"""在 ImageNet 上微调出 T=4 的 VGG16-SNN（步数制 + 断点续训）。

以 torchvision ImageNet 预训练 VGG16-BN 的 卷积/BN/全连接 权重为初始化，用替代梯度
（LIFNode 自带 ATan surrogate）直接做 BPTT 训练，得到一个真正的 T=4 脉冲网络。

按「训练步数」组织，而非「epoch」：每隔 CKPT_EVERY 步在 val 子集上评估并存档，
存档含 优化器/调度器/scaler 状态，崩溃后可从最近存档无缝续训（最多损失 CKPT_EVERY
步）。这对需要跑数天的无人值守任务很重要。

从 ANN 权重微调成 SNN，本质是让卷积特征适配 LIF 动力学。学习率过高会在前期把
预训练特征冲垮（实测 AdamW lr=1e-3 / 2e-4 时 loss 上升、发散），故采用「低学习率
+ 线性 warmup + 梯度裁剪」：前 WARMUP 步学习率线性升温，让 AdamW 的动量估计在
早期大梯度阶段先稳定下来，其后余弦退火。

训练损失采用 TET（Temporal Efficient Training）：对 T 个时间步的输出各算一次
交叉熵再平均，而非先对 T 求平均再算一次。逐步损失给每个时间步直接的梯度信号，
能显著缓解低 T SNN 训练早期陷入的损失平台（实测仅对 T-均值算 CE 时 loss 平台
在 4.9 / top-1≈23% 处停滞）。

环境变量：
  TOTAL_STEPS  总训练步数（默认 320000，约 2 个 ImageNet epoch @BATCH=8）
  BATCH        批大小（默认 8；T=4 下经 VGG16 的等效批为 4*BATCH）
  LR           AdamW 峰值学习率（默认 2e-4）
  WARMUP       线性升温步数（默认 2000）
  PRINT_EVERY  打印间隔步数（默认 100）
  CKPT_EVERY   评估+存档间隔步数（默认 4000）
  EVAL_N       每次评估用的 val 图片数（默认 5000）
  SMOKE        设置则跑 60 步的冒烟测试后退出

产出：vgg16_snn_imagenet.pth（best）、vgg16_snn_imagenet_latest.pth（含续训状态）。
"""
import os, sys, time, math, torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision import transforms, datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vgg16_test import VGG16SNN, NUM_CLASSES, T
from spikingjelly.activation_based import neuron, functional

IMAGENET = "/home/charlley/Code/Dataset/imagenet"
CKPT_BEST = os.path.join(HERE, "vgg16_snn_imagenet.pth")
CKPT_LAST = os.path.join(HERE, "vgg16_snn_imagenet_latest.pth")

TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 320000))
BATCH = int(os.environ.get("BATCH", 8))
LR = float(os.environ.get("LR", 2e-4))
WARMUP = int(os.environ.get("WARMUP", 2000))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", 100))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 4000))
EVAL_N = int(os.environ.get("EVAL_N", 5000))
SMOKE = bool(os.environ.get("SMOKE"))
if SMOKE:
    TOTAL_STEPS, CKPT_EVERY, EVAL_N, PRINT_EVERY, WARMUP = 60, 50, 200, 10, 10

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
val_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), norm])
print("索引 ImageNet（train 约 128 万张，首次扫描需 1-2 分钟）...", flush=True)
train_ds = datasets.ImageFolder(os.path.join(IMAGENET, "train"), train_tf)
val_ds = datasets.ImageFolder(os.path.join(IMAGENET, "val"), val_tf)
# persistent_workers=False：训练循环内层 for 会遍历完整数据集（约 16 万步）才重建
# worker，故持久 worker 几乎无加速；而它会在主进程异常时令 DataLoader 关闭流程
# 死锁、占着显存不退出（无人值守的多天任务必须避免）。
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                       num_workers=10, pin_memory=True, drop_last=True)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=True,
                                     num_workers=4, pin_memory=True)

print("构建 VGG16-SNN，载入 ImageNet ANN 权重作初始化 ...", flush=True)
ann = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
model = VGG16SNN(NUM_CLASSES)
WT = (nn.Conv2d, nn.BatchNorm2d, nn.Linear)
for a, s in zip([m for m in ann.modules() if isinstance(m, WT)],
                [m for m in model.modules() if isinstance(m, WT)]):
    s.load_state_dict(a.state_dict())
del ann
functional.set_step_mode(model, 'm')
model = model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


def lr_at(step):
    """前 WARMUP 步线性升温 0→1，其后余弦退火 1→0（作用于 base lr 的倍率）。"""
    if step < WARMUP:
        return (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

lif_modules = [m for m in model.modules() if isinstance(m, neuron.LIFNode)]


def forward_seq(x):
    """整个 SNN 前向，返回逐时间步输出 [T, N, num_classes]。"""
    xseq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)   # [T,N,C,H,W]
    functional.reset_net(model)
    return model(xseq)                            # [T,N,num_classes]


@torch.no_grad()
def evaluate(n_max):
    model.eval()
    correct = total = 0
    for x, y in val_ld:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            out = forward_seq(x).mean(0)          # 对 T 平均后分类
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if total >= n_max:
            break
    model.train()
    return 100 * correct / total


@torch.no_grad()
def calibrate_thresholds(target=0.2):
    """逐层数据驱动地校准 LIF 阈值，消除 ANN→LIF 的网络坍缩。

    ANN 权重喂进默认阈值(1.0)的 LIF 会令各层全静默、网络坍缩到均匀输出、梯度
    落在平坦区无法学习（实测各层发放率≈0%）。本函数按前向顺序逐层校准：第 L 层
    的输入取决于前 L-1 层已定的阈值，故每定一层就重跑一次前向取其输入，再二分
    搜索阈值使该层输出发放率≈target（不可达时取下界，即该层尽量多发放），为
    替代梯度训练提供一个不坍缩的起点。
    """
    calib_imgs, _ = next(iter(val_ld))
    cx = calib_imgs.to(device).unsqueeze(0).repeat(T, 1, 1, 1, 1)
    model.eval()
    cap = {}

    def probe_rate(x, thr):
        probe = neuron.LIFNode(v_threshold=thr)
        functional.set_step_mode(probe, 'm')
        probe.eval()
        return (probe(x) != 0).float().mean().item()

    for li, lif in enumerate(lif_modules):
        h = lif.register_forward_pre_hook(
            lambda m, inp: cap.__setitem__('x', inp[0]))
        functional.reset_net(model)
        model(cx)
        h.remove()
        x = cap['x']
        lo, hi = 0.02, 50.0
        for _ in range(24):                       # 二分：发放率随阈值单调下降
            mid = (lo + hi) / 2
            if probe_rate(x, mid) > target:
                lo = mid                          # 发放过多 → 阈值需调高
            else:
                hi = mid
        thr = round((lo + hi) / 2, 4)
        lif.v_threshold = thr
        print(f"  LIF{li:2d}: 阈值={thr:.3f}  发放率={probe_rate(x, thr)*100:.1f}%",
              flush=True)
        del x
        torch.cuda.empty_cache()
    model.train()
    return [m.v_threshold for m in lif_modules]


def save(path, acc, full):
    obj = {'state_dict': model.state_dict(), 'global_step': global_step,
           'acc': acc, 'best': best, 'T': T, 'thresholds': thresholds}
    if full:                                      # 续训存档：附带优化器状态
        obj.update(opt=opt.state_dict(), sched=sched.state_dict(),
                   scaler=scaler.state_dict())
    torch.save(obj, path)


print("逐层校准 LIF 阈值（消除 ANN→LIF 的网络坍缩）...", flush=True)
thresholds = calibrate_thresholds()

global_step, best = 0, 0.0
if os.path.exists(CKPT_LAST):
    ck = torch.load(CKPT_LAST, map_location='cpu', weights_only=False)
    if 'global_step' in ck:                       # 本脚本的续训存档
        model.load_state_dict(ck['state_dict'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        global_step, best = ck['global_step'], ck.get('best', 0.0)
        if 'thresholds' in ck:                    # 沿用存档时的阈值
            thresholds = ck['thresholds']
            for m, th in zip(lif_modules, thresholds):
                m.v_threshold = th
        print(f"  断点续训：从 step {global_step}/{TOTAL_STEPS} 继续（best {best:.2f}%）",
              flush=True)


if global_step == 0:
    print(f"  校准后（未训练）SNN top-1 ≈ {evaluate(500):.2f}%  "
          f"(>0% 即网络已不再坍缩、可训练)", flush=True)

print(f"开始训练：TOTAL_STEPS={TOTAL_STEPS} BATCH={BATCH} T={T} LR={LR:.1e} "
      f"WARMUP={WARMUP} CKPT_EVERY={CKPT_EVERY} SMOKE={SMOKE}", flush=True)
model.train()
running, running_g, running_n, blk_t0 = 0.0, 0.0, 0, time.time()
done = False
while not done:
    for x, y in train_ld:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out_seq = forward_seq(x)                       # [T,N,num_classes]
            loss = sum(crit(out_seq[t], y) for t in range(T)) / T   # TET 逐步损失
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                                    # 裁剪前先反缩放
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        global_step += 1
        running += loss.item()
        running_g += float(gnorm)
        running_n += 1

        if global_step % PRINT_EVERY == 0:
            dt = (time.time() - blk_t0) / running_n
            print(f"  step {global_step}/{TOTAL_STEPS}  loss={running/running_n:.3f}  "
                  f"grad={running_g/running_n:.2f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"{dt:.3f}s/it", flush=True)
            running, running_g, running_n, blk_t0 = 0.0, 0.0, 0, time.time()

        if global_step % CKPT_EVERY == 0 or global_step >= TOTAL_STEPS:
            acc = evaluate(EVAL_N)
            if acc > best:
                best = acc
                save(CKPT_BEST, acc, full=False)
            save(CKPT_LAST, acc, full=True)
            print(f"[step {global_step}] val top-1 = {acc:.2f}%  (best {best:.2f}%)  "
                  f"已存档", flush=True)
            running, running_g, running_n, blk_t0 = 0.0, 0.0, 0, time.time()

        if global_step >= TOTAL_STEPS:
            done = True
            break

print(f"训练结束。best top-1 = {best:.2f}%", flush=True)
if SMOKE:
    print("SMOKE 测试通过。", flush=True)
