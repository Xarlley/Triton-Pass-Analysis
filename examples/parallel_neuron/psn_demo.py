"""并行脉冲神经元（PSN / MaskedPSN / SlidingPSN）调查与示例。

SpikingJelly 在 `spikingjelly/activation_based/neuron/psn.py` 提供三种「并行神经元」，
出自论文《Parallel Spiking Neurons with High Efficiency and Long-term Dependencies
Learning Ability》(arXiv:2304.12760)：

  PSN        H = W·X + B,  W ∈ R^{T×T}（可学习）, X ∈ R^{T×N};  S = Θ(H)
             —— 整个时间维一次稠密 GEMM 算完。W 把所有输入时间步线性混合成所有输出
             时间步，时间步之间全耦合。仅支持多步模式。
  MaskedPSN  H = (W ⊙ M_k)·X + B —— W 被带状下三角掩码 M_k 约束，输出步 t 只依赖
             输入步 t-k+1..t（因果、窗口 k）。多步前向仍是一次（掩码后的）GEMM。
  SlidingPSN H[t] = Σ_{i=0}^{k-1} W_i·X[t-k+1+i] —— k 个跨时间共享的权重，时间维退化
             为一维卷积（多步有 gemm / conv1d 两种后端）。

与 LIF 的根本区别
-----------------
LIF：逐步递推 v[t] = f(v[t-1], x[t])，时间维是**串行依赖**，必须按 t 顺序算。
并行神经元：用一个**显式的时间维矩阵乘 / 卷积**取代递推，去掉了串行依赖——这正是
论文标题里「Parallel」与「Long-term Dependencies」的来源。

本脚本：(1) 单独演示三种并行神经元的前向；(2) 用 PSN 搭一个小型 CNN-SNN，随机权重
跑一次推理；(3) 同结构换成 LIF 对照，说明二者可直接互换、但计算结构不同。

注意：仓库内 `spikingjelly/` 子模块版本才含 PSN（conda 环境里安装的旧版没有），故本
脚本把子模块路径插到 sys.path 最前。子模块的 layer 模块依赖 einops。
"""
import os, sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "spikingjelly"))   # 用含 PSN 的子模块版本
from spikingjelly.activation_based import neuron, layer, functional

torch.manual_seed(42)
T, N = 4, 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备 {device} | T={T} N={N}\n")


# ===================== 第一部分：三种并行神经元的前向 =====================
print("=" * 66)
print("第一部分：三种并行神经元单独前向")
print("=" * 66)
x = torch.rand(T, N, 8, device=device) * 4.0      # [T,N,8] 随机突触电流（放大以便发放）
neurons = [
    ("PSN(T=4)",          neuron.PSN(T=T)),
    ("MaskedPSN(k=2,T=4)", neuron.MaskedPSN(k=2, T=T, step_mode="m")),
    ("SlidingPSN(k=3)",    neuron.SlidingPSN(k=3, step_mode="m")),
]
for name, m in neurons:
    m = m.to(device)
    s = m(x)
    npar = sum(p.numel() for p in m.parameters())
    print(f"\n{name}")
    print(f"  输入 {tuple(x.shape)} → 输出 {tuple(s.shape)}；输出取值 ⊆ {{0,1}}: "
          f"{set(s.unique().tolist()) <= {0.0, 1.0}}，发放率 {s.mean().item():.3f}")
    print(f"  可学习参数 {npar} 个（时间维权重 {('T×T=%d' % (T*T)) if 'Sliding' not in name else 'k 个共享权重'} + 阈值）")
print("\n要点：三者的时间维计算都是「一次 GEMM / 卷积」，无逐时间步递推。")


# ===================== 第二/三部分：小型 CNN-SNN ==========================
def make_cnn(make_neuron):
    """conv-bn-神经元 ×2 + 池化 + fc-神经元 + fc 读出。神经元层由 make_neuron 决定。"""
    return nn.Sequential(
        layer.Conv2d(3, 16, 3, padding=1), layer.BatchNorm2d(16), make_neuron(),
        layer.Conv2d(16, 32, 3, padding=1), layer.BatchNorm2d(32), make_neuron(),
        layer.MaxPool2d(2),
        layer.Flatten(),
        layer.Linear(32 * 16 * 16, 64), make_neuron(),
        layer.Linear(64, 10),
    )


@torch.no_grad()
def run(net, name, xin):
    functional.set_step_mode(net, "m")          # conv/bn/pool/fc 切多步；PSN 本就多步
    net = net.to(device).eval()
    functional.reset_net(net)                   # PSN 无状态、此调用对其为空操作
    out = net(xin)                              # [T, N, 10]
    logits = out.mean(0)                        # 对 T 平均后作为分类输出
    npar = sum(p.numel() for p in net.parameters())
    print(f"\n{name}")
    print(f"  输入 {tuple(xin.shape)} → 输出 {tuple(out.shape)}  (= [T,N,类别])")
    print(f"  参数量 {npar:,}")
    print(f"  对 T 平均后预测类别: {logits.argmax(1).tolist()}")
    print(f"  输出 logits[0] = {logits[0].cpu().numpy().round(3)}")


xin = torch.rand(T, N, 3, 32, 32, device=device)    # [T,N,C,H,W] 随机输入

print("\n" + "=" * 66)
print("第二部分：小型 CNN-SNN（神经元 = PSN），随机权重跑一次推理")
print("=" * 66)
run(make_cnn(lambda: neuron.PSN(T=T)), "PSN-CNN", xin)

print("\n" + "=" * 66)
print("第三部分：对照——LIF 的逐步递推（并行神经元正是要消除它）")
print("=" * 66)
lif = neuron.LIFNode(step_mode="s")          # 单步模式，在 CPU 上逐步演示
xt = torch.rand(T, N, 8)
functional.reset_net(lif)
print("LIF 必须按时间步顺序前向，膜电位 v 在步间携带（v[t] 依赖 v[t-1]）：")
for t in range(T):
    s_t = lif(xt[t])
    v_mean = lif.v.mean().item() if torch.is_tensor(lif.v) else float(lif.v)
    print(f"  t={t}: 该步脉冲发放率 {s_t.mean().item():.3f}，"
          f"步末膜电位 v 均值 {v_mean:.3f}")
print("对比：PSN 在第一部分里一次 GEMM 就同时算出了全部 4 个时间步，无此串行链。")
print("（注：子模块版 spikingjelly 给 LIF 多步专门手写了 Triton kernel——但它与本仓库")
print("  定制版 Triton 的 API 不兼容、无法编译；侧面说明 LIF 递推连 spikingjelly 都要")
print("  专门写 kernel 去补救，而 PSN 是标准 GEMM、无需任何特殊 kernel。)")


# ============================== 观察 =====================================
print("\n" + "=" * 66)
print("观察（关系到 dev-plan）")
print("=" * 66)
print("""\
1. PSN / MaskedPSN / SlidingPSN 与 LIF 在网络里可直接互换（同样 [T,N,*] 进出）。
2. PSN 的神经元前向 = 一次 addmm（T×T 的 GEMM）+ 一次替代函数（Heaviside）；
   没有逐时间步递推，没有膜电位状态，没有 reset。
3. 时间维被 PSN 的稠密 T×T 权重「全耦合」——每个输出时间步依赖全部输入时间步；
   且 PSN 必须一次拿到全部 T 步输入才能算（强制全 T 物化）。
4. 对编译器而言，PSN 把 SNN 里唯一「非标准」的结构（串行递推）换成了标准 GEMM +
   逐元素算子——更像普通 CNN，留给「面向 SNN 的自定义 Pass」的特殊结构反而更少。""")
