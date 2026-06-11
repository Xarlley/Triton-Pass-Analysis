"""Numerically verify the spikingjelly rewrites == the original repo models (fp32, torch backend).
Run with: SJ_NEURON_BACKEND=torch python verify_equiv.py
"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OTTT_REPO = "/home/liushifeng/charlley/snn_infer/repos/OTTT-SNN"
MSR_REPO = "/home/liushifeng/charlley/snn_infer/repos/MS-ResNet"
OTTT_CK = "/home/liushifeng/lsf/checkpoints/ottt_vgg11_cifar10/cifar10_ottta.pth"
MSR_CK = "/home/liushifeng/lsf/checkpoints/ms_resnet104_imagenet/resnet104.pth"
assert os.environ.get("SJ_NEURON_BACKEND") == "torch", "set SJ_NEURON_BACKEND=torch for exact compare"


def verify_ottt():
    print("\n===== OTTT VGG-11-WS equivalence =====")
    sys.path.insert(0, OTTT_REPO)
    from models import spiking_vgg
    from modules import neuron, surrogate
    import ottt_vgg_triton as MY
    orig = spiking_vgg.online_spiking_vgg11_ws(
        single_step_neuron=neuron.OnlineLIFNode, tau=2.0, surrogate_function=surrogate.Sigmoid(),
        track_rate=True, c_in=3, num_classes=10, neuron_dropout=0.0, grad_with_rate=True,
        fc_hw=1, v_reset=None).cuda().eval()
    orig.load_state_dict(torch.load(OTTT_CK, map_location="cpu")["net"])
    mine = MY.build_ottt(OTTT_CK).cuda().eval()
    x = torch.randn(4, 3, 32, 32, device="cuda")
    with torch.no_grad():
        out_o = None
        for t in range(MY.T_STEPS):
            o = orig(x, init=(t == 0))
            out_o = o if t == 0 else out_o + o
        out_m = mine(x)
    print("  orig logits[0,:5]:", out_o[0, :5].tolist())
    print("  mine logits[0,:5]:", out_m[0, :5].tolist())
    print("  max|orig-mine| = %.3e   argmax-match = %s" % (
        (out_o - out_m).abs().max().item(), bool((out_o.argmax(1) == out_m.argmax(1)).all())))


def verify_msresnet():
    print("\n===== MS-ResNet-104 neuron + load =====")
    sys.path.insert(0, MSR_REPO)
    import importlib
    M = importlib.import_module("models.MS_ResNet")
    orig_mem = M.mem_update().cuda().eval()          # original torch-loop neuron (time_window=6)
    from spikingjelly.activation_based import neuron, functional
    lif = neuron.LIFNode(tau=4.0 / 3.0, decay_input=False, v_threshold=0.5, v_reset=0.0,
                         step_mode="m", backend="torch").cuda().eval()
    x = torch.randn(6, 4, 16, 8, 8, device="cuda")   # [T,B,C,H,W]
    with torch.no_grad():
        o1 = orig_mem(x)
        functional.reset_net(lif); o2 = lif(x)
    print("  mem_update vs LIF(tau=4/3,soft=False,vth=0.5)  max|diff| = %.3e  spikes_equal=%s" % (
        (o1 - o2).abs().max().item(), bool((o1 == o2).all())))
    # full model load check (uses build which monkeypatches)
    import ms_resnet_triton as MR
    net = MR.build_msresnet104(MSR_CK)
    nparam = sum(p.numel() for p in net.parameters())
    print("  full MS-ResNet-104 built+loaded; params=%.2fM" % (nparam / 1e6))


if __name__ == "__main__":
    verify_ottt()
    verify_msresnet()
    print("\nDONE")
