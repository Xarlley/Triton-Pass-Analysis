"""AutoChunkInference 单元测试。

分两层：
1. **驱动正确性（CPU，无需 triton/CUDA）**：用一个跨调用累加膜电位的 mock 神经元，
   验证 "reset 一次 → 按 chunk 喂 → 块间不 reset" 的分块结果与整段**逐位一致**，
   且 fixed_chunk / 多次调用各自 reset / duck-type reset 都正确。
2. **自动选块（仅 CUDA）**：在真显存上跑探针+倍增搜索，验证选出的 chunk 合法、
   不 OOM、且分块输出与整段一致。

判定：mock 是线性累加（无阈值），分块与整段应**严格相等**（atol=0）。
"""
import pathlib
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from snn_compiler.nn import AutoChunkInference


class _CumNeuron(nn.Module):
    """跨调用累加的"膜电位"：forward([chunk,B,F]) 沿时间累加，状态 self.s 跨调用保留。

    它精确模拟 SpikingJelly 多步神经元"不 reset 即跨块串接"的语义，但纯 CPU、可逐位校验。
    """
    def __init__(self):
        super().__init__()
        self.s = None

    def reset(self):
        self.s = None

    def forward(self, x):
        s = self.s if self.s is not None else torch.zeros_like(x[0])
        out = []
        for t in range(x.shape[0]):
            s = s + x[t]
            out.append(s)
        self.s = s
        return torch.stack(out, 0)


class _MockSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.neuron = _CumNeuron()

    def forward(self, x):                       # x: [T,B,4]
        return self.neuron(self.lin(x))


def _ref_full(model, x):
    """整段参考：reset 一次后一次性前向。"""
    model.neuron.reset()
    return model(x)


def test_fixed_chunk_bit_exact_cpu():
    torch.manual_seed(0)
    model = _MockSNN()
    x = torch.randn(10, 3, 4)
    y_full = _ref_full(model, x)
    for chunk in (1, 2, 3, 4, 5, 7, 10, 20):    # 含整除/不整除/超过 T
        auto = AutoChunkInference(model, fixed_chunk=chunk)
        y = auto(x)
        assert y.shape == y_full.shape
        assert torch.equal(y, y_full), f"chunk={chunk} 分块结果与整段不一致"
        assert auto.last_plan["chunk_t"] == min(chunk, 10)


def test_repeated_calls_reset_each_time_cpu():
    """每次 forward 都 reset 一次 → 重复调用结果一致（状态不串到下一次 forward）。"""
    torch.manual_seed(1)
    model = _MockSNN()
    auto = AutoChunkInference(model, fixed_chunk=3)
    x = torch.randn(8, 2, 4)
    y1 = auto(x)
    y2 = auto(x)
    assert torch.equal(y1, y2)


def test_default_duck_type_reset_cpu():
    """不传 reset_fn 时，duck-type 调用子模块 .reset()；分块仍与整段一致。"""
    torch.manual_seed(2)
    model = _MockSNN()
    x = torch.randn(9, 2, 4)
    y_full = _ref_full(model, x)
    auto = AutoChunkInference(model, fixed_chunk=4)   # reset_fn=None → _default_reset
    assert torch.equal(auto(x), y_full)


def test_custom_reset_fn_called_cpu():
    flag = {"n": 0}

    def my_reset(m):
        flag["n"] += 1
        m.neuron.reset()

    model = _MockSNN()
    auto = AutoChunkInference(model, reset_fn=my_reset, fixed_chunk=2)
    auto(torch.randn(6, 1, 4))
    assert flag["n"] >= 1                         # 自定义 reset 确被调用


@pytest.mark.skipif(not torch.cuda.is_available(), reason="auto-select 需要 CUDA 显存探针")
def test_auto_select_cuda():
    torch.manual_seed(3)
    model = _MockSNN().cuda()
    x = torch.randn(64, 8, 4, device="cuda")
    y_full = _ref_full(model, x).clone()
    auto = AutoChunkInference(model, memory_fraction=0.5, compile_cap=32)
    y = auto(x)
    assert y.shape == y_full.shape
    # mock 无卷积/无阈值 → 分块严格相等
    assert torch.allclose(y, y_full, atol=1e-5)
    p = auto.last_plan
    assert 1 <= p["chunk_t"] <= 64
    assert p["regime"] in ("compute-bound", "launch-bound")


if __name__ == "__main__":
    test_fixed_chunk_bit_exact_cpu()
    test_repeated_calls_reset_each_time_cpu()
    test_default_duck_type_reset_cpu()
    test_custom_reset_fn_called_cpu()
    print("CPU autochunk tests passed")
    if torch.cuda.is_available():
        test_auto_select_cuda()
        print("CUDA autochunk test passed")
