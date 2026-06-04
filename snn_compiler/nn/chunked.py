"""T-chunked execution driver。

把整网 forward 拆成 chunk-by-chunk 推理：每个 chunk 在所有层之间流过去，
LIF 的 v 状态保存在 `chunk_state[layer_name] = v_tensor`，进入下一个 chunk
继续用。如此最大显存占用是 `chunk_t / T` 倍单 chunk 激活，让 T=128 的大模型
塞进 16 GiB 卡。

使用方式
========
两种风格：

1. `run_chunked(model, x_seq, chunk_t=16)` — 简便函数，要求 model.forward
   有 `chunk_state` 参数（默认 None）；逐 chunk 调用并 cat 输出。

2. `ChunkedForward(model, chunk_t=16)` — Module 包装，把 chunk_t 固化进
   wrapper；可像普通 module 一样调用 `wrapper(x_seq)`。

约束
====
- 模型里所有 LIF/IF/CubaLIF 必须替换为 StatefulLIFNode 之类持状态变体。
- 模型 forward 必须把所有"过时 t-维"操作（如 BN running stats、Conv
  T-axis padding）放在 chunk 内（current 框架天然满足，每个 chunk 都过
  完整 conv+BN）。
"""
from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn


def run_chunked(
    forward_fn: Callable[[torch.Tensor, dict | None], tuple[torch.Tensor, dict]],
    x_seq: torch.Tensor,
    *,
    chunk_t: int = 16,
) -> torch.Tensor:
    """通用 chunked driver。

    Args:
        forward_fn: 接受 ``(x_chunk, state_dict)`` 返回 ``(y_chunk, new_state_dict)``。
                    state_dict 是 ``{layer_name: v_tensor}``，第一次调用传 None。
        x_seq: ``[T, B, ...]``
        chunk_t: 每个 chunk 的 T 长度

    Returns:
        ``[T, B, ...]`` 拼接的最终输出。
    """
    T = x_seq.shape[0]
    if chunk_t >= T:
        out, _ = forward_fn(x_seq, None)
        return out
    chunks: List[torch.Tensor] = []
    state = None
    for i in range(0, T, chunk_t):
        c = min(chunk_t, T - i)
        x_c = x_seq[i:i + c].contiguous()
        y_c, state = forward_fn(x_c, state)
        chunks.append(y_c)
    return torch.cat(chunks, dim=0)


class ChunkedForward(nn.Module):
    """包装一个网络让 ``forward(x_seq)`` 内部走 chunked driver。

    包装的网络必须支持 ``model.forward_chunked(x_chunk, state)`` 接口，
    其中 ``state`` 是 dict 或 None。如果 model 没有该方法，回退到普通 forward
    （等价于无 chunking）。
    """
    def __init__(self, model: nn.Module, *, chunk_t: int = 16):
        super().__init__()
        self.model = model
        self.chunk_t = chunk_t

    def forward(self, x_seq):
        if not hasattr(self.model, "forward_chunked"):
            return self.model(x_seq)
        return run_chunked(self.model.forward_chunked, x_seq, chunk_t=self.chunk_t)
