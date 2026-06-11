"""自动分块推理（AutoChunkInference）：在**不触发 OOM** 的前提下**最大化利用 GPU 显存**。

适用对象：在"全 T / 多步"模式下会把激活 ``[T, B, ...]`` 整段物化的 SNN —— 典型是
**SpikingJelly 多步 + triton 路线**（神经元 ``step_mode='m', backend='triton'``，整段 T 一次前向），
也包括 snn_compiler 自身的有状态前向。这类模型峰值显存 ∝ T，大 T 必 OOM。

本模块**不修改、不依赖任何被包装模型/三方库的内部**：它只在外层把时间维 T 切成 ``chunk_t``，
``reset`` 一次后按块喂 ``x[i:i+chunk_t]``、**块间不 reset**，靠神经元自身的状态跨块串接
（已实测 SpikingJelly 多步神经元满足此性质：分块多步 ≈ 整段多步，差异仅为卷积按 ``[chunk·B]`` vs
``[T·B]`` 批次的 cuDNN 数值差 ~1e-3，与"多步 vs 单步"同源、功能等价）。

自动选块策略（基于本仓库 experiments 的实测定律）：
- **显存**：分块峰值在 chunk 上**仿射** ``mem ≈ a·chunk + b``（实测 R²≈1）。本模块用两次小 chunk 探针拟合
  ``(a, b)``，解出**能放进显存预算的最大 chunk**，从而把显存用满又不 OOM；并带"真跑一次、OOM 就减半"的兜底。
- **速度 / 编译（compute-bound vs launch-bound 的不同策略）**：两类网络的最优 chunk 策略不同，但都受
  同一个**编译天花板**约束——神经元 kernel 用 ``static_range(chunk)`` 在编译期展开，编译耗时随 chunk
  **超线性**增长，**与 regime 无关**（实测：chunk=64 可编译，chunk≥128 编译卡死 / full@T=256 编译不返回）。
  因此两类都封顶在**编译安全上限**（``compile_cap`` / ``launchbound_cap``，默认均 64）。差别在于**谁先成为瓶颈**：
  - **launch-bound**（小 batch/小空间/碎小算子）：chunk 越小越慢（实测最高 54×），而显存极宽裕 →
    选块由**编译上限**封顶（≈64）——这是"为速度尽量取大块"的策略。
  - **compute-bound**：速度对 chunk 不敏感，但激活 ∝ chunk → 选块由**显存预算**先封顶（常 <上限）——
    这是"为显存利用率取最大可放块、速度无损"的策略。
  本模块用一次（已热身、排除编译）的轻量速度探针判别 regime 并记入 ``last_plan`` 供诊断。
  （注：早期设想 launch-bound 应取**远大于** compute-bound 的块；实测表明 >64 触发编译悬崖且速度收益已饱和，
  故两者收敛到同一编译安全上限，真正的差异体现在"由显存还是由编译上限封顶"。）

这是对 snn_compiler 的**增量、非侵入**新增：不改任何既有 kernel/pass/module，既有推理加速功能不受影响。
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn


def _default_reset(model: nn.Module) -> None:
    """duck-type 复位：调用每个带 reset() 的子模块（兼容 SpikingJelly 神经元，不 import 它）。"""
    for m in model.modules():
        r = getattr(m, "reset", None)
        if m is not model and callable(r):
            try:
                r()
            except Exception:
                pass


def _as_tensor(y):
    return y[0] if isinstance(y, (tuple, list)) else y


class AutoChunkInference(nn.Module):
    """把一个"整段 T 物化"的 SNN 包成自动分块推理。``forward(x_seq)`` 接 ``[T, B, ...]``。

    Args:
        model: 被包装模型（其神经元需在不 reset 时跨调用串接状态，如 SJ 多步神经元）。
        reset_fn: ``callable(model)`` 复位状态；默认 duck-type 调子模块 ``reset()``。
        memory_fraction: 用"当前空闲显存"的多大比例做激活预算（默认 0.85，留余量防碎片/他人占用）。
        compile_cap: compute-bound 时 chunk 的上限（默认 64；限制 static_range 展开、避免超线性编译）。
        launchbound_cap: launch-bound 时 chunk 的上限（默认 64；同为编译安全上限——实测 >64 触发编译悬崖，
            且 launch 开销在 ~64 帧后已摊薄、再增大块速度收益饱和。若你的 kernel 在大展开下编译仍很快，可上调）。
        fixed_chunk: 若给定则跳过自动选块，直接用它（便于复现/调试）。
        min_chunk: 自动选块的下限（默认 1）。
        verbose: 打印选块过程。
    """

    def __init__(self, model: nn.Module, *, reset_fn=None, memory_fraction: float = 0.85,
                 compile_cap: int = 64, launchbound_cap: int = 64,
                 fixed_chunk: int | None = None, min_chunk: int = 1, verbose: bool = False):
        super().__init__()
        self.model = model
        self.reset_fn = reset_fn
        self.memory_fraction = float(memory_fraction)
        self.compile_cap = int(compile_cap)
        self.launchbound_cap = int(launchbound_cap)
        self.fixed_chunk = fixed_chunk
        self.min_chunk = int(min_chunk)
        self.verbose = verbose
        self._cache: dict = {}        # (T,B,trailing) -> chunk_t
        self.last_plan: dict | None = None

    def _reset(self):
        (self.reset_fn or _default_reset)(self.model)

    @torch.no_grad()
    def _probe(self, x_seq, c):
        """跑一次 chunk=c 的分块前向，返回 (新增激活字节, 单次该块前向墙钟 ms)。

        ``added`` = 运行该块时分配器峰值 − 运行前基线（≈ 模型权重）。基线在 ``empty_cache`` 后取，
        故 ``added`` 即一块所需的激活（含跨块串接的膜电位 v）。
        """
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        self._reset()
        t0 = time.perf_counter()
        self.model(x_seq[0:c].contiguous())
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        added = torch.cuda.max_memory_allocated() - base
        return added, dt

    @torch.no_grad()
    def _timed_probe(self, x_seq, c):
        """先热身一次（吃掉 triton 编译 / cuDNN autotune），再测——使耗时反映稳态而非冷启动。"""
        self._probe(x_seq, c)            # 热身：编译/autotune，丢弃
        return self._probe(x_seq, c)     # 稳态测量

    @torch.no_grad()
    def _select_chunk(self, x_seq) -> int:
        """**实测倍增搜索**选块：只对"预测能放下"的尺寸真正探针，绝不主动 OOM（OOM 会污染分配器，
        使后续更小的尺寸也放不下）。逐步把 chunk 翻倍并实测峰值，直到再翻倍会超预算 / 触上限 / 到 T。

        为何不再用"仿射外推一步到位"：仿射律仅对 snn_compiler 自带的融合 triton 卷积成立；SpikingJelly
        走 ``torch.nn.Conv2d``→cuDNN，大 batch 下 cuDNN 切换算法 / 申请工作区，显存对 chunk **非仿射**，
        从 2,4 外推到 64 会严重低估而 OOM。倍增搜索每步都在**真实尺寸**上测量，对 cuDNN 也稳健。
        """
        T = int(x_seq.shape[0])
        if self.fixed_chunk is not None:
            c = max(self.min_chunk, min(self.fixed_chunk, T))
            self.last_plan = dict(T=T, chunk_t=c, regime="fixed", source="fixed")
            return c
        torch.cuda.empty_cache()
        free0, total = torch.cuda.mem_get_info()
        budget = free0 * self.memory_fraction          # 给激活的显存预算（留余量防碎片/他人占用）
        c1, c2 = min(2, T), min(4, T)
        m1, t1 = self._timed_probe(x_seq, c1)
        m2, t2 = self._timed_probe(x_seq, c2) if c2 > c1 else (m1, t1)
        # regime 判别（已热身、排除编译）：单帧耗时随块增大是否显著下降 → launch-bound。
        per1, per2 = t1 / c1, t2 / max(1, c2)
        launch_bound = (c2 > c1) and (per1 > 1.5 * per2)
        cap = self.launchbound_cap if launch_bound else self.compile_cap
        # 仿射拟合仅用于报告（a·chunk+b）
        a = max(1.0, (m2 - m1) / max(1, c2 - c1)); b = m1 - a * c1
        # 倍增搜索：从已实测放得下的最大点出发，仅当"再翻倍预计 ≤ 预算"时才真正探针下一档
        c = c2 if m2 <= budget else (c1 if m1 <= budget else self.min_chunk)
        best_m = m2 if c == c2 else (m1 if c == c1 else m1)
        while c * 2 <= min(cap, T):
            if best_m > 0.45 * budget:                 # 再翻倍（≈2×）会逼近/超预算 → 收手，不冒险探针
                break
            cnext = c * 2
            mnext, _ = self._probe(x_seq, cnext)       # 安全：预计 ≈2·best_m ≤ 0.9·预算
            if mnext > budget:                          # 实测确认超预算 → 不采用，停
                break
            c, best_m = cnext, mnext
        chunk_t = max(self.min_chunk, min(c, T, cap))
        self.last_plan = dict(T=T, free_GiB=free0 / 2**30, budget_GiB=budget / 2**30,
                              a_MiB=a / 2**20, b_MiB=b / 2**20,
                              measured_peak_GiB=best_m / 2**30, cap=cap,
                              regime="launch-bound" if launch_bound else "compute-bound",
                              chunk_t=chunk_t, source="search")
        if self.verbose:
            print(f"[autochunk] T={T} free={free0/2**30:.2f}GiB regime={self.last_plan['regime']} "
                  f"→ chunk_t={chunk_t} (实测峰值≈{best_m/2**30:.2f}/预算{budget/2**30:.2f}GiB cap={cap})")
        torch.cuda.empty_cache()
        return chunk_t

    @torch.no_grad()
    def forward(self, x_seq):
        T = int(x_seq.shape[0])
        key = (T, int(x_seq.shape[1]), tuple(x_seq.shape[2:]))
        chunk_t = self._cache.get(key)
        if chunk_t is None:
            chunk_t = self._select_chunk(x_seq)
            self._cache[key] = chunk_t
        # 驱动：reset 一次 → 按块喂、块间不 reset；OOM 兜底：减半重试（理论上选块已留余量，极少触发）
        while True:
            try:
                self._reset()
                outs = []
                for i in range(0, T, chunk_t):
                    outs.append(_as_tensor(self.model(x_seq[i:i + chunk_t].contiguous())))
                out = torch.cat(outs, 0)
                if self.last_plan is not None:
                    self.last_plan["chunk_t"] = chunk_t          # 记录真正生效的 chunk
                return out
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk_t <= self.min_chunk:
                    raise
                chunk_t = max(self.min_chunk, chunk_t // 2)
                self._cache[key] = chunk_t
                if self.last_plan is not None:
                    self.last_plan["chunk_t"] = chunk_t
                if self.verbose:
                    print(f"[autochunk] OOM → 减半 chunk_t={chunk_t} 重试")
