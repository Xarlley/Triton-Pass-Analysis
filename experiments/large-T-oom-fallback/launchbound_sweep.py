"""验证「分块免费仅 compute-bound」的边界：造一组从 compute-bound 到 launch-bound 的配置，
逐个测分块的稳态速度，看小 chunk 何时、以多大程度变慢。

"launch-bound 程度"用每次 launch 的算力 ∝ B·H²（越小越 launch-bound）刻画。固定 T=64，对每个 (B,H)
扫 chunk∈{1,4,16,64=full}，测稳态时间，给出 chunk=1 相对 full 的减速倍数。

复用 chunk_sweep.py 的子进程 worker（每个单元全新 CUDA 上下文，残留内存隔离 + 冷启动剔除）。
跑法：python experiments/large-T-oom-fallback/launchbound_sweep.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chunk_sweep as cs    # 提供 run_cell（内部起 chunk_sweep.py --worker 子进程）


def main():
    T = 64
    chunks = [1, 4, 16, 64]
    # 从 compute-bound 到 launch-bound：B·H² 从 ~10万 降到 ~64（跨 ~1500×）
    configs = [(8, 112), (8, 32), (2, 32), (1, 32), (1, 16), (1, 8)]
    print(f"GPU: 子进程隔离 + 冷启动剔除；固定 T={T}，扫 chunk={chunks}（chunk=64=full）\n")

    res = {}
    for (B, H) in configs:
        for ck in chunks:
            res[(B, H, ck)] = cs.run_cell("chunked", T, ck, B, H, timeout=240)

    print(f"{'配置(B,H)':>10} | {'B·H²':>8} | " + " | ".join(f"{('ck='+str(c)):>8}" for c in chunks)
          + f" | {'chunk1/full':>11} | 判定")
    for (B, H) in configs:
        wpl = B * H * H
        cells = []
        for c in chunks:
            d = res[(B, H, c)]
            cells.append(f"{d['steady']:>8.2f}" if d.get("status") == "ok" else f"{d.get('status','?')[:8]:>8}")
        d1 = res[(B, H, 1)]; df = res[(B, H, 64)]
        if d1.get("status") == "ok" and df.get("status") == "ok" and df["steady"] > 0:
            ratio = d1["steady"] / df["steady"]
            rs = f"{ratio:>10.1f}×"
            verdict = "几乎免费" if ratio < 1.3 else ("略慢" if ratio < 3 else ("明显变慢" if ratio < 10 else "急剧变慢"))
        else:
            rs = f"{'-':>11}"; verdict = "-"
        print(f"{f'({B},{H})':>10} | {wpl:>8} | " + " | ".join(cells) + f" | {rs} | {verdict}")

    print("\n说明：稳态时间(ms)；chunk=64 即一块到底=full。chunk1/full = 把同一推理切成 64 块(逐时间步)相对一把梭的减速倍数。")
    print("LAUNCHBOUND_SWEEP_DONE")


if __name__ == "__main__":
    main()
