# snn_compiler

A generalized Triton-backend optimization framework for **IF / LIF / CubaLIF /
EIF Spiking Neural Network inference**. Single MLIR-level pattern; multiple
neuron families; scalar / per-channel / per-neuron thresholds; soft & hard
reset; arbitrary `v_reset` constant; fp32 / bf16 / fp16; NCHW / NHWC.

- **Methodology**: [`Document/Paper/snn_compiler_paper.md`](../Document/Paper/snn_compiler_paper.md)
- **Exploration journal** (13 stages including IR captures and bench data):
  [`Document/Exploration/mlir-perf-exploration-journal.md`](../Document/Exploration/mlir-perf-exploration-journal.md)
- **Per-method docs** (§5 below)

## Quick start

```python
import torch
from snn_compiler.zoo import vgg16_snn, resnet18_snn, mobilenet_v2_snn

m = resnet18_snn(num_classes=1000, neuron="lif", tau=2.0,
                  soft_reset=False, layout="NHWC",
                  fused=True).cuda().eval().to(torch.bfloat16)
y = m(torch.randn(4, 16, 3, 224, 224, device='cuda', dtype=torch.bfloat16))
#                 ↑ T   ↑ B
```

For attaching the framework to an existing model, see
[Document/Skill/snn-compiler-usage-guide.md](../Document/Skill/snn-compiler-usage-guide.md).

---

## 1. Why this framework accelerates SNN — the principles

### 1.1 Where SNN inference time actually goes

We profiled VGG-16 / ResNet-{18,34} SNN at T = 4 / 16 / 64 / 128 (bf16 + NHWC,
BATCH=4, RTX 5070 Ti) by isolating the LIF-kernel sum from end-to-end via
forward hooks ([benchmarks/bench_largeT.py](benchmarks/bench_largeT.py)):

| Architecture | T | end-to-end ms/img | LIF kernel sum ms | **LIF share** |
|---|---:|---:|---:|---:|
| VGG-16 SNN | 4–128 | 2.04 → 62.24 | 1.15 → 36.82 | **~15%** |
| ResNet-18 SNN | 16–128 | 1.23 → 10.13 | 0.70 → 6.31 | **~15%** |
| ResNet-34 SNN | 16–128 | 1.98 → 16.23 | 1.12 → 9.79 | **~15%** |

> **Conv dominates at 85%, LIF at 15%, across all T and architectures.**

That number stays remarkably stable because conv output and LIF input scale
together by T. The implication for "what optimization helps":

| Bottleneck | Optimization knob | Maximum end-to-end gain |
|---|---|---|
| Conv (85%) | layout (NHWC), dtype (bf16), Triton conv backend (Inductor max_autotune) | **dominant** |
| LIF (15%) | kernel fusion / bandwidth tricks | bounded by 15% |
| Launch overhead | Conv-BN-Neuron fusion (eliminates 2 / 3 launches per layer) | this is what this framework primarily exploits |

This framework's measured 1.85–4.6× speedup vs naive PyTorch-SNN
([§4](#4-benchmarks)) **does not come from making the LIF kernel
faster** — it's already at GDDR7 peak (see §1.4). It comes from:

1. **Layer-level launch reduction** (Conv-BN-Neuron → 1 launch instead of 3)
2. **bf16 + NHWC** (tensor cores + smaller footprint)
3. **Residual fusion** (ResNet `out = neuron(conv_bn(x) + identity)` → 1 kernel)
4. **Eliminating Inductor's "spike output buffer" round-trip** (own kernel writes spike directly)

### 1.2 The unified kernel pattern: outer-parallel + T-register-loop

All four neuron kernels (IF/LIF/CubaLIF/EIF) share one template:

```python
@triton.jit
def kernel(x_ptr, spike_ptr, T, NCL, ..., BLOCK_NCL):
    pid = tl.program_id(0)                                  # outer: B*C*H*W tile
    ncl_idx = pid.to(tl.int64) * BLOCK_NCL + tl.arange(0, BLOCK_NCL).to(tl.int64)
    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)             # state in register
    for t in tl.static_range(0, T, 1):                       # T-loop inside kernel
        t_off = tl.full([], t, dtype=tl.int64) * NCL_i64    # ★ i64 for T·NCL > 2³¹
        x_t   = tl.load(x_ptr + t_off + ncl_idx, ...)
        v     = step(v, x_t)                                 # ← neuron-specific
        spike = (v >= v_th).to(...)
        v     = reset(v, spike, v_reset, RESET_MODE)
        tl.store(spike_ptr + t_off + ncl_idx, spike, ...)
```

Three properties make this optimal:

- **`grid = (NCL / BLOCK_NCL,)`** parallelizes over B·C·H·W. Every (b, c, h, w)
  has its own SM lane, no inter-lane synchronization needed.
- **`v` lives in register for all T steps** — no global-memory round-trip
  for state. Without this, naive SNN does `T × (load v + write v)` per neuron.
- **`tl.static_range(0, T)`** unrolls the time loop at compile time, so the
  compiler can schedule the load-compute-store pipeline across t steps.

This template **already hits the GPU's GDDR7 peak bandwidth** (§1.4), so there's no
further unrolling / staging / pipelining win available at the kernel level.

### 1.3 Three orthogonal fusion strategies covering all common SNN topologies

| Pattern | Module | Triton kernel | Use |
|---|---|---|---|
| `Conv→BN→IF/LIF` | `FusedConvBNNeuron` | `_bias_if_lif_kernel` (no residual) | VGG, plain ConvNet-SNN |
| `Conv→BN→Add→IF/LIF` | `FusedConvBNAddNeuron` | `_bias_if_lif_kernel` (residual=True) | ResNet's second conv path |
| `Add → IF/LIF` | `FusedAddNeuron` | same kernel, bias=None | multi-branch SNN merge points |
| `Linear→IF/LIF` | `FusedLinearNeuron` | same kernel | classifier head |
| Last-layer spike-count | `RateCodedLIFNode` | `_bias_if_lif_rate_kernel` (T-collapsed write) | architectures whose final LIF is a vote |

All five share the same scalar+constexpr design — `HAS_BIAS`,
`HAS_RESIDUAL`, `RESET_MODE`, `THR_MODE`, `CHANNEL_LAST`, `HAS_VINIT` are
constexprs, so Triton specializes the kernel at JIT time and the
specialization with no residual produces identical PTX to a pure
Conv-BN-LIF kernel. **No "abstraction tax".**

### 1.4 Why the kernel is already at GDDR7 peak

Microbenchmark of `_bias_if_lif_kernel` at multiple T, BATCH = 16, bf16+NHWC,
RTX 5070 Ti ([explore/large_T/bench_baseline.py](explore/large_T/bench_baseline.py)):

| T | NCL | Effective bandwidth (GiB/s) |
|---:|---:|---:|
| 4 | 3.2M | 705 |
| 16 | 3.2M | 712 |
| 64 | 3.2M | 707 |
| 128 | 3.2M | 704 |

RTX 5070 Ti GDDR7 advertised peak ≈ **672 GiB/s** (this card is *not* HBM —
it ships with GDDR7 chips). Our kernel sustains **705 GiB/s ≈ 105%** thanks
to L2-residency near small CTA tiles.

The arithmetic intensity is ~0.875 flop/byte (7 fmadd + 8 bytes load+store
per element per step), far below the roofline turning point (~30 flop/byte),
so the kernel is firmly **bandwidth-bound** at every T value tested. There
is no remaining optimization knob *within* the kernel — further gains must
come from **reducing total bytes moved across the network boundary**.

### 1.5 What this round of exploration discovered (May 2026)

Stage 13 of the exploration journal (linked above) ran four prototypes
focused on large T (up to T=128):

| Prototype | What it tried | Outcome |
|---|---|---|
| `tl.range` runtime loop | Replace `static_range`, smaller PTX | Bit-equal; perf neutral (0.99×) |
| Rate-coded last LIF | Sum-over-T spike count, skip per-t store | Kernel **2.13–2.21×** at T≥16; **end-to-end < 0.1%** because LIF is only 15% and rate-coded only replaces 1/15 LIFs |
| T-chunked execution | Carry v across chunks, smaller activation buffers | Bit-equal; **47% slower** but 8× lower peak memory |
| Pool epilogue fusion | Conv-BN-LIF-AvgPool2x2 in one kernel | Long autotune; gains overlap with rate-coded; deferred |

Plus one correctness discovery:

| Bug | Detection | Fix |
|---|---|---|
| `t × NCL × sizeof(elem)` precomputed as i32 in TTIR | T=128, BATCH=4, NCL=12.8M crashed at `cudaErrorIllegalAddress` once the byte offset exceeded 2³¹ ≈ 2.15 GiB | Force i64 in 4 kernels — see [Document/Skill/snn-i64-offset-fix.md](../Document/Skill/snn-i64-offset-fix.md) |

The honest summary: **kernel-level optimization for SNN is now closed**. The
framework hits the GPU's GDDR7 peak, the i64 bug is fixed, and rate-coded / chunked are
integrated for the architecture/memory cases that benefit. Remaining
end-to-end win at large T (85% of time) requires *conv-side* work — either
a custom Triton conv with LIF as epilogue, or algorithmic latency-coding
to reduce T itself.

---

## 2. How to choose parameters

This section is decision-oriented: input is your scenario, output is the
concrete setting.

### 2.1 Memory layout: NHWC vs NCHW

**Default: NHWC** unless you have a hard reason otherwise.

| Layout | When it wins |
|---|---|
| **NHWC (channels_last)** | bf16+tensor-core conv ; smaller scratch ; **2× faster** in our 50K bench |
| NCHW | fp32 only; integration into legacy code that exclusively reshapes by (B,C,H,W) |

The framework's NHWC path stores conv weights with `memory_format=channels_last`
at construction (FusedConvBNNeuron.\_\_init\_\_) and passes them straight to
`F.conv2d` — the only "NHWC-specific" code in user-land is the conv weight
conversion loop in zoo factories.

### 2.2 Precision: bf16 vs fp32

**Default: bf16** for inference; fp32 only when the model is verified to
need it (very unusual for SNN).

| dtype | Speed | Memory | Notes |
|---|---|---|---|
| **bf16** | ~2× (tensor cores) | ~½ | Loss vs fp32 < 0.1 ppt on ImageNet-class accuracy in our experiments |
| fp16 | ~2× | ~½ | Range narrower; can overflow on LIF v register (we use fp32 accumulator internally) |
| fp32 | baseline | baseline | Use only for tiny networks where speed is irrelevant |

The kernel always uses **fp32 for v register internally** regardless of
input dtype — so bf16 input + fp32 v + bf16 spike output preserves
LIF dynamics correctly and bit-equals fp32 down to spike values.

### 2.3 Neuron model: IF vs LIF vs CubaLIF vs EIF

Decision tree by architecture:

```
Is this an ANN converted to SNN?      → IF (no decay) or LIF with tau≈2
Custom-trained SNN with τ_mem in {2,4} → LIF
Two-state (synaptic + membrane) model? → CubaLIF
Need spike-frequency adaptation?       → EIF
```

If you don't know, **start with LIF, tau=2, decay_input=True** — that's
SpikingJelly's default and what most ANN→SNN conversions use.

The framework supports an explicit `decay` parameter on every neuron that
overrides the τ-derived default — useful for matching custom training
recipes.

### 2.4 Reset mode: soft vs hard

| Reset | Behavior | When |
|---|---|---|
| **hard** (`v ← v_reset` on spike) | Stable across τ; default in most SJ models | Inference, ANN→SNN conversion |
| soft (`v ← v − θ` on spike) | Preserves residual potential; needed by some surrogate-gradient training schemes | Trained SNN that specifically uses soft reset |

The two modes use the **same Triton kernel** (chosen by `RESET_MODE` constexpr,
zero runtime overhead), so this is purely a model-defined choice.

### 2.5 Time-step count T

Inference cost is **linear in T**. There's no "free T". Pick the smallest T
that meets your accuracy bar:

| T | Typical use | Latency multiplier |
|---|---|---|
| 1 | Best for latency-coded / temporal-binary SNN | 1× |
| 4 | Most ANN→SNN conversion papers, default for ImageNet classification | 4× |
| 16 | Higher-accuracy SNN training, low-rate inputs | 16× |
| 64–128 | Continuous event-camera streams, neural simulation | up to 128× |

If you need T=128 but the activation buffer doesn't fit, use
`ChunkedForward(model, chunk_t=16)` (§2.7).

### 2.6 BATCH

Find the saturation point by sweeping; framework is memory-bandwidth-bound so doubling
BATCH past the saturation point gives no throughput.

Empirical rules (RTX 5070 Ti, 16 GiB, bf16+NHWC):

| Architecture | BATCH × T = max | Saturation BATCH at T=4 |
|---|---:|---:|
| VGG-16 SNN | ~256–512 | 96 (peak 7.3 GiB) |
| ResNet-18 SNN | ~512+ | 128 (peak 1.9 GiB) |
| ResNet-34 SNN | ~512+ | 128 (peak 1.9 GiB) |
| MobileNet-V2 SNN | ~1024+ | 192 |

Once throughput stops growing (~5% change between two BATCH values), you've
saturated. Going further just costs memory for no speed.

### 2.7 When to use the new (Stage 13) options

| Option | Use when | Don't use when |
|---|---|---|
| **i64 byte offsets** | Always on. Hard requirement for T·NCL·dtype_size > 2 GiB. | — (already mandatory) |
| **`RateCodedLIFNode`** | Your network's final LIF is the *voting output* (spike sum → argmax) | Final layer is `nn.Linear` producing logits (most ANN-converted SNN); rate-coded would break the math |
| **`StatefulLIFNode` + `ChunkedForward`** | Activation `[T, B, …]` > GPU memory budget | Memory fits → 47% slower with no benefit |
| **`fuse_modules_path`** | Custom non-Sequential model (e.g., your own ResNet block) | Plain `nn.Sequential` (just use `fuse_snn_model`) |

### 2.8 Decision flowchart for end-to-end use

```
Standard ANN→SNN inference  (VGG/ResNet/MobileNet)
├─→ Use snn_compiler.zoo factory with fused=True
├─→ layout='NHWC', dtype=bf16
├─→ T as small as accuracy permits (typically 4)
└─→ BATCH at saturation point per §2.6

Custom topology  (your own model class)
├─→ Replace neurons with snn_compiler.nn.{IF,LIF,CubaLIF,EIF}Node
├─→ For Conv→BN→Neuron blocks: FusedConvBNNeuron
├─→ For residual blocks: FusedConvBNAddNeuron in second path
├─→ Else: fuse_modules_path(model, […]) for in-place fusion

Large T (>= 32) with memory pressure
├─→ Wrap model.forward_chunked(...) in ChunkedForward(model, chunk_t=16)
└─→ Use StatefulLIFNode for each LIF layer

Spike-count vote architecture
└─→ Replace the final LIFNode with RateCodedLIFNode
```

---

## 3. Support matrix

| Axis | Values |
|------|--------|
| Neuron model | IF, LIF (decay_input ±), CubaLIF, EIF |
| Decay | Every model accepts an explicit `decay` (CubaLIF: `decay_syn`/`decay_mem`) overriding the τ-derived default. `None` → use τ recipe. |
| Reset | soft (`v ← v − θ·spike`), hard (`v ← v_reset`) |
| `v_reset` | any constant (compiled into `constexpr`) |
| Threshold | scalar, per-channel `[C]`, per-neuron `[B·C·H·W]` |
| Memory layout | NCHW, NHWC (channels_last) |
| dtype | fp32, bf16, fp16 |
| T-axis range | **1 — 128** validated (i64 offsets, ChunkedForward path) |
| Output | spike train `[T, B, …]` or rate code `[B, …]` |

All combinations are bit-equal to a naïve PyTorch reference (**221 cases**
across `tests/test_correctness.py`, `test_graph_pass.py`,
`test_residual_and_zoo.py`, `test_largeT_and_rate.py`).

---

## 4. Benchmarks

### 4.1 VGG-16 SNN end-to-end (T=4, BATCH=32, RTX 5070 Ti)

| Config | Naive (ms/img) | Fused (ms/img) | Speedup |
|---|---|---|---|
| LIF / hard / bf16 / NHWC | 3.93 | **1.95** | **2.02×** |
| LIF / hard / fp32 / NHWC | 6.55 | 3.92 | 1.67× |
| LIF / hard / bf16 / NCHW | 2.92 | 2.28 | 1.28× |

At BATCH = 96, bf16+NHWC fused: **513 images/second**.

### 4.2 Multi-architecture (BATCH=16, T=4, bf16+NHWC)

| Architecture | Naive (ms/img) | Fused (ms/img) | Speedup |
|---|---|---|---|
| VGG-11 SNN | 2.10 | 1.14 | **1.84×** |
| VGG-16 SNN | 3.91 | 1.97 | **1.99×** |
| ResNet-18 SNN | 0.587 | 0.307 | **1.91×** |
| ResNet-34 SNN | 0.958 | 0.498 | **1.93×** |
| MobileNet-V2 SNN | 1.10 | 0.240 | **4.60×** |

### 4.3 50,000-sample benchmark vs SpikingJelly (BATCH @ saturation, bf16)

| Network | Backend | Total (s) | per-img (ms) | img/s | Speedup vs ours |
|---|---|---:|---:|---:|---:|
| VGG-16 SNN | **ours** | **97.6** | **1.95** | **513** | — |
|  | SJ-compile | 105.2 | 2.10 | 476 | 1.08× |
|  | SJ-eager | 161.4 | 3.22 | 310 | 1.65× |
| ResNet-18 SNN | **ours** | **16.0** | **0.319** | **3131** | — |
|  | SJ-compile | 17.4 | 0.347 | 2873 | 1.09× |
|  | SJ-eager | 24.5 | 0.488 | 2044 | 1.53× |
| ResNet-34 SNN | **ours** | **25.7** | **0.514** | **1945** | — |
|  | SJ-compile | 28.6 | 0.570 | 1751 | 1.11× |
|  | SJ-eager | 40.1 | 0.800 | 1247 | 1.56× |

Full report: [Document/Benchmark/snn-compiler-vs-spikingjelly.md](../Document/Benchmark/snn-compiler-vs-spikingjelly.md).

### 4.4 Large-T validation (Stage 13, BATCH=4, bf16+NHWC)

| Architecture | T | end-to-end ms/img | LIF kernel share |
|---|---:|---:|---:|
| VGG-16 SNN | 128 | 62.24 | 14.8% |
| ResNet-18 SNN | 128 | 10.13 | 15.6% |
| ResNet-34 SNN | 128 | 16.23 | 15.1% |

All T = 4 / 16 / 64 / 128 paths bit-equal to naïve; full table in
[Document/Skill/snn-large-T-analysis.md](../Document/Skill/snn-large-T-analysis.md).

---

## 5. Per-method documentation

| Document | What it covers |
|---|---|
| [Document/Paper/snn_compiler_paper.md](../Document/Paper/snn_compiler_paper.md) | Method paper: motivation, design, benchmark methodology |
| [Document/Skill/snn-compiler-usage-guide.md](../Document/Skill/snn-compiler-usage-guide.md) | How to attach the framework to any SNN (Sequential / ResNet / multi-branch / SJ migration) |
| [Document/Skill/snn-i64-offset-fix.md](../Document/Skill/snn-i64-offset-fix.md) | The T=128 correctness fix (Stage 13) |
| [Document/Skill/snn-rate-coded-output.md](../Document/Skill/snn-rate-coded-output.md) | Rate-coded mode — kernel-level 2.2× speedup explained |
| [Document/Skill/snn-t-chunked-execution.md](../Document/Skill/snn-t-chunked-execution.md) | T-chunked execution for memory-constrained large-T inference |
| [Document/Skill/snn-large-T-analysis.md](../Document/Skill/snn-large-T-analysis.md) | Honest analysis: what optimization works at large T and what doesn't |
| [Document/Benchmark/snn-compiler-vs-spikingjelly.md](../Document/Benchmark/snn-compiler-vs-spikingjelly.md) | 50K-sample comparison: ours vs SJ-eager vs SJ-compile |

---

## 6. Module layout

```
snn_compiler/
├── kernels/
│   ├── neurons.py         _if_lif/_cuba_lif/_eif kernels (all i64-safe)
│   └── fused.py           _bias_if_lif (+residual/+stateful/+rate variants)
│                          conv_neuron, linear_neuron, conv_bn_neuron
│                          fold_conv_bn helper
├── nn/
│   ├── modules.py         IF/LIF/CubaLIF/EIFNode, FusedConv(BN)(Add)Neuron,
│   │                       FusedLinearNeuron, FusedAddNeuron,
│   │                       RateCodedIF/LIFNode, StatefulLIFNode
│   └── chunked.py         ChunkedForward, run_chunked
├── passes/
│   └── fuse.py            fuse_snn_model, fuse_modules_path
├── zoo/                   VGG, ResNet, MobileNet-V2 reference SNN
├── tests/                 221 bit-equal tests
│   ├── test_correctness.py
│   ├── test_graph_pass.py
│   ├── test_residual_and_zoo.py
│   └── test_largeT_and_rate.py
├── benchmarks/            50K vs SJ, multi-T, multi-architecture
│   ├── bench_vgg16.py
│   ├── bench_zoo.py
│   ├── bench_largeT.py
│   ├── comparison/        SJ-equivalent models + bench_50k.py
│   └── sweep_all.sh
└── explore/               Stage 13 prototypes (kept for reproducibility)
    └── large_T/
        ├── bench_baseline.py
        ├── kernel_variants.py
        ├── chunked_lif_proto.py
        └── pool_epilogue_proto.py
```

---

## 7. Status & limitations

- **Inference only.** Training mode (running BN stats + surrogate gradient backward) is not implemented.
- **v state is per-`forward` by default.** For continual-time models, use `StatefulLIFNode` and explicitly pass `v_init`/`return_v`.
- **Graph pass auto-matching covers `nn.Sequential`.** Residual / multi-branch topologies use the explicit `FusedConvBNAddNeuron` / `FusedAddNeuron` constructors or `fuse_modules_path`.
- **Conv backend is `F.conv2d` (cuDNN / Inductor's Triton).** Writing a Triton conv with LIF as epilogue is the next frontier for end-to-end gains beyond what this framework can currently deliver (the LIF kernel is memory-bandwidth-bound on the GPU's main memory; conv is ~85% of total).
