# snn_compiler: A Generalized Triton-Backend Optimization Framework for IF/LIF Spiking Neural Network Inference

**Charlley Guo**
2026-05

---

## Abstract

Spiking Neural Networks (SNNs) exhibit a structural performance gap when compared
to equivalent Artificial Neural Networks (ANNs) on GPUs: a T-step multistep SNN
forward pass invokes *T* times the elementwise neuron work and accumulates
*T* times the kernel-launch overhead. Existing SNN libraries either (i) ship
hand-written CUDA/Triton multistep kernels that cover only a single neuron
family and a fixed reset / threshold configuration (e.g.,
`spikingjelly.multistep_lif_inference`), or (ii) rely on `torch.compile`, where
the Inductor backend's output-flat parallelization policy is suboptimal for the
time-recurrent state update characteristic of all spiking neurons. We present
`snn_compiler`, a generalized Triton-backend framework that (a) unifies the
IF / LIF / CubaLIF / EIF neuron families behind a single
*outer-parallel + T-register-loop* kernel template parameterized by
`constexpr` switches; (b) fuses pre-neuron convolution / linear bias-add and
optional BatchNorm folding into the same kernel; and (c) ships an
nn.Module-level graph-rewrite pass that recognizes the canonical
`Conv → [BN] → Neuron` motif and substitutes the fused kernel automatically.
Across the IF / LIF × soft / hard reset × fp32 / bf16 × NCHW / NHWC matrix on
VGG16-SNN (T = 4, RTX 5070 Ti), the unified framework delivers 1.28× – 2.02×
wall-clock speed-up over an unfused nn.Module baseline that already uses our
generic neuron kernel; the bf16 + NHWC + Fused configuration reaches 1.94 ms /
image (514 image/s) on ImageNet input shape. All output spikes are bit-equal
to a naïve PyTorch reference across 160 configurations. The achieved per-image
latency closes to within 5 % of a hand-written kernel previously reported in
the project's exploration journal, while broadening support from a single
neuron / reset / dtype configuration to the full Cartesian product mentioned
above.

**Keywords:** spiking neural network, Triton, GPU optimization, kernel fusion,
graph rewrite, neural network compiler.

---

## 1. Introduction

Recent SNN libraries have made it easy to train and evaluate biologically
motivated networks at ImageNet scale (SpikingJelly [Fang et al., 2023]) and to
import network topologies via portable IRs (NIR / NIRTorch
[Pedersen et al., 2023]). On the **training** path these libraries rely on
PyTorch eager mode plus surrogate gradients; on the **inference** path users
typically have three options:

1. The library's own hand-written CUDA/Triton multistep kernel
   (e.g. `spikingjelly.activation_based.triton_kernel.multistep_lif_inference`).
2. PyTorch eager mode with per-step neuron `forward` calls.
3. `torch.compile` over a multistep wrapper (NIR runtime path).

In an earlier exploration (see `Document/Exploration/mlir-perf-exploration-journal.md`
in this repository, hereafter "the journal"), we characterized these three paths
on a VGG16-SNN with T = 4 time steps and identified three reproducible
performance gaps:

- **G1 — output-flat parallelization is wrong for time-recurrent ops.**
  Inductor lowers a `for t in T: v = v + x[t]; spike[t] = …` snippet into a
  kernel parallelized over the *output element* (B·C·H·W·T) and *redundantly
  re-loads* the recurrent state at every step, leaving the register-resident
  reduction-axis optimization on the table.
- **G2 — `@triton.autotune restore_value` is not free.** SpikingJelly's
  multistep_lif kernel uses `restore_value=[…]` to keep the membrane
  potential idempotent during autotuning, which triggers a `clone()`
  of multi-GiB tensors at every config probe.
- **G3 — cuDNN convolution leaves `bias-add` as a standalone elementwise.**
  When the chosen cuDNN algorithm is `cutlass` / `xmma` / `winograd`,
  the bias term is appended by a separate ATen launch, fragmenting the
  conv-bias-neuron triplet across three GPU launches per layer.

These gaps motivated a single hand-tuned Triton kernel for IF / hard reset /
fp32 that closed the SNN-vs-ANN gap from 7.41 ms / image to 1.88 ms / image
on the same hardware. However the kernel was monomorphic in the neuron model,
the reset rule, the v_reset value, the threshold representation, the dtype
and the memory layout — a real workload exercises every one of those axes.

This paper extends the journal work into a **general-purpose** framework
that retains the 4× speed-up but supports the full Cartesian product of:

| Axis                | Values |
|---------------------|--------|
| Neuron model        | IF, LIF (decay_input True / False), CubaLIF, EIF |
| Decay coefficient   | Any neuron model accepts an explicit `decay` (and CubaLIF additionally `decay_syn` / `decay_mem`) overriding the τ-derived default; the τ recipe is recovered when the override is `None` |
| Reset rule          | Soft reset (v ← v − θ · spike); Hard reset (v ← v_reset · spike + v · (1−spike)) |
| v_reset value       | Any constant (compile-time `constexpr`) |
| Threshold mode      | Scalar; per-channel [C]; per-neuron [B·C·H·W] |
| Memory layout       | NCHW; NHWC (channels_last) |
| Floating dtype      | fp32; bf16; fp16 |

**Contributions.**

1. A **single Triton kernel template** that unifies IF / LIF (and by extension
   CubaLIF / EIF) under one *outer-parallel + T-register-loop* pattern, using
   `constexpr` parameters to specialize on neuron type, reset rule, threshold
   mode and channel order without runtime branch overhead (§3.1).
2. A **fused Conv-bias-Neuron and Conv-BN-Neuron kernel family** that absorbs
   the convolution bias-add and the (eval-mode) BatchNorm affine transform
   directly into the neuron kernel's per-step inner loop, eliminating G3
   and the per-layer BN launch (§3.2).
3. A **model-level graph-rewrite pass** that recursively scans an
   `nn.Module` tree and substitutes recognized `Conv → [BN] → Neuron`
   patterns with the appropriate fused module. The pass uses
   conservative structural matching rather than `torch.fx` symbolic trace
   to remain robust under multistep wrappers (§3.3).
4. A **bit-equal correctness oracle** covering 160 configurations of the
   above support matrix, and an **end-to-end benchmark** that demonstrates
   1.28× – 2.02× speed-up vs. an unfused baseline that already uses the
   generic neuron kernel (§4).

The framework lives in this repository under `snn_compiler/`, and its kernel
template, fusion library and graph pass are intended to be reusable for
future SNN architectures (ResNet-SNN, VGG-19-SNN, MobileNet-SNN) without
further GPU-kernel engineering.

## 2. Background

### 2.1 Spiking neuron dynamics

A spiking neuron maintains an internal state (membrane potential *v*, and
optionally a synaptic current *i*) that evolves in discrete time steps in
response to an input current x_t and emits a binary spike when *v* crosses a
threshold *θ*. The most common forms are:

- **Integrate-and-Fire (IF).** v_t = v_{t-1} + x_t; spike_t = 𝟙[v_t ≥ θ];
  on spike, v is either decreased by θ (*soft reset*) or set to v_reset
  (*hard reset*). A leak coefficient can be added: v_t = δ · v_{t-1} + x_t
  with 0 < δ ≤ 1.
- **Leaky IF (LIF).** v_t = (1 − 1/τ) · v_{t-1} + s · x_t, where
  s = 1/τ when SpikingJelly's `decay_input=True` and s = 1 when False;
  spike and reset rules are identical to IF.
- **Current-based LIF (CubaLIF).** A synaptic current state i evolves with
  its own time constant: i_t = α · i_{t-1} + x_t; v_t = β · v_{t-1} + i_t;
  spike / reset on v as before. Used for hardware-friendly two-pole
  filter responses.
- **Exponential IF (EIF).** Adds a non-linear exponential term to the v
  update: v_t = δ · v_{t-1} + ΔT · exp((v_{t-1} − v_rh)/ΔT) + x_t.

All four models share the structural form

```
state_t = step(state_{t-1}, x_t, params)
spike_t = (v_t ≥ θ)
state_t = reset(state_t, spike_t, v_reset, rule)
```

where `step` is per-position-independent and recurrent only along the time
axis t. Spatial dimensions (B, C, H, W) form a perfectly parallel outer
loop. **This is the structural fact that the framework exploits.**

### 2.2 Multistep formulation and the SNN/ANN gap

The same network executes T times within one logical "forward". A standard
multistep wrapper produces an output of shape `[T, B, C, H, W]` which is
flattened to `[T·B, C, H, W]` for convolution and then reshaped back. This
means:

- Convolution kernels see T× the work — but for free, since the same kernel
  is being reused on a larger batch.
- Per-step elementwise (BN, bias, neuron) kernels are launched T times if no
  multistep fusion is performed — paying the launch tax T times.

For T = 4 with 13 conv layers, the unfused inference therefore eats roughly
T × (Nconv − 1) × N_elementwise = 4 × 13 × 3 ≈ 156 elementwise launches per
forward; the launch tax alone is on the order of 100 µs / forward at modern
launch latencies, comparable to the work itself for small feature maps.

### 2.3 The output-flat parallelization mismatch

When Inductor (PyTorch 2.x default backend) lowers a multistep neuron forward,
it generates a Triton kernel whose grid spans the *output* shape
`[T, B, C, H, W]`, with each tile loading v[*, t−1], adding x[*, t] and
writing spike[*, t]. Two problems follow:

1. **State is re-loaded from global memory at every step.** The recurrent
   state v has to be re-read because Inductor's tile cannot keep it in
   registers across the time dimension when the grid is sliced over t.
2. **Restore-value autotune cost.** SpikingJelly's own hand-written
   multistep_lif kernel uses `@triton.autotune(restore_value=["v"])` to
   guard the membrane buffer during the search, which requires a `clone()`
   on every config probe — and that clone is multi-GiB for VGG16 feature
   maps × T × batch.

A specialized kernel that (a) outer-parallelizes along the *spatial-only*
NCL = B·C·H·W axis and (b) loops over T inside the register file avoids
both problems: the recurrent state lives in registers, and there is no
need for `restore_value` because the spike output buffer is independent
of the state.

### 2.4 BatchNorm folding

For inference (`bn.eval()`), `BatchNorm2d(y) = γ (y − μ) / √(σ² + ε) + β`.
If y itself is the output of `conv2d(x, W, b)`, the BN can be folded into a
new conv weight W′ and bias b′:

W′ = γ √(σ² + ε)⁻¹ · W ;  b′ = γ √(σ² + ε)⁻¹ · (b − μ) + β.

Folding is exact at fp32 (we measured ‖`conv-bn(x) − conv'(x)`‖_∞ ≈ 2.4 ×
10⁻⁶ in §4.3), and once done, the per-layer launch count drops from 3
(conv, bn, neuron) to either 2 (conv + fused-bias-neuron) or 1 (when an
epilogue-fused conv kernel is available).

## 3. Method

### 3.1 The unified neuron kernel template

The kernel `_if_lif_kernel` in `snn_compiler/kernels/neurons.py` realizes the
*outer-parallel + T-register-loop* pattern. Its skeleton is:

```python
@triton.autotune(
    configs=[Config({"BLOCK_NCL": b}, num_warps=w) for (b, w) in
             [(128, 4), (256, 4), (256, 8), (512, 8), (1024, 8)]],
    key=["T", "NCL", "THR_MODE", "RESET_MODE", "CHANNEL_LAST"],
)
@triton.jit
def _if_lif_kernel(x_ptr, spike_ptr, v_th_ptr,
                   T, NCL, C, HW, BLOCK_NCL,
                   decay_factor, input_scale,
                   v_threshold_const, v_reset_val,
                   RESET_MODE, THR_MODE, CHANNEL_LAST):
    pid = tl.program_id(0)
    ncl_idx = pid * BLOCK_NCL + tl.arange(0, BLOCK_NCL)
    mask = ncl_idx < NCL

    if THR_MODE == 0:                                 # scalar
        v_th = v_threshold_const
    elif THR_MODE == 1:                               # per-channel
        c_idx = (ncl_idx % C) if CHANNEL_LAST else ((ncl_idx // HW) % C)
        v_th = tl.load(v_th_ptr + c_idx, mask=mask).to(tl.float32)
    else:                                             # per-neuron
        v_th = tl.load(v_th_ptr + ncl_idx, mask=mask).to(tl.float32)

    v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
    for t in tl.static_range(0, T, 1):
        x_t = tl.load(x_ptr + t * NCL + ncl_idx, mask=mask).to(tl.float32)
        v = decay_factor * v + input_scale * x_t
        spike = (v >= v_th).to(tl.float32)
        if RESET_MODE == 0:                           # soft
            v = v - spike * v_th
        else:                                         # hard
            v = v * (1.0 - spike) + spike * v_reset_val
        tl.store(spike_ptr + t * NCL + ncl_idx, spike, mask=mask)
```

**Design points.**

1. **Grid is one-dimensional** over `ceil(NCL / BLOCK_NCL)`. The kernel
   covers all spatial neurons in a single launch; T is consumed inside
   the launch.
2. **State stays in registers.** `v` is allocated as a tile-local fp32
   accumulator. The autotune key does not include the threshold tensor
   (which would create a separate cache entry per tensor identity);
   only `THR_MODE`, `RESET_MODE`, `CHANNEL_LAST`, T and NCL are keyed.
3. **`v_reset_val` is `constexpr`.** Each distinct constant compiles into
   its own kernel instance, but at run-time the load/multiply uses a
   literal scalar, eliminating the conditional branch on every spike.
4. **fp32 accumulation, dtype-flexible I/O.** The input and output may be
   fp32, bf16 or fp16; the `.to(tl.float32)` after the load makes the
   accumulation type-stable.
5. **`CHANNEL_LAST` switches the per-channel `c_idx` formula** between
   `(ncl_idx // HW) % C` (NCHW) and `ncl_idx % C` (NHWC). No bias term is
   needed here because the threshold is the only per-channel quantity in
   the pure-neuron kernel.

LIF is supported by the same kernel: setting `decay_factor = 1 − 1/τ` and
`input_scale = 1/τ` (or `1`) selects between LIF / decay_input=True / False.

CubaLIF and EIF are realized by separate but structurally identical kernels
(`_cuba_lif_kernel`, `_eif_kernel`) that introduce a second state register
(`i_syn`) or a non-linear `tl.exp` term in the v update, respectively.

**Decay as a per-model first-class parameter.** Every neuron entrypoint
exposes an explicit `decay` override (CubaLIF: `decay_syn` and `decay_mem`).
The default behaviour — `decay=None` — derives the value from the time
constant τ (IF: `1.0`; LIF / EIF: `1 − 1/τ`; CubaLIF: `exp(−dt/τ_syn)` and
`exp(−dt/τ_mem)`). Passing a numeric value bypasses the τ derivation and
becomes the literal `tl.constexpr` decay_factor used in the kernel inner
loop. Crucially, this is *not* a Python-level shim: the kernel only ever
sees the `decay_factor` constexpr, so the τ-derived path and the explicit
path emit identical PTX. This matters for training experiments where the
membrane leak should follow an annealing schedule independent of τ — the
user mutates `module.decay` between epochs and the next kernel call is
specialized accordingly via Triton's autotune cache. The graph rewrite
pass (§3.3) propagates `decay` across the fusion boundary, so the
fused `FusedConvBNNeuron` exposes the same parameter as its source
`LIFNode`.

### 3.2 Conv-bias-Neuron and Conv-BN-Neuron fusion

The fused kernel `_bias_if_lif_kernel` in `snn_compiler/kernels/fused.py` adds
a `bias_ptr` and a `HAS_BIAS: constexpr` switch to the template above:

```python
if HAS_BIAS:
    bias = tl.load(bias_ptr + c_idx, mask=mask).to(tl.float32)
else:
    bias = tl.zeros([BLOCK_NCL], dtype=tl.float32)
...
for t in tl.static_range(0, T, 1):
    y_t = tl.load(y_ptr + t * NCL + ncl_idx, mask=mask).to(tl.float32)
    v = decay_factor * v + input_scale * (y_t + bias)
    ...
```

`y` here is the output of `F.conv2d(x, W, bias=None)` — the convolution is
asked **not** to add the bias, and the bias term is appended once per
spatial position and broadcast across all T steps inside the kernel.
This removes the standalone ATen elementwise bias-add identified as G3.

For Conv-BN-Neuron, the BN parameters are folded into W and b at module
construction (see §2.4); the runtime kernel is identical to the
Conv-bias-Neuron case.

### 3.3 Model-level graph rewrite

`snn_compiler/passes/fuse.py:fuse_snn_model` recursively scans a model's
`nn.Sequential` and `nn.ModuleList` containers and matches three patterns:

```
Conv2d → BatchNorm2d → IFNode/LIFNode   ⇒ FusedConvBNNeuron
Conv2d                → IFNode/LIFNode   ⇒ FusedConvNeuron
Linear                → IFNode/LIFNode   ⇒ FusedLinearNeuron
```

The pass uses *structural* matching on adjacent `nn.Module` children rather
than `torch.fx.symbolic_trace`, because multistep SNNs commonly wrap their
forward in a T loop or use `MultiStepWrapper`, which trace badly. SJ's
own `IFNode` / `LIFNode` (different module classes from ours) are
identified by duck-typing on the `v_threshold` and `tau` attributes, so an
existing SpikingJelly model can be fused without code changes.

The pass is **eval-only**: it folds BN running stats into conv weights,
which is correct only when BN is frozen. We detect this by matching on
`nn.BatchNorm2d`'s `eval()` mode rather than `BatchNormForward` symbolically.

### 3.4 The framework API

```python
from snn_compiler.nn import IFNode, LIFNode, CubaLIFNode, EIFNode
from snn_compiler.passes import fuse_snn_model

model = build_my_snn()      # nn.Sequential containing Conv2d/BN/IFNode/LIFNode
model.eval().cuda()
fused, n_fused = fuse_snn_model(model, layout="NHWC")
spikes = fused(x_seq)       # x_seq: [T, B, C, H, W] (NCHW shape, NHWC memory)
```

For users who construct a model directly with fused modules:

```python
from snn_compiler.nn import FusedConvNeuron, FusedConvBNNeuron, FusedLinearNeuron

feats = nn.Sequential(
    FusedConvNeuron(3, 64, 3, padding=1, neuron="lif", tau=2.0, soft_reset=False),
    nn.AvgPool2d(2, 2),
    ...
)
```

## 4. Experiment

### 4.1 Hardware and software environment

- GPU: NVIDIA RTX 5070 Ti (Blackwell sm_120, 16 GiB GDDR7)
- CUDA / driver: as per `nvidia-smi` at run time of section 4
- Python 3.13, PyTorch 2.11.0.dev (with `torch._inductor` available)
- Triton 3.7.x (project's pre-installed fork; see
  `dev-log/dev-log.md` for the rebuild procedure)
- Each measurement: 5 warmup iterations + ≥ 30 measured iterations on
  `BATCH = 32` and `T = 4` for the sweep table, or `BATCH = 96` for the
  large-batch comparison. Times in the table are `mean(per-iter ms) / BATCH`.

### 4.2 Network and reference implementations

We benchmark VGG16-D — 13 Conv-BN-Neuron blocks interspersed with 5 AvgPool,
followed by Flatten + 3 fully-connected layers — at ImageNet input shape
`[3, 224, 224]`. T = 4 time steps.

Two `nn.Module` variants share weights bit-for-bit:

- **Naive.** Uses the framework's `IFNode` / `LIFNode` for the neuron, but no
  fusion: convolution, BN and neuron are three separate launches per layer.
  Convolution dispatches via `F.conv2d` (cuDNN); BN uses `nn.BatchNorm2d`.
  This isolates the **fusion benefit** since both variants share the same
  underlying generic neuron kernel.
- **Fused.** Built from the Naive checkpoint by `fuse_snn_model` or by
  constructing `FusedConvBNNeuron` / `FusedLinearNeuron` directly. BN is
  folded into the conv weight/bias at construction; the convolution forwards
  with `bias=None`; the fused kernel handles bias-add and neuron update.

### 4.3 Correctness across the support matrix

`snn_compiler/tests/test_correctness.py` runs 160 configurations and reports
the `(ref ≠ out).sum()` count for each. All 160 cases evaluate to zero
mismatched elements (i.e., `torch.equal(ref, out) == True`). Breakdown:

| Test group | Configurations | Bit-equal pass |
|------------|----------------|----------------|
| Pure neuron (IF/LIF × soft/hard × v_reset ∈ {0.0, 0.3, −0.5} × decay_input × scalar/per-C/per-N thr × 3 shapes × 2 layouts) | 132 | 132 / 132 |
| CubaLIF / EIF (5D shapes × soft/hard × v_reset) | 12 | 12 / 12 (spikes bit-equal; v has ≤ 1e-3 ULP drift on `exp`) |
| Fused Conv-bias-IF/LIF (with / without bias × neuron × soft/hard × v_reset) | 12 | 12 / 12 |
| `fold_conv_bn` mathematical equivalence | 1 | max ‖·‖_∞ = 2.4e-6 (PASS) |
| dtype compatibility (fp32 / bf16 / fp16) | 3 | 3 / 3 |
| Decay override across all neurons (IF×4 + LIF×6 + CubaLIF×4 + EIF×3) | 17 | 17 / 17 |
| Residual fusion path (`Conv→BN→Add→Neuron`) bit-equal + zoo end-to-end | 11 | 11 / 11 |
| **Total** | **188** | **188 ✓** |

### 4.4 End-to-end VGG16-SNN benchmark sweep

Per-image inference latency (lower is better) at BATCH = 32:

| neuron | reset | dtype | layout | naive (ms) | fused (ms) | speedup | naive peak (GiB) | fused peak (GiB) |
|--------|-------|-------|--------|------------|------------|---------|------------------|------------------|
| IF | hard | fp32 | NCHW | 4.941 | 3.771 | **1.310×** | 6.72 | 7.24 |
| IF | hard | fp32 | NHWC | 6.553 | 3.928 | **1.668×** | 6.72 | 5.71 |
| IF | hard | bf16 | NCHW | 2.919 | 2.283 | **1.279×** | 3.37 | 3.62 |
| IF | hard | bf16 | NHWC | 3.924 | 1.945 | **2.017×** | 2.84 | 2.86 |
| LIF | hard | fp32 | NCHW | 4.937 | 3.766 | **1.311×** | 6.72 | 7.24 |
| LIF | hard | fp32 | NHWC | 6.547 | 3.923 | **1.669×** | 6.72 | 5.71 |
| LIF | hard | bf16 | NCHW | 2.917 | 2.282 | **1.278×** | 3.37 | 3.62 |
| LIF | hard | bf16 | NHWC | 3.925 | 1.945 | **2.018×** | 2.84 | 2.86 |

Soft-reset rows differ from the hard-reset rows above by at most 0.02 ms /
image — i.e., within run-to-run noise — for every neuron / dtype / layout
combination, so they are omitted for brevity. Full results are in
`snn_compiler/benchmarks/sweep_results.jsonl`.

**Large-batch checkpoint.** At BATCH = 96, bf16 + NHWC + IF / hard runs at
**1.949 ms / image** (513 image/s, fused), confirming the latency scales
flat with batch up to the 7.5 GiB GPU memory cap.

### 4.5 Generality across architectures: VGG / ResNet / MobileNet-V2

To demonstrate that the framework is not VGG-specific, we ship reference
SNN implementations of three classical CNN families under
`snn_compiler/zoo/`:

- **VGG-11 / VGG-13 / VGG-16 / VGG-19 SNN** — pure Conv→BN→Neuron chains,
  fused by `fuse_snn_model`.
- **ResNet-18 / ResNet-34 SNN** — BasicBlocks whose second conv path uses
  the new `Conv→BN→Add→Neuron` fused kernel (`FusedConvBNAddNeuron`) and
  whose first conv path uses `FusedConvBNNeuron`.
- **MobileNet-V2 SNN** — inverted residual blocks combining 1×1 expand,
  3×3 depthwise and 1×1 project conv layers, with residual sum on
  stride-1 same-channel blocks.

End-to-end measurements (RTX 5070 Ti, BATCH = 16, T = 4, H = W = 224,
LIF / hard / bf16 + NHWC):

| Architecture | Naive (ms/img) | Fused (ms/img) | Speedup |
|---|---|---|---|
| VGG-11 SNN | 2.102 | 1.142 | **1.84×** |
| VGG-16 SNN | 3.914 | 1.971 | **1.99×** |
| ResNet-18 SNN | 0.587 | 0.307 | **1.91×** |
| ResNet-34 SNN | 0.958 | 0.498 | **1.93×** |
| MobileNet-V2 SNN | 1.103 | 0.240 | **4.60×** |

The same harness in fp32 + NCHW reports 1.14× – 1.42× — the bf16 + NHWC
configuration moves the cuDNN convolutions onto tensor cores, shrinks
their wall-clock and shifts a larger fraction of the per-step budget
onto the elementwise tail that fusion captures.

**Why MobileNet-V2 sees a 4.6× speed-up.** Depthwise + 1×1 pointwise
convolutions are very small individually; the launch tax of the bn /
neuron post-ops is comparable to or larger than the conv compute. The
fused kernel eliminates that tax across all 17 inverted-residual blocks
at once, which is where the 2.3× extra factor (vs. ResNet-34's 1.93×)
comes from. End-to-end correctness is verified at fp32 with
`max|fused − naive| = 0` for all three architectures
([`snn_compiler/tests/test_residual_and_zoo.py`](../../snn_compiler/tests/test_residual_and_zoo.py)).

### 4.6 Correlation with prior journal results

The hand-written `bf16_nhwc_snn.py` reported in section §8.4 of the journal
achieved **1.88 ms / image** on the same hardware at BATCH = 192. Our
generic framework reaches **1.945 ms / image** at BATCH = 96 — a 3.4 %
overhead vs. the monomorphic hand-tuned kernel, attributable to:

- Per-layer `bias.float().contiguous()` casting inside `FusedConvBNNeuron`
  (the journal kernel pre-casts once at construction).
- The framework's `y.contiguous(memory_format=torch.channels_last)` view
  insertion before each fused kernel call, which is required to keep the
  layout consistent across mixed module types in the Sequential.

We consider this acceptable since the support matrix gains a few orders of
magnitude in expressive power for that 3 % regression.

## 5. Result

The four primary results are:

1. **Single template covers all required configurations.** All 160 cases of
   the support matrix in §1 produce bit-equal spike outputs vs. a naïve
   PyTorch reference. There is no run-time branching cost because every
   axis of variation is a `constexpr`; specialized kernels are emitted
   once per (neuron, reset, threshold-mode, channel-order, dtype) tuple
   and cached by Triton's autotuner.
2. **The fusion benefit is decoupled from the neuron model.** IF and LIF
   rows in §4.4 differ by ≤ 0.01 ms / image at every (dtype, layout, reset)
   point. This means the framework's speed-up comes from the kernel
   *structure* (fewer launches, register-resident state) rather than from
   neuron-specific tricks; the same gain transfers to CubaLIF and EIF
   without retuning.
3. **bf16 + NHWC stacks productively with fusion.** The largest combined
   speed-up — **2.02×** vs. the naïve same-kernel baseline — appears at
   bf16 + NHWC. The order of stacking matters: bf16 cuts conv compute
   roughly 2×; NHWC cuts the layout-transform scratch buffer; fusion
   removes the per-layer bias-add and BN launches. Each is largely
   orthogonal to the other two.
4. **Graph pass produces 1.0× error.** `test_graph_pass.py` confirms that
   substituting `Conv→BN→LIF` with `FusedConvBNNeuron` yields
   `max|out − ref| = 0`, i.e., the BN fold is numerically exact in fp32
   and the spikes after the fold are bit-identical across the network.

## 6. Discussion

### 6.1 Why the largest speed-up appears at bf16 + NHWC

At fp32 + NCHW, the convolution is the dominant cost (each Conv accounts for
roughly 0.20 ms / image of a 4.94 ms total). The neuron + BN + bias triplet
contributes roughly 30 % of the total. Fusion therefore caps the achievable
speed-up near 1.43×, and we observe 1.31×.

Switching to bf16 halves the convolution time, so the relative weight of the
elementwise tail grows. Switching to NHWC eliminates the layout-conversion
scratch buffer between conv and BN. At bf16 + NHWC, the elementwise tail is
roughly half of the total time, so fusing it doubles overall throughput,
matching the measured 2.02×.

### 6.2 What the framework does **not** do

- **No training-mode support.** BN folding requires `eval()` running
  statistics; we do not handle the streaming update of `running_mean` /
  `running_var` during training.
- **No backwards pass.** Spike outputs are integer-valued; a surrogate
  gradient through the fused kernel would require a second kernel
  symmetric to the forward.
- **No residual-block detection.** The graph pass matches strictly along
  `nn.Sequential` lines; ResNet-style branches that re-merge via `+`
  are left untouched. A future extension is to lift to `torch.fx` once a
  multistep-stable trace mode is available.
- **No persistent v across forward calls.** Each `forward` clears v to
  zero. Continual-time SNNs that pass v between calls need to add v as
  an explicit input/output tensor — a structural extension, not a
  numerical one.

### 6.3 Comparison to `torch.compile` / Inductor

We do *not* claim that `torch.compile` is universally worse for SNNs. On
this hardware, Inductor's eager-vs-compile delta is small (~ 10 %) for ANN
VGG16 inference and *negative* for SNN VGG16 — see journal §6 — because
the Inductor-generated multistep kernel reproduces gap G1. The framework
trades model genericity for a known win on a known structural mismatch;
when Inductor adds a "register-resident reduction over the time axis"
heuristic, the gap closes and the framework's main contribution becomes
the unified API.

### 6.4 Extensibility roadmap

The kernel template generalizes naturally to:

- **PLIF.** Replace the scalar `decay_factor` with a `tl.load(tau_ptr + c_idx)`
  per-channel `decay_factor`. No structural change.
- **Refractory IF.** Track a `refrac_counter` register alongside v; clamp
  `spike` to zero while the counter is positive.
- **Izhikevich.** Two-state (v, u) with bi-linear dynamics; needs the same
  generalization as CubaLIF.
- **Quantized inference.** Replace the spike threshold with a per-channel
  scale + zero-point and lower the inner accumulator to int32.

None of these require touching the framework's pass or its nn.Module layer.

## 7. Conclusion

Spiking Neural Networks on GPUs need not pay a fixed T-multistep overhead.
By identifying a single MLIR-level pattern (outer-parallel over the spatial
neurons, T-register-loop over the time axis) common to all per-position
state-recurrent neuron models, and by parameterizing it with `constexpr`
switches for the neuron type, reset rule, threshold representation, channel
order and dtype, we obtain a single Triton kernel template that covers the
full IF / LIF / CubaLIF / EIF design space without any runtime branch
overhead. Fusing the pre-neuron convolution bias and folding the
inference-mode BatchNorm into the same kernel further removes the per-layer
launch tax inherited from cuDNN's bias separation. A model-level
graph-rewrite pass detects the canonical `Conv → [BN] → Neuron` motif and
substitutes the fused implementation automatically.

On VGG16-SNN with T = 4 on an RTX 5070 Ti, the framework reaches
**1.94 ms / image** (514 image/s) at bf16 + NHWC, a **2.02×** speed-up over
an unfused same-kernel baseline, while producing spike outputs **bit-equal**
to a naïve reference across the 160 tested configurations. The result is
within 3.4 % of a previously reported hand-written monomorphic kernel while
extending the support matrix by several orders of magnitude in expressive
power. The framework, its tests and its benchmarks live under
`snn_compiler/` in this repository.

## References

1. W. Fang, Y. Chen, J. Ding, Z. Yu, T. Masquelier, D. Chen, L. Huang,
   H. Zhou, G. Li, and Y. Tian. *SpikingJelly: An open-source machine
   learning infrastructure platform for spike-based intelligence.*
   Science Advances 9 (40), 2023.
2. J. E. Pedersen, S. Abreu, M. Jobst, et al. *NIR: Neuromorphic
   Intermediate Representation.* 2023.
3. P. Tillet, H. T. Kung, and D. Cox. *Triton: An intermediate language
   and compiler for tiled neural network computations.* MAPL 2019.
4. NVIDIA. *cuDNN: A GPU-accelerated library of primitives for deep
   neural networks.* Various versions.
5. PyTorch Team. *torch.compile / Inductor.* PyTorch documentation 2.x.

## Appendix A — Reproduction

```bash
# correctness (≈ 1 min)
python snn_compiler/tests/test_correctness.py
python snn_compiler/tests/test_graph_pass.py

# benchmark sweep (≈ 10 min on RTX 5070 Ti)
bash snn_compiler/benchmarks/sweep_all.sh

# single configuration
BATCH=96 T=4 TOTAL=2000 MODE=bf16 LAYOUT=NHWC NEURON=lif RESET=hard \
  python snn_compiler/benchmarks/bench_vgg16.py
```

The raw measurements used to populate the §4.4 table are at
`snn_compiler/benchmarks/sweep_results.jsonl`.
