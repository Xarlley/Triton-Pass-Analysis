# 审计「全 Triton 路径」是否真的完整跑通

> 配套文档：[full-triton-compilation.md](full-triton-compilation.md) 讲**怎么配置**让 SNN 走全 Triton；
> 本文讲**怎么验证**配置真的生效 —— 没有 cuDNN / cuBLAS extern fallback、没有 dynamo graph_break、
> 没有 SJ LIF kernel try/except 退回 Python loop。
>
> 适用场景：SNN Pass 开发上线前 / 切换 BATCH / 升级 torch / 升级 triton fork / 升级 spikingjelly
> 之后，对全 Triton 路径做一次端到端 sanity check。
>
> 验证环境（本文示例）：torch 2.11.0+cu130、spikingjelly 0.0.0.0.15、triton 3.7.0+gitef02d646
> （本仓库 fork）、RTX 5070 Ti。

---

## 1. 总体思路

「全 Triton 路径」= 把 SpikingJelly VGG16-SNN 整网（13 Conv + 13 BN + 15 LIF + 5 MaxPool + 3 FC）
通过 `torch.compile` 下沉到 Triton + Inductor，让：

1. **dynamo** 把整网编进**单一计算图**（`graph_break = 0`）；
2. **Inductor** 把所有 conv / gemm / pool / elementwise **全部代码生成为 Triton kernel**（不
   走 cuDNN / cuBLAS 的 `extern_kernels.*` fallback）；
3. **SpikingJelly 手写的 `_multistep_lif_forward_kernel`** 在 eval/CUDA/spiking-surrogate 三条满足
   时正常被 LIF dispatch 调用，**不触发** SJ 的 fallback 分支
   ([lif.py:582-595](../../spikingjelly/spikingjelly/activation_based/neuron/lif.py#L582-L595)) 退回 Python for-t-in-range(T) 循环。

任何一项失守，"自定义 Triton Pass 作用于整网"这条假设就不成立 —— 在 SNN Pass 开发上线前
必须有可重复的审计程序。本文列出 **10 个独立可观测指标 (A–J)** 与对应的 shell 命令。

---

## 2. 准备：run-and-capture

入口脚本两选一，区别只在 BATCH 与「是否做黄金比对」：

| 场景 | 命令 |
|---|---|
| 单次推理 BATCH=1 + 黄金比对 | `python examples/vgg16_snn/vgg16_test.py` |
| 任意 BATCH，100-iter 平均（不做比对） | `MODE=A BATCH=<N> python examples/vgg16_snn/benchmark_compare.py` |

两个脚本都内部调 `configure_full_triton_compilation()` + `patch_spikingjelly_for_full_graph()`，
配置等价。`vgg16_test.py` 跑得快、最后有现成的 `dynamo 图中断数: ...` 打印；`benchmark_compare.py`
省墙钟、且允许 `BATCH` 参数化。

把整次运行的 stderr + stdout **同时**抓到文件，再离线 grep：

```bash
# ⚠️ 重定向顺序很关键：先 > file 把 stdout 接到文件，再用 2>&1 把 stderr 合并到 stdout。
#    反过来写 `2>&1 > file` 只会把 stdout 写文件，stderr 仍然飘到终端 —— 错失全部 TORCH_LOGS。
TORCH_LOGS="output_code" python examples/vgg16_snn/vgg16_test.py \
    > /tmp/audit_b1.log 2>&1

# 或 BATCH=50：
TORCH_LOGS="output_code" MODE=A BATCH=50 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python examples/vgg16_snn/benchmark_compare.py \
    > /tmp/audit_b50.log 2>&1
```

`TORCH_LOGS="output_code"` 是 PyTorch 内置的日志开关，让 Inductor 把每个生成的 kernel
源码（含 `@triton.jit`, `def triton_...`, 启动 launcher 等）以日志形式 dump 出来。所有指标
都从该日志中过滤得到，**不**需要修改 PyTorch / SpikingJelly 源码或加 print。

每条日志行有 `[__output_code]` 前缀 + 一段时间戳 + 日志级别。这意味着你**不能**用
`^def triton_` 这种锚定行首的 grep —— 命中数会是 0。正确写法见下。日志文件量级：约 5000 行 / 800 KB
一次 run（BATCH=1 / BATCH=50 都差不多）。

---

## 3. 十个验证指标（A–J）与 grep 命令

下方一段 bash 把十项一次性跑完，针对 `/tmp/audit_b1.log` 或 `/tmp/audit_b50.log` 都适用：

```bash
LOG=/tmp/audit_b1.log    # 改成 b50.log 复测另一 BATCH

echo "=== A) extern_kernels.X 计数 (应为 0) ==="
grep -oE "extern_kernels\.\w+\(" "$LOG" | sort | uniq -c | sort -rn

echo "=== B) Triton kernel 定义数 ==="
grep -cE "def triton_|@triton\.jit" "$LOG"

echo "=== C) Triton kernel 启动调用数 ==="
grep -cE "\.run\(|_kernel_\d*\[.*grid|triton_.*\[grid" "$LOG"

echo "=== D) max_autotune 决策胜出者类型分布 ==="
grep -oE "\"best_kernel\":\s*\"[^_]+_[^_]+" "$LOG" | sort | uniq -c | sort -rn
echo "AUTOTUNE 总决策次数:"
grep -cE "\"best_kernel\":" "$LOG"

echo "=== E) cudnn / cublas 字样出现次数 (应为 0) ==="
grep -ciE "cudnn|cublas" "$LOG"

echo "=== F) Inductor 'Output code:' 段数 = 编译产出的 subgraph 数 (应为 1) ==="
grep -cE "Output code: $|\[__output_code\] Output code:" "$LOG"

echo "=== G) dynamo graph_break / Recompiling / Restarting analysis 字样 ==="
grep -ciE "graph_break|Restarting analysis|dynamo.*Recompiling|cant trace" "$LOG"

echo "=== H) SJ LIF kernel fallback 触发次数 (应为 0) ==="
grep -ciE "Falling back from Triton LIF" "$LOG"

echo "=== I) _multistep_lif_forward_kernel 引用次数 (>0 = SJ LIF kernel 在编译流程中) ==="
grep -cE "_multistep_lif_forward_kernel" "$LOG"

echo "=== J) (仅 vgg16_test.py) 黄金输出一致性 ==="
grep "黄金输出" "$LOG" | tail -1
echo "=== J') (vgg16_test.py 独有) dynamo 图中断数 ==="
grep "图中断数" "$LOG" | tail -1
```

### 每项的含义与判读标准

| 指标 | 命令骨架 | 通过条件 | 失败含义 |
|---|---|---|---|
| **A** | `grep extern_kernels\.\w+\(` | **0** | Inductor 把某 conv/gemm 降级为 cuDNN/cuBLAS extern。说明 `max_autotune_{conv,gemm}_backends="TRITON"` 没生效，或某个形状下 Triton 模板找不到能跑的 cfg。本仓库目标网络的预期值是 0。|
| **B** | `grep "def triton_\|@triton.jit"` | 远大于 0（VGG16 实测 54）| 0 说明编译流水线根本没生成 Triton kernel —— 可能 dynamo trace 阶段就 fail / fallback。|
| **C** | `grep ".run(\|kernel\[grid"` | 远大于 0（VGG16 实测 86）| 0 说明生成了 kernel 但没启动。理论上不会发生，除非编译产出和 launcher 解耦。|
| **D** | `grep "best_kernel"` | 全是 `triton_convolution2d_*` / `triton_mm_*` / `triton_addmm_*`，**没有** `extern_*` | 出现 `extern_*` 等同 A 失败：Inductor 在 autotune 阶段把某种后端选成了非 Triton。|
| **E** | `grep -i "cudnn\|cublas"` | **0** | 编译产物里只要出现 cudnn/cublas 字样（除非是 import 或注释），就有可能 fallback。本目标 0。|
| **F** | `grep "Output code:"` | **1** | >1 说明 dynamo 把网络切成多段 subgraph 各自编译 → 有 graph_break。1 = 单一编译图。|
| **G** | `grep "graph_break\|Restarting analysis\|Recompiling"` | **0** | 任何匹配都说明 dynamo 至少一次中断 trace、把代码踢回 Python interpreter 跑 → 网络有部分走 eager。|
| **H** | `grep "Falling back from Triton LIF"` | **0** | SJ 在 [lif.py:583](../../spikingjelly/spikingjelly/activation_based/neuron/lif.py#L583) 处打的 debug log。出现 >0 说明 SJ multistep_lif kernel 抛了预期异常退回 Python loop —— 通常是 surrogate / dtype 不支持，本仓库 patch 后预期 0。|
| **I** | `grep "_multistep_lif_forward_kernel"` | **>0**（VGG16 实测 33） | 0 说明 LIF 完全没走过 SJ 手写 kernel。可能 1. 不在 eval 模式 / 2. 不在 CUDA / 3. surrogate_function 的 `spiking=False`。|
| **J** | `grep "黄金输出"` | **逐位一致** 或 **容差内一致** | 出 `❌ 超出容差` 说明编译路径产生了数值错误的输出，整个审计废。|
| **J'** | `grep "图中断数"` | `图中断数: 0` | 与 F、G 三者互为佐证。|

### `TORCH_LOGS="output_code"` 之外可以加的开关

调试更细粒度时，把环境变量改成：

```bash
TORCH_LOGS="dynamo,inductor,output_code,graph_breaks" python ...
```

- `dynamo`：每次进出 dynamo 的事件
- `graph_breaks`：每个 graph_break 的具体行号和原因（最有用）
- `inductor`：Inductor 调度细节
- `output_code`：Inductor 生成的 kernel 源码（本文用的这个）

日志量级会涨到 5 MB+，grep 路径不变。

---

## 4. 实测案例：BATCH=1 + BATCH=50 完整通过

把上面那段一次性脚本对两个 BATCH 的日志各跑一遍：

| 指标 | BATCH=1 (vgg16_test.py) | BATCH=50 (benchmark_compare.py) | 通过? |
|---|---:|---:|---|
| A) extern_kernels.* | 0 | 0 | ✅ |
| B) Triton kernel 定义数 | 54 | 54 | ✅ |
| C) kernel 启动调用数 | 86 | 86 | ✅ |
| D) autotune 决策 conv/mm | 9 + 3 = 12 (全 Triton) | 9 + 3 = 12 (全 Triton) | ✅ |
| E) cudnn/cublas 字样次数 | 0 | 0 | ✅ |
| F) Output code 段数 | 1 | 1 | ✅ |
| G) graph_break / Recompiling | 0 | 0 | ✅ |
| H) LIF fallback 触发 | 0 | 0 | ✅ |
| I) `_multistep_lif_forward_kernel` 引用 | 33 | 33 | ✅ |
| J) 黄金输出 | ✅ 逐位一致 | (脚本不做比对) | n/a |

### 跨 BATCH 的不变量观察

- conv autotune 都是 **9** 次：VGG16-D 里有 9 个独特 `(in_ch, out_ch, kernel, stride, padding)`
  组合（conv4_2/4_3 同 shape 共用；conv5_2/5_3 同 shape 共用）。
- gemm autotune 都是 **3** 次：3 个全连接层 weight shape 都不同。
- kernel 定义/调用数固定 54/86：编译产物的结构与 BATCH 无关。

BATCH 变化只影响 autotune cache key 中的 M 维（= T·B），即每个 BATCH 都要重新 autotune
一次（首次 50–120 s 编译开销），但**最终走的 kernel 仍全是 Triton**。

---

## 5. 常见 failed-audit 的根因速查

| 现象 | 看哪一项 | 常见根因 |
|---|---|---|
| **A > 0** | Inductor 把某 conv/gemm 降级 extern | `configure_full_triton_compilation` 没调；或 `inductor.config.max_autotune_{conv,gemm}_backends` 在被调之后又被别处覆盖；或某个 conv 形状所有 Triton cfg 都 OOM 被全部剔除后被迫回退 extern。|
| **F > 1 / G > 0** | dynamo 把网络切成多段 | `patch_spikingjelly_for_full_graph()` 没调，BN 处仍触发 isinstance graph break；或 `recompile_limit` 触顶（默认 8 太小），dynamo 把某帧标记为 skip 后回 eager。|
| **H > 0** | LIF kernel fallback | SJ multistep_lif 抛了预期异常：通常是输入 dtype 不支持（fp16/bf16/int），或 surrogate `spiking=False`，或上游 triton API 微小变动 SJ kernel 内部 `convert_and_store` 等不兼容（→ 看 [SpikingJelly-Triton-Patch.md](../../examples/vgg16_snn/SpikingJelly-Triton-Patch.md) §3）。|
| **B = 0 / I = 0** | 整个编译流水线没启动 | `torch.compile(model)` 没包；或推理时绕过了 wrapper（直接调原 model 而不是 compiled）。|
| **D 里出现 `extern_*` / cuDNN / cuBLAS** | autotune 选了 extern 后端 | conv/gemm backend 配置未生效。`grep "max_autotune_conv_backends" /home/charlley/.../torch/_inductor/config.py` 看实际默认值是不是被覆盖。|
| **J 显示 ❌ 超出容差** | 编译路径数值错 | 某个 conv autotune 选了配置错误的 kernel；或自定义 SNN Pass 引入了非等价变换；或某个 Triton kernel 编进了错的 vector layout（先排查最近的 Pass diff，再排查 Triton fork 的 commit）。|

---

## 6. 完整可复用脚本

下面这段 bash 适合塞进 `examples/vgg16_snn/audit_triton_path.sh`（如果想固化的话），跑完打印
PASS/FAIL：

```bash
#!/usr/bin/env bash
# audit_triton_path.sh — 一键审计「全 Triton 路径」
set -eo pipefail

cd "$(dirname "$0")/../.."   # 退到仓库根
LOG=/tmp/triton_audit_$(date +%s).log

if [[ -n "${BATCH:-}" && "$BATCH" != "1" ]]; then
    TORCH_LOGS="output_code" MODE=A BATCH=$BATCH \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python examples/vgg16_snn/benchmark_compare.py > "$LOG" 2>&1
    HAS_GOLDEN=0
else
    TORCH_LOGS="output_code" \
        python examples/vgg16_snn/vgg16_test.py > "$LOG" 2>&1
    HAS_GOLDEN=1
fi

declare -i FAIL=0
check() {
    local name="$1" expected="$2" actual="$3"
    if [[ "$actual" == "$expected" ]]; then
        printf "  [PASS] %-40s %s\n" "$name" "$actual"
    else
        printf "  [FAIL] %-40s expected=%s actual=%s\n" "$name" "$expected" "$actual"
        FAIL+=1
    fi
}

echo "=== Triton path audit  log=$LOG  BATCH=${BATCH:-1} ==="
check "A) extern_kernels.X"      0  "$(grep -cE 'extern_kernels\.\w+\(' "$LOG" || true)"
check "E) cudnn/cublas mentions" 0  "$(grep -ciE 'cudnn|cublas' "$LOG" || true)"
check "F) Inductor output_code segments" 1 \
    "$(grep -cE 'Output code: $|\[__output_code\] Output code:' "$LOG" || true)"
check "G) dynamo graph_break"    0  "$(grep -ciE 'graph_break|Restarting analysis|dynamo.*Recompiling' "$LOG" || true)"
check "H) LIF kernel fallback"   0  "$(grep -ciE 'Falling back from Triton LIF' "$LOG" || true)"

triton_def=$(grep -cE 'def triton_|@triton\.jit' "$LOG" || true)
[[ "$triton_def" -gt 30 ]] \
    && printf "  [PASS] %-40s %s\n" "B) Triton kernel defs" "$triton_def" \
    || { printf "  [FAIL] %-40s expected>30 actual=%s\n" "B)" "$triton_def"; FAIL+=1; }

lif_refs=$(grep -cE '_multistep_lif_forward_kernel' "$LOG" || true)
[[ "$lif_refs" -gt 0 ]] \
    && printf "  [PASS] %-40s %s\n" "I) SJ LIF kernel referenced" "$lif_refs" \
    || { printf "  [FAIL] %-40s expected>0 actual=%s\n" "I)" "$lif_refs"; FAIL+=1; }

if [[ "$HAS_GOLDEN" == "1" ]]; then
    grep -q "黄金输出逐位一致" "$LOG" \
        && printf "  [PASS] %-40s %s\n" "J) golden output match" "bit-identical" \
        || { printf "  [FAIL] %-40s\n" "J) golden output mismatch"; FAIL+=1; }
fi

if (( FAIL == 0 )); then
    echo; echo "=== ALL CHECKS PASSED ==="
else
    echo; echo "=== $FAIL CHECKS FAILED.  Full log: $LOG ==="
    exit 1
fi
```

用法：

```bash
bash audit_triton_path.sh              # BATCH=1
BATCH=50 bash audit_triton_path.sh     # BATCH=50
BATCH=56 bash audit_triton_path.sh     # 边界 BATCH，会触发 50–120s 重新 autotune
```

---

## 7. 该审计**不**保证的几件事

明确告知边界：

- **不**做性能比对。9.34 ms/张 还是 12 ms/张，要靠 [benchmark_inference.py](../../examples/vgg16_snn/benchmark_inference.py)
  或 [benchmark_compare.py](../../examples/vgg16_snn/benchmark_compare.py) 单独测。审计只判"路径是否完整"，不判"路径是否够快"。
- **不**保证 Triton kernel 的算术正确性。Inductor 生成的 kernel 是否数值正确，依赖 J) 黄金输出
  比对 —— 而 J 只在 `vgg16_test.py` 路径里有，`benchmark_compare.py` 用的是随机权重。若改了 SNN
  Pass 或 Triton fork commit，应该用 `vgg16_test.py` 跑一次拿到 J ✅。
- **不**覆盖 fp16 / bf16 / int8 路径。当前 VGG16-SNN 全 fp32；切换 dtype 后 SJ LIF kernel 可能
  落到 fallback（H 项会触发），需要单独跑一次审计。
- **不**保证别的 SNN 网络（Spikformer、SEW-ResNet 等）的全 Triton 路径。本审计基于 VGG16-SNN
  的具体结构（13 Conv + 3 FC，无 attention，无 residual），常量 54/86/9+3 都是 VGG16 实测值；
  换模型 B/C/D 三项的数字会变，其余指标 (A/E/F/G/H/I/J) 的"应为 0/应为 1"判断仍适用。
