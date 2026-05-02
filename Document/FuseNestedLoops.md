# FuseNestedLoops.cpp 代码分析

## 简要概述
`FuseNestedLoops.cpp` 实现了一个重要的循环优化 Pass——嵌套循环融合（Nested Loop Fusion）。其核心目的是将多层嵌套的循环（如 `for i in ...: for j in ...:`）展平（Flatten）为一个单层的超级循环。这一步骤主要为了配合后续的软件流水线（Software Pipelining）Pass，使流水线能够跨越原本不同的内层循环边界，最大化地掩盖全局内存延迟。

## 详细分析

### 1. 核心功能与目的
在矩阵乘法等密集计算场景中，代码往往包含多重循环（例如分别遍历 M 维和 N 维的 Tiles）。如果直接对内层循环进行流水线化，每次内层循环启动和排空（Prologue/Epilogue）时都会导致流水线气泡，降低 Tensor Core 的利用率。
通过将循环树融合为单一的 `scf.for` 循环，软件流水线只需要在整个大循环的开头和结尾经历一次启动和排空。

### 2. 主要数据结构与遍历
- **`LoopNest` & `LoopNestNode`**: 将函数中的 `scf.for` 构建成一棵树状数据结构。由于 Triton 限制了 CFG（控制流图）的出现，基本上只有结构化的 SCF（Structured Control Flow）操作，这种树状表达非常直观。
- **自底向上融合 (`fuseOneLevel`)**: 融合是按从叶子节点向根节点（Leaf-to-Root）的顺序，每次将一层子循环融合进父循环中。

### 3. 循环展平算法（Flattening）
假设父循环有 N 次迭代，内部有个执行 M 次的子循环，另外还有部分只在父循环里执行的代码（如每次进入内层前的 Prologue 和结束后的 Epilogue）。
展平算法并非简单相乘，而是引入了一个全局的迭代变量 `T`：
- 计算内层循环的总执行次数，结合可能外提（Hoist）的不变代码。
- 创建单一的大循环：`for _ in range(total_iters)`。
- 通过在新的大循环体内部插入大量的条件判断（`scf.if`）来重建原有的控制流：
  - 例如 `if T == 0`: 执行原外层循环的 Prologue 并初始化内层循环变量。
  - `if T >= 0 and T < len_j`: 执行内层循环的主体（Body），并更新内部迭代变量。
- 对于跨层的数据依赖（SSA Values），算法会将它们统统塞入大循环的 `iter_args` 中（可能初始赋值为 `ub.poison` 毒药值），从而将隐式捕获转化为显式的循环携带依赖。

### 4. 限制与注意事项
目前该算法只融合边界与父循环无关（Outer-Loop Invariant）的子循环。对于边界随父循环迭代变量变化的子循环（如三角矩阵遍历），处理起来会显著增加生成的 `scf.while` 复杂度，对流水线不友好，因此目前做了一定限制。

### 5. 关键代码段分析

```cpp
// FuseNestedLoops.cpp - LoopNest::print
void LoopNest::print(raw_ostream &os) const {
  // ...
  SmallVector<std::pair<LoopNestNode *, unsigned>> stack;
  stack.emplace_back(root, 0);
  while (!stack.empty()) {
    auto [node, indent] = stack.pop_back_val();

    // Print the current loop.
    os << std::string(indent * 2, ' ');
    printLoopFirstLine(node->loop);
    os << "\n";

    // Push the children of the current loop.
    for (LoopNestNode *child : node->children)
      stack.emplace_back(child, indent + 1);
  }
}
```
* **代码功能说明**: 这是用于遍历和打印循环嵌套结构的代码段。`LoopNest` 将函数中的所有嵌套循环构建为一棵树，这里使用了栈来进行前序遍历（Pre-order traversal），通过维护深度（`indent`）来可视化输出多层嵌套的信息。
* **原理解析**: Triton 编译器在执行嵌套融合时，并非盲目处理所有的 `for` 循环。由于 Triton GPU 中的控制流较为简单（严格受限于 `scf` 结构化控制流），它可以将代码中的嵌套关系精确地建为一棵树。这个类及其遍历方式就是后续自底向上（Bottom-Up）执行循环融合操作的基础设施。
* **在整个 PASS 中起到的作用**: 建立针对多维迭代（如处理多维 Blocked 张量的迭代逻辑）的内部抽象图，以供后续复杂的展平（Flatten）算法查询和修改控制流节点。
