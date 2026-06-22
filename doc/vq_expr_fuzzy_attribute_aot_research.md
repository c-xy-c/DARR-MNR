# VQ-Expr 收敛方案：Fuzzy Quantity Attributes、AoT 表达式与 Negative Choice 生成

日期：2026-06-21

本文专门回应当前三件事：

```text
1. 明确 4-5 种适合 symbolic reasoning 的 fuzzy quantity attributes。v1 主线改为 1-9；1-99 只作为扩展版。
2. 明确数学表示的底层 rule：如何用 AoT 生成 expression 和另外 7 个 negative choices。
3. 调研 fuzzy、symbolic、visual reasoning 和相关领域，说明我们借鉴什么、不借鉴什么。
```

核心修正：

```text
不要从 inner/boundary/outlier 出发。
先定义可复算的 fuzzy 单位数量属性，再用 AoT 生成数学表达式，最后用 visual rules 把 AoT 渲染成图像。
```

## 0. 一句话结论

我们建议把数据集正式收敛为：

```text
VQ-Expr: Fuzzy Visual Quantity Expression Reasoning
```

它不是 “MNR + fuzzy decoration”，而是：

```text
quantity object 的数值由 fuzzy visual attributes 表示；
expression 由 Arithmetic Operation Tree (AoT) 生成；
visual rules 把 AoT 渲染成无阿拉伯数字、无 + - * / = 的图像；
8-way choices 由 AoT-level counterfactual operators 生成。
```

## 1. 研究对象：不是 fuzzy image，而是 fuzzy attribute

### 1.1 什么叫符合 fuzzy 的 attribute

这里的 fuzzy 不是把图像画得模糊，也不是让答案不确定，而是：

```text
一个视觉属性对应一个 fuzzy set over possible numbers。
```

对每个 quantity object `q`：

```text
mu_q(n) in [0, 1], n in {1, ..., 9}
```

其中：

```text
numeric_value(q) = argmax_n mu_q(n)
```

但模型不能直接看到 `numeric_value`，只能从视觉属性恢复 `mu_q`。这继承 Zadeh fuzzy set 的基本思想：
对象对集合的隶属度是连续值，而不是二值归属。

### 1.2 Fuzzy attribute 的验收标准

一个 attribute family 能进入 VQ-Expr，必须满足：

| 标准 | 要求 |
| --- | --- |
| Coverage | v1 必须稳定表示 1-9；1-99 只作为后续 place-value 扩展，不作为第一版硬目标。 |
| Symbolic decomposability | 可分解成可复算 components，例如 tens/ones、length units、cells、path edges。 |
| Fuzzy membership | 能自然产生 `mu_q(n)`，而不是人为给一个 noise label。 |
| Visual plausibility | 人类能从图像理解该属性确实与数量有关。 |
| Confound audit | 面积、密度、周长、颜色、primitive count 不能单独决定 label。 |
| Cross-family exchange | 同一个 `numeric_value` 能用不同 attribute family 表示，支持 Family-OOD。 |

## 2. 五种建议 attribute

我建议第一版保留 5 种，全部围绕 1-9 做稳。这样更符合无阿拉伯数字视觉表示的可读性，也避免把任务做成“数 99 个视觉元素”。颜色可以作为其中一种 family，但必须配合 fuzzy color membership，不能变成任意 lookup table。

### 2.1 Attribute A: Grouped Units

**v1 表示范围：** 1-9。  
**扩展版：** 可以通过 tens/ones 组合扩展到 1-99，但 v1 不把它作为主目标。

视觉规则：

```text
n = number of grouped unit elements
```

适合原因：

```text
1. 它是最稳的 grouped cardinality 表示。
2. 1-9 范围内不需要 place-value，也不需要大量散点。
3. 可自然加入 ghost unit，形成 n 与 n+1 的 fuzzy 竞争。
```

Fuzzy 形式：

```text
mu_q(n) = membership that the visible group contains n unit elements
```

Fuzzy 来源：

```text
unit overlap；
partial occlusion；
group boundary ambiguity；
ghost unit 造成 n 与 n+1 的竞争解释。
```

风险：

```text
如果所有 family 都是 grouped units，它会退化成普通计数任务。
```

控制：

```text
只作为一种 family；
训练/测试跨 family；
同一个值可以换成 length/area/path/color family。
```

### 2.2 Attribute B: Rod / Length Train

**v1 表示范围：** 1-9。  
**扩展版：** 多 rod train 或 scale segment 可扩展到 1-99。

视觉规则：

```text
n = unit_length(rod), or sum_i unit_length(rod_i), n in 1..9
```

适合原因：

```text
1. 连续视觉长度与离散数值之间有天然 fuzzy 边界。
2. 适合测试 scale、endpoint uncertainty、number-line style reasoning。
3. 1-9 范围内读数稳定，适合 fuzzy endpoint。
```

Fuzzy 形式：

```text
mu_len_i(k) = membership that rod_i has length k
mu_q(n) = max_{sum_i k_i=n} product_i mu_len_i(k_i)
```

Fuzzy 来源：

```text
端点不清；
tick mark 缺失；
rod overlap；
scale jitter；
透视变形。
```

### 2.3 Attribute C: Area / Tile Patch

**v1 表示范围：** 1-9。  
**扩展版：** rows * columns 可以扩展到 1-99，但 v1 先用小 patch。

视觉规则：

```text
n = number of filled unit cells, n in 1..9
```

或 factorized：

```text
n = rows * columns - holes
```

适合原因：

```text
1. 面积天然连接 multiplication/division。
2. 小 patch 易读，适合表达 fuzzy boundary cell。
3. 对 `*` 和 `/` 的 visual operator 很自然。
```

Fuzzy 形式：

```text
mu_cell_j(filled) in [0,1]
mu_q(n) = membership that exactly n cells are filled
```

实际实现可用近似：

```text
mean_count = sum_j mu_cell_j
sigma = boundary_uncertainty + occlusion_uncertainty
mu_q(n) = exp(-(n - mean_count)^2 / (2 sigma^2))
```

Fuzzy 来源：

```text
partial cell；
边界模糊；
holes；
grid alignment ambiguity。
```

### 2.4 Attribute D: Graph Path / Topological Count

**v1 表示范围：** 1-9。  
**扩展版：** chunked path 可扩展到 1-99。

视觉规则：

```text
n = selected path length, n in 1..9
```

或：

```text
n = number of selected nodes / edges / route segments
```

适合原因：

```text
1. 它把 quantity 和 topology 绑定，适合测试 long-range visual binding。
2. 可以与 visual expression graph 形成双层结构：path-as-value 与 graph-as-program。
3. 很适合长度 OOD 和 branch ambiguity。
```

Fuzzy 形式：

```text
mu_edge(e selected)
mu_q(n) = membership that selected route has length n
```

Fuzzy 来源：

```text
branch ambiguity；
edge gap；
path crossing；
node occlusion；
near-overlapping route。
```

控制：

```text
v1 限制 path 长度 1-9，避免拥挤；
扩展版可使用 chunked path；
metadata 保存 exact selected path。
```

### 2.5 Attribute E: Color-Lattice

**v1 表示范围：** 1-9。  
**扩展版：** 如果要 1-99，必须组合成 place-value；v1 不做。

视觉规则：

```text
hue cluster -> value in 1..9
```

为什么保留颜色：

```text
1. 颜色天然适合 fuzzy membership，例如红/橙边界、蓝/紫边界。
2. 可以作为 attribute ambiguity 的强来源。
3. 它能测试模型是否把 perceptual fuzzy attribute 传播到 symbolic value。
```

为什么不能把颜色作为唯一主 family：

```text
颜色到数字很容易变成人工密码；
缺少数学教具或数量感基础；
reviewer 可能认为它是 lookup-table benchmark。
```

推荐用法：

```text
Color-Lattice 用作 auxiliary family、Style/Fuzzy-OOD family，或与 grouped/rod/area 混合。
不要让颜色单独承担整个 number system。
```

Fuzzy 形式：

```text
mu_q(n) = membership of color to fuzzy color sector n, n in 1..9
```

## 3. 第一版 attribute 选择

建议 v1 用这 5 个：

| Family | Range | 是否主 family | Fuzzy 机制 | 推荐用途 |
| --- | --- | --- | --- | --- |
| Grouped Units | 1-9 | 主 | group membership / ghost unit | cardinality grounding |
| Rod-Length Train | 1-9 | 主 | endpoint/tick/scale uncertainty | length/value grounding |
| Area-Tile Patch | 1-9 | 主 | cell occupancy/boundary uncertainty | multiplication/division |
| Graph-Path Count | 1-9 | 主 | route/edge selection membership | long-range binding |
| Color-Lattice | 1-9 | 辅助或主的 fuzzy split | hue/saturation membership | fuzzy perception, OOD |

暂时不要把 raw dot array 作为主 family。它可以作为 stress split，但不适合承载 symbolic expression，因为面积、密度、凸包等 confounds 太强。

### 3.1 为什么 v1 改成 1-9

这是一个必要 pivot：

```text
1-99 的视觉数量表示太重，会把任务变成视觉计数/教具识别；
1-9 更适合做 clean fuzzy attribute；
复杂性改由 AoT tree depth、non-commutative binding、scope、fuzzy trap 提供。
```

这意味着：

```text
MNR 中依赖大整数范围、复杂 number sampling、carry/borrow 的规则不能照搬。
```

但这不是削弱任务，而是换研究对象：

```text
旧复杂性: 大数字 + 算式模板
新复杂性: 小 fuzzy quantity + visual rule induced AoT
```

## 4. 数学底层 rule：AoT 而不是文本 expression

### 4.1 从现有 MNR 继承什么，以及彻底放弃什么

仓库中有两套东西要分开看。第一套是可以继承的“表达式树/反向采样”思想；第二套是必须放弃的
`holistic / analytical` 问题氛围。

可以继承：

```text
mnr_dataset/Calculator_Tree.py:
  用 calculator tree 表示 arithmetic expression。
  build_calculator_tree(postfix_expression) 构建二叉运算树。
  number_sampler(root_value) 从根值反向采样叶子，保证 + - * / 的整数约束。
```

必须放弃：

```text
mnr_dataset/AOT.py 的 Problem/Compartment/Condition 语义
mnr_dataset/Mathematics.py 的 Interpret = holistic / analytical
mnr_dataset/Mathematics.py 的 Analytical parts
mnr_dataset/Num_Arrange.py 的 holistic_parser / analytical_parser
```

原因：

```text
holistic / analytical 是 MNR 原来的 number arrangement 氛围；
它假设可见数字已经存在，重点是整体/局部分组解释；
VQ-Expr 的问题不是整体看还是局部看，而是：
  fuzzy visual attribute 如何变成 latent number，
  visual binding 如何生成 AoT，
  fuzzy uncertainty 如何沿 AoT 传播。
```

因此 VQ-Expr 不应复用旧 visible digit pipeline，也不应照搬旧 MNR 的 holistic/analytical rule；只能继承 calculator tree 的思想：

```text
先采样 AoT / expression tree；
再反向采样 leaf numeric values；
最后把每个 leaf value 渲染成 fuzzy quantity attribute。
```

### 4.2 哪些 MNR rule 不能用了

改成 1-9 后，下面这些 MNR 原 rule 不再适合作为主机制：

| MNR 原机制 | 为什么不适合 VQ-Expr v1 | 替代机制 |
| --- | --- | --- |
| 大范围 integer sampling | 1-9 不再依赖大值域制造难度。 | AoT depth / tree shape / operator composition。 |
| carry/borrow 类规则 | 1-9 下 carry/borrow 不自然。 | Sub/Div 的非交换 port-binding。 |
| 纯表达式模板匹配 | 会退回 MNR，只是把数字换成图形。 | visual rule compiler 先生成 AoT，再执行表达式。 |
| 随机 wrong answers | 容易产生 shortcut 或重复答案。 | 7 类 counterfactual negatives。 |
| number-only difficulty | 没有显式数字，且数值范围小。 | fuzzy attribute + binding + scope 共同决定答案。 |
| holistic / analytical interpretation | 它是旧布局氛围，不是 fuzzy visual-program rule。 | Attribute-Binding-Scope rule taxonomy。 |

因此 v1 的 rule bank 应该是新的：

```text
R1 Attribute Rule:
  每个 leaf 是 1-9 的 fuzzy quantity object。

R2 Operator Rule:
  每个 internal node 是视觉门，对应 Add/Sub/Mul/Div 或 v1 的 typed variant。

R3 Binding Rule:
  visual edges/ports 决定 children 顺序，尤其影响 Sub/Div。

R4 Composition Rule:
  layout/scope/nesting 决定 AoT tree，不由文本括号给出。

R5 Fuzzy Truth Rule:
  leaf fuzzy value + edge membership + gate membership 沿 AoT 传播。

R6 Candidate Rule:
  1 correct + 7 counterfactual negatives，每个 negative 只破坏一个主 rule。
```

### 4.2.1 新 rule taxonomy：替代 hol / ana

旧的 hol/ana 不能用了。VQ-Expr 的 rule taxonomy 应该直接围绕“fuzzy attribute -> AoT”：

| Rule name | 作用 | 对应失败模式 |
| --- | --- | --- |
| Attribute Decode Rule | 从视觉属性得到 `mu_q(n)`，n in 1..9。 | 模型只看外观，不恢复数量分布。 |
| Gate Semantics Rule | 从视觉门形态得到 `Add/Sub/Mul/Div` 或 typed gate。 | 模型不识别 operator。 |
| Port Binding Rule | 连线接入 left/right/input port，决定 child order。 | 模型忽略非交换运算的输入顺序。 |
| Scope / Tree Rule | 嵌套、flow、lane、container 决定 AoT tree shape。 | 模型只看 operator multiset，不看 tree。 |
| Fuzzy Propagation Rule | `mu_q`、edge membership、gate membership 沿 AoT 传播。 | 模型 harden 成单一 parse，错过 fuzzy truth。 |
| Candidate Consistency Rule | candidate 必须匹配 context 的 visual rule + AoT output。 | candidate-only 或 number-only shortcut。 |

这套 taxonomy 才是新数据集的核心，不再出现：

```text
holistic
analytical
analytical_part
```

### 4.3 VQ-Expr AoT 定义

VQ-Expr 的 AoT 包含四层：

```text
Root
  -> ExpressionSeed
      -> ArithmeticTree
      -> QuantityAttributePlan
      -> VisualRulePlan
      -> CandidatePlan
```

更具体：

```json
{
  "aot": {
    "expression_tree": {
      "root": "g4",
      "nodes": [
        {"id": "q1", "type": "leaf", "domain": "quantity"},
        {"id": "q2", "type": "leaf", "domain": "quantity"},
        {"id": "g1", "type": "op", "op": "Add", "children": ["q1", "q2"]},
        {"id": "g2", "type": "op", "op": "Mul", "children": ["g1", "q3"]},
        {"id": "g3", "type": "op", "op": "Sub", "children": ["q4", "q5"]},
        {"id": "g4", "type": "op", "op": "Div", "children": ["g2", "g3"]}
      ]
    },
    "quantity_attributes": {
      "q1": "grouped_units",
      "q2": "rod_length_train",
      "q3": "color_lattice",
      "q4": "area_tile",
      "q5": "graph_path_count"
    },
    "visual_rules": {
      "family": "circuit_flow",
      "binding": "port_attachment",
      "order": "directed_dag"
    }
  }
}
```

### 4.4 Arithmetic grammar

```text
E ::= Q
    | Add(E, E)
    | Sub(E, E)
    | Mul(E, E)
    | Div(E, E)
```

生成约束：

```text
leaf values in 1..9
candidate output values in 1..9
intermediate values may exceed 9, but final answer must return to 1..9
Sub(a,b): require a > b, unless using AbsDiff explicitly
Div(a,b): require b != 0 and a % b == 0
avoid degenerate operations:
  +0, -0, *1, /1, a-a, a/a
must include at least:
  one non-commutative operator: Sub or Div
  one composition depth >= 3
  one mixed attribute-family leaf
```

### 4.5 v1 rule templates

1-9 下不建议一开始就开放任意表达式树。应先用可控 rule templates，逐步增加组合复杂度。

| Rule family | Expression schema | 约束 | 视觉压力点 |
| --- | --- | --- | --- |
| Pair-Compose | `Op2(Op1(a,b), c)` | final in 1..9 | operator + binding |
| Balance | `Op1(a,b) = Op2(c,d)` | 两边结果相等 | cross-panel/candidate consistency |
| Inverse-Chain | `Div(Mul(Add(a,b), c), d)` | exact division, final 1..9 | long AoT + Div port |
| Scope-Contrast | `Mul(Add(a,b),c)` vs `Add(a,Mul(b,c))` | 两个结果不同且 1..9 | layout/scope |
| Fuzzy-Trap | hard parse answer equals candidate, fuzzy score lower | margin valid | fuzzy membership propagation |

第一版推荐主 rule：

```text
Inverse-Chain:
  y = Div(Mul(Add(a,b), c), Sub(d,e))

constraints:
  a,b,c,d,e in 1..9
  Add(a,b) in 2..18
  Mul(Add(a,b), c) may exceed 9
  Sub(d,e) in 2..9
  numerator % denominator == 0
  y in 1..9
```

这保留了长 expression，但 leaf/candidate 都是 1-9，可视化更稳。

### 4.6 Reverse sampling rule

和 `Calculator_Tree.number_sampler` 类似，VQ-Expr 从目标答案 `r` 反向采样子节点。

对每个 internal node：

```text
Add:
  choose a in [1, r-1]
  b = r - a

Sub:
  choose b in [1, 9-r]
  a = r + b

Mul:
  choose factor a of r
  b = r / a

Div:
  choose b in [2, floor(M/r)] where M is intermediate bound
  a = r * b
```

注意：这里的 `a,b` 可以是 internal child result，不一定是 leaf。v1 的 leaf 必须在 1..9；
internal value 可以短暂超过 9，但最终 answer 必须回到 1..9。

生成器必须拒绝：

```text
重复候选答案；
除法非整数；
负数或 0 结果；
所有 leaves 太小导致表达式 trivial；
不同 counterfactual 得到同一个 answer value。
```

### 4.7 Crisp oracle 与 fuzzy oracle

VQ-Expr 必须同时有两个 oracle：

**Crisp oracle：**

```text
read numeric_value(q)
execute AoT exactly
produce deterministic correct answer
```

**Fuzzy oracle：**

```text
read mu_q(n)
propagate fuzzy values through AoT
multiply by gate/edge visual memberships
produce candidate truth score
```

Fuzzy arithmetic 用 extension principle：

```text
mu_{A+B}(z) = max_{x+y=z} Tnorm(mu_A(x), mu_B(y))
mu_{A-B}(z) = max_{x-y=z} Tnorm(mu_A(x), mu_B(y))
mu_{A*B}(z) = max_{x*y=z} Tnorm(mu_A(x), mu_B(y))
mu_{A/B}(z) = max_{x=y*z} Tnorm(mu_A(x), mu_B(y))
```

最终 candidate score：

```text
score(candidate c) =
  Tnorm(
    mu_root(value(c)),
    all selected edge memberships,
    all gate-type memberships,
    all quantity decoding confidences
  )
```

标签仍然 deterministic：

```text
correct = argmax score(c)
valid iff score(correct) - max_negative_score >= margin
```

## 5. 8-way choices：1 correct + 7 negatives

每题有 8 个 choices。第 0 个是 correct，另外 7 个不是随机错数，而是 AoT/visual-rule counterfactual。

### 5.1 Correct choice

```text
C0 = render_quantity(answer_value, answer_attribute_family)
```

`answer_value` 来自 crisp AoT execution。answer family 可以和 leaves 同 family，也可以随机换 family，以防候选只靠样式匹配。

### 5.2 Negative 1: Output Value Perturbation

改变：

```text
只改变 answer candidate 的 quantity attribute。
```

生成：

```text
v1 = answer_value +/- delta
delta in {1,2,10}
v1 in 1..99
```

目的：

```text
测最终数值识别和近邻 fuzzy confusion。
```

### 5.3 Negative 2: Leaf Attribute Perturbation

改变：

```text
选一个 leaf q_i，把其 visual attribute 改成邻近值 q_i'
AoT structure 不变。
```

生成：

```text
q_i' = q_i +/- 1 or +/- 10
recompute AoT result v2
render answer candidate v2
```

目的：

```text
测模型是否真正从 leaf visual attribute 读取数值。
```

### 5.4 Negative 3: Operator Swap

改变：

```text
选一个 internal node，把 op 改成另一个合法 op。
```

例子：

```text
Add -> Sub
Mul -> Div
```

约束：

```text
新表达式必须合法；
输出在 1..99；
不等于 correct answer。
```

目的：

```text
测模型是否识别视觉门的 operator semantics。
```

### 5.5 Negative 4: Port Swap

改变：

```text
对 Sub 或 Div 节点交换 left/right port。
```

例子：

```text
Sub(a,b) -> Sub(b,a)
Div(a,b) -> Div(b,a)
```

约束：

```text
结果合法且不同。
```

目的：

```text
测视觉 port-binding，尤其是非交换运算。
```

### 5.6 Negative 5: Binding Rewire

改变：

```text
交换两条 incoming binding edges，但不改变 gate type 和 leaf values。
```

例子：

```text
Mul(Add(q1,q2), q3)
-> Mul(Add(q1,q3), q2)
```

目的：

```text
测 visual graph binding，而不是 quantity multiset shortcut。
```

### 5.7 Negative 6: Scope / Tree Rotation

改变：

```text
改变 AoT tree structure，但保留 operator multiset 和 leaf multiset。
```

例子：

```text
Mul(Add(q1,q2), q3)
-> Add(q1, Mul(q2,q3))
```

目的：

```text
测 layout-to-execution-order / precedence / nesting visual rule。
```

### 5.8 Negative 7: Fuzzy Ambiguity Trap

改变：

```text
选择一个在 hard parse 下看起来合理、但 fuzzy truth 低的 candidate。
```

生成：

```text
candidate value v7 lies in support(mu_root), but membership is below correct by margin;
or candidate uses a low-confidence alternative binding edge.
```

目的：

```text
测模型是否传播 fuzzy membership，而不是 harden 成 argmax parse。
```

这是最关键的 fuzzy negative。没有它，数据集只是 non-symbolic arithmetic circuit。

### 5.9 Candidate validity

每个 sample 必须通过：

```text
all candidate values are unique
correct score is highest
score(correct) - max_negative_score >= margin
each negative has exactly one primary counterfactual operator
number-only baseline cannot identify correct above chance
graph-only baseline cannot identify correct above chance
candidate-only audit near chance
```

## 6. Visual rule 与 AoT 的关系

AoT 是抽象数学结构；visual rule 是把 AoT 变成图像并可从图像恢复 AoT 的机制。

```text
AoT:
  expression tree, operators, leaves, answer

visual rule:
  grouping-to-quantity
  place/scale/color-to-value
  port-binding
  layout-to-execution-order
  cross-panel correspondence
```

这两个不能混：

```text
如果只有 AoT，没有 visual rule -> 文本数学题的图像版。
如果只有 visual rule，没有 AoT -> RAVEN/ARC-like visual pattern task。
VQ-Expr 必须是 visual rule induces AoT, AoT executes arithmetic.
```

## 7. 相关领域调研

### 7.1 Fuzzy logic / fuzzy arithmetic

| 工作 | 借鉴点 | 对 VQ-Expr 的作用 |
| --- | --- | --- |
| [Zadeh, Fuzzy Sets, 1965](https://doi.org/10.1016/S0019-9958(65)90241-X) | 集合隶属度可以是 `[0,1]` 连续值。 | 每个 quantity attribute 生成 `mu_q(n)`。 |
| [Logic Tensor Networks / Real Logic](https://arxiv.org/abs/1606.04422) | first-order formulas 可以用 many-valued semantics。 | visual predicates、edge binding、gate semantics 都可进入 rule truth。 |
| [Logic Tensor Networks, AIJ 2022](https://arxiv.org/abs/2012.13635) | neuro-symbolic fuzzy logic 可以把 predicates、functions、quantifiers 统一。 | 提供 object-variable + predicate truth 的理论语言。 |
| Fuzzy arithmetic / extension principle | fuzzy numbers 经过 `+ - * /` 传播为新的 fuzzy number。 | 让 `mu_q(n)` 沿 AoT 传播，而不只是 leaf uncertainty。 |

关键结论：

```text
VQ-Expr 的 fuzzy 要落在 attribute 和 program truth，而不是落在图像噪声。
```

### 7.2 Symbolic / programmatic visual reasoning

| 工作 | 借鉴点 | 对 VQ-Expr 的作用 |
| --- | --- | --- |
| [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) | scene graph + functional program + bias control。 | 每题保存 scene graph、AoT、visual_rule_program、oracle trace。 |
| [GQA](https://arxiv.org/abs/1902.09506) | scene graph 生成 compositional questions，强调 consistency/grounding。 | VQ-Expr 要报告 quantity grounding、operator grounding、program consistency。 |
| [RAVEN](https://arxiv.org/abs/1903.02741) | abstract visual reasoning 使用结构化规则。 | 我们承认其方向，但差异是 arithmetic AoT + fuzzy attributes。 |
| [Bongard-LOGO](https://arxiv.org/abs/2010.00763) | program-guided visual concept generation。 | visual rule family 要可程序化、可复算、human-interpretable。 |
| [COG](https://openaccess.thecvf.com/content_ECCV_2018/html/Guangyu_Robert_Yang_A_dataset_and_ECCV_2018_paper.html) | configurable visual task generator。 | VQ-Expr 需要可配置 family、depth、noise、binding demand。 |

关键结论：

```text
我们不能只交图像和答案。
必须交可执行 metadata：quantity attributes, visual rules, AoT, counterfactual logs。
```

### 7.3 Long expression / compositional generalization

| 工作 | 借鉴点 | 对 VQ-Expr 的作用 |
| --- | --- | --- |
| [ListOps](https://arxiv.org/abs/1804.06028) / [Long Range Arena](https://arxiv.org/abs/2011.04006) | 长树结构和长度泛化。 | VQ-Expr 要有 depth/leaf-count OOD split。 |
| [CLUTRR](https://arxiv.org/abs/1908.06177) | 组合推理的长度系统泛化。 | AoT depth 和 visual binding path length 要分开做 OOD。 |
| [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset) | 程序化生成数学题并控制模块。 | 我们借鉴 expression generator，但不用文本题面。 |

关键结论：

```text
长 expression 不是把图画复杂，而是用 AoT depth、tree shape、operator composition 做系统泛化。
```

### 7.4 Visual numerosity / non-symbolic number

| 方向 | 借鉴点 | 对 VQ-Expr 的作用 |
| --- | --- | --- |
| [non-symbolic numerosity database](https://www.nature.com/articles/s41597-023-01933-6) | dot ratio、area、convex hull、perimeter、distance 等视觉属性会影响 numerosity processing。 | 不把 raw dots 作为主 family；必须做 confound audit。 |
| [visual numerosity properties](https://pmc.ncbi.nlm.nih.gov/articles/PMC3355123/) | convex hull、aggregate surface、density 等会影响数量估计。 | 记录 `visual_confounds`，并做 confound-balanced negatives。 |
| [base-ten / place value](https://fhsu.pressbooks.pub/ecumath/chapter/chapter-10-whole-number-place-value/) | 1-99 可以由 tens/ones 结构化表示。 | 作为扩展版背景；v1 先做 1-9。 |
| area model | 面积/阵列适合乘除法。 | Area-Tile family 与 Mul/Div gate 对齐。 |
| [fuzzy color naming dataset](https://www.cvc.uab.es/color_naming/) / [fuzzy color naming](https://pubmed.ncbi.nlm.nih.gov/18830336/) | hue、lightness、saturation 可以用 fuzzy category membership 表示。 | Color-Lattice 作为辅助 fuzzy attribute，而不是唯一数制。 |

关键结论：

```text
我们需要 symbolic-friendly visual quantity attributes，
而不是纯 approximate number sense。
```

## 8. Yangshi-style proxy split

| Item | VQ-Expr 定义 |
| --- | --- |
| Proxy A | 模型能读显式数字、浅层数量或固定视觉模板。 |
| Construct B | 模型能从 fuzzy visual attributes 恢复数量分布，并沿 AoT 和 visual binding 执行长表达式。 |
| Regime R | 数量 multiset、operator multiset、像素统计近似平衡，但 attribute membership、binding edge 或 AoT tree 被局部 counterfactual 改变。 |
| Mechanism H | Fuzzy Quantity-AoT Binding Collapse：模型把数量解码、视觉绑定和表达式执行分开 harden，不能传播 fuzzy uncertainty。 |
| Artifact O | 由 5 类 quantity attributes、AoT generator、visual rule compiler、7 类 counterfactual negatives 和 fuzzy oracle 组成的数据集。 |

一句话 thesis：

```text
Current visual arithmetic benchmarks can still reduce reasoning to explicit numerals or short templates;
VQ-Expr tests whether models can recover fuzzy visual quantities and propagate them through AoT-generated arithmetic expressions under controlled counterfactual choices.
```

## 9. 第一版实现路线

### Phase 1: Attribute bank

实现：

```text
Grouped Units
Rod-Length Train
Area-Tile Patch
Graph-Path Count
Color-Lattice
```

测试：

```text
coverage 1..99 for first four families
coverage 1..9 for all v1 families
true numeric_value in fuzzy support
no Arabic digit or operator glyph
confound stats recorded
```

### Phase 2: AoT generator

实现：

```text
sample tree shape
sample operator labels
reverse-sample leaf values from root answer
assign attribute family to each leaf
execute crisp and fuzzy oracle
```

### Phase 3: Choice generator

生成：

```text
C0 correct
C1 output perturbation
C2 leaf attribute perturbation
C3 operator swap
C4 port swap
C5 binding rewire
C6 scope/tree rotation
C7 fuzzy ambiguity trap
```

### Phase 4: Visual rule compiler

至少实现：

```text
circuit-flow family
container-machine family
path-transformation family
```

### Phase 5: Evaluation splits

```text
IID
Value-OOD
Attribute-Family-OOD
Color-Fuzzy-OOD
AoT-Depth-OOD
Operator-Composition-OOD
Binding-Ambiguity-Stress
Candidate-Bias-Audit
```

## 10. 当前 prototype 与本方案的差距

当前 prototype 已有：

```text
5 quantity families except color
one fixed long expression
fuzzy_value metadata
fuzzy-visible ghost primitives and alternative edges
presentation/debug images
tests/test_vqexpr.py
```

仍缺：

```text
general AoT sampler
reverse sampling from arbitrary root answer
seven negative choices as real rendered candidate panels
fuzzy arithmetic propagation through arbitrary tree
confound-balancing audit
OOD split generator
```

因此当前可以说：

```text
we have a VQ-Expr probe
```

不能说：

```text
we have completed the benchmark
```

## 11. 当前允许与禁止 claim

允许：

```text
VQ-Expr is designed around fuzzy visual quantity attributes and AoT-generated arithmetic expressions.
The first probe demonstrates that a long expression can be rendered without Arabic numerals or operator glyphs.
The next implementation step is a general AoT sampler plus seven counterfactual choice operators.
```

禁止：

```text
VQ-Expr proves models cannot do fuzzy symbolic reasoning.
VQ-Expr is already an Oral-level benchmark.
VQ-Expr is the first fuzzy visual arithmetic benchmark.
Color-Lattice alone is a natural complete number system.
```
