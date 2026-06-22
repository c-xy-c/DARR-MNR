# VQ-Expr 设计草案：无显式数字的视觉数量属性与长表达式程序

日期：2026-06-21

本文档修正此前 AFVNB/FVNB 方案的核心问题：我们不应继续用 `inner / boundary / outlier`
这类三角色空间绑定，也不应在图像中显示阿拉伯数字。新的研究对象应是：

```text
visual quantity attributes -> fuzzy value distribution over 1..99
visual operator graph -> executable arithmetic expression
program evaluation -> 8-way candidate label
```

也就是说，数值不再由 glyph/OCR 读取，而由可见但非阿拉伯数字的视觉属性解码；长表达式不再由文本
`3 + 5 * 2` 表示，而由视觉门、端口和连线表示。

## 1. 先修正研究顺序

旧路线的问题：

```text
先有 inner/boundary/outlier 三个 role
再把可见数字填进去
再说 fuzzy role 改变算式语义
```

这会导致两个风险：

1. 视觉上仍然像 MNR：模型先读数字，再读槽位。
2. 表达式长度很难上去：三个 spatial roles 只能支撑 `a-b=c` 级别的 toy rule。

新路线必须反过来：

```text
Step A: 先定义 4-5 种可控的视觉数量 attribute，能表示 1..99。
Step B: 每个 quantity object 产生 fuzzy value distribution，而不是裸 integer。
Step C: 再定义视觉 operator gates 和 binding edges，组合成长 expression。
Step D: 用 executable program graph 生成候选和 counterfactual negatives。
```

## 2. 调研结论：该借鉴什么，不该借鉴什么

| 来源 | 可借鉴机制 | 对 VQ-Expr 的具体启发 | 不照搬 |
| --- | --- | --- | --- |
| [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) | scene graph + functional program + bias control | 每个样本必须保存 scene graph、program graph、operator trace，可复算标签。 | 不做语言 VQA；不让 prompt 泄露运算。 |
| [Bongard-LOGO](https://arxiv.org/abs/2010.00763) | program-guided visual generation | 图像应由程序生成，且概念/规则 human-interpretable。 | 不做二分类概念学习；我们做 arithmetic program execution。 |
| [COG](https://openaccess.thecvf.com/content_ECCV_2018/html/Guangyu_Robert_Yang_A_dataset_and_ECCV_2018_paper.html) | configurable visual tasks + compositional visual reasoning | 数据集要有可配置 task families、长度、噪声、memory/binding demand。 | 不做视频或 working memory 主任务。 |
| [CLUTRR](https://arxiv.org/abs/1908.06177) / [ListOps](https://arxiv.org/abs/1804.06028) | systematic generalization and length split | 必须有 expression length OOD、held-out operator compositions、tree-depth split。 | 不用文本 prefix expression。 |
| [non-symbolic numerosity studies](https://www.nature.com/articles/s41597-023-01933-6) | dot ratio、area、convex hull、density 等视觉 confound | raw dot arrays 容易和非数值视觉特征纠缠；需要显式控制 confounds。 | 不把随机点阵作为主表示。 |
| [base-ten/place-value learning](https://access.openupresources.org/curricula/our-k5-math/en/grade-1/unit-4/family.html) | tens/ones decomposition up to 99 | 1..99 的主表示应采用十进制组合，而不是 99 个散点。 | 不显示阿拉伯数字或文字数名。 |
| [abacus representation](https://www.math-only-math.com/representation-of-numbers-on-the-abacus.html) | rods/columns encode place value | 可用作紧凑、可精确、可 fuzzy 的 1..99 表示。 | 不让它退化成另一个数字符号表；要做 style/OOD 和混合表征。 |
| [area model](https://doodlelearning.com/maths/skills/multiplication/area-model) | multiplication/division as visual area/partition | `*` 和 `/` 不应只是门形状；可以用 area/partition 让操作语义视觉化。 | 不在图中写 factors 或 equations。 |

核心结论：

```text
Raw dots are useful for approximate numerosity,
but not enough for symbolic long-expression reasoning over 1..99.

VQ-Expr should use decomposable visual attributes:
place value, grouped cardinality, length, area, topology/path count.
```

## 3. Quantity attribute 的硬约束

每种 attribute 都必须满足：

| 约束 | 说明 |
| --- | --- |
| No Arabic glyph | 图像中不出现 `0-9`、`+`、`-`、`*`、`/`、`=`。 |
| 1..99 coverage | 每个 attribute family 都能覆盖至少 1..99，不能只支持小数值。 |
| Symbolic-friendly | 值能分解成可复算的 components，不是不可解释黑箱视觉纹理。 |
| Fuzzy value | 每个 object 不是只给 `numeric_value`，还给 `mu_value[n]` over `n in 1..99`。 |
| Confound control | 面积、密度、周长、凸包、颜色、位置不能单独泄露标签。 |
| Renderer OOD | 同一个数值能用多个 style 渲染，支持 attribute-family/style OOD。 |

统一 value object：

```json
{
  "id": "q_03",
  "class": "quantity_carrier",
  "attribute_family": "base10_bundle",
  "numeric_value": 47,
  "components": {
    "tens": {"count": 4, "confidence": 0.93},
    "ones": {"count": 7, "confidence": 0.88}
  },
  "fuzzy_value": {
    "support": [36, 37, 46, 47, 48, 57],
    "membership": {"36": 0.05, "37": 0.12, "46": 0.31, "47": 1.0, "48": 0.42, "57": 0.18}
  },
  "visual_confounds": {
    "area": 381.2,
    "convex_hull": 510.5,
    "perimeter": 124.1,
    "density": 0.42
  },
  "no_visible_digits": true
}
```

`numeric_value` 是 generator/oracle 的 canonical value；`fuzzy_value` 是模型应该从视觉属性恢复的数值隶属度。
程序评估可以有两个版本：

```text
crisp oracle: read numeric_value exactly, used for deterministic label.
fuzzy oracle: propagate mu_value through arithmetic, used for ambiguity/fuzzy split.
```

## 4. 五种主 quantity attribute

### 4.1 Base10-Bundle Attribute

视觉形式：

```text
tens = tower/bundle of 10 identical units
ones = loose units
number n = 10 * T + O
T in 0..9, O in 0..9, n in 1..99
```

例子：

```text
47 = four ten-bundles + seven loose units
```

为什么适合 symbolic：

```text
它天然给出 tens/ones decomposition；
不需要让模型数 99 个散点；
支持 carry/borrow、+/-、place-value OOD。
```

Fuzzy 设计：

```text
mu_T(t): 第 t 个 ten-bundle count 的隶属度
mu_O(o): loose-unit count 的隶属度
mu_value(n) = max_{10t+o=n} Tnorm(mu_T(t), mu_O(o))
```

可控 fuzzy 来源：

```text
bundle 边界轻微遮挡
loose units 部分重叠
one bundle 被画成 9.5/10.5-like ambiguous completion
不同 bundle style 造成 attribute-family OOD
```

适合 splits：

```text
IID main, value-range OOD, carry/borrow stress, style OOD.
```

### 4.2 Abacus-Place Attribute

视觉形式：

```text
two rods/columns: tens rod and ones rod
bead activation count on each rod encodes T and O
n = 10 * T + O
```

例子：

```text
73 = seven active beads on tens column + three active beads on ones column
```

为什么适合 symbolic：

```text
compact;
place-value explicit;
1..99 覆盖稳定；
容易生成端口/连线，把一个 carrier 接入 expression graph。
```

风险：

```text
它接近一种替代数字符号，可能被 reviewer 认为只是换了 notation。
```

控制策略：

```text
只作为一个 family，不作为唯一 family；
训练/测试跨 family；
同一数值可在 abacus/base10/rod/area/path 之间互换；
不要在图中标注 tens/ones 文字。
```

Fuzzy 设计：

```text
mu_T(t): active bead count or bead-position state on tens rod
mu_O(o): active bead count or bead-position state on ones rod
mu_value(n) = mu_T(floor(n/10)) * mu_O(n mod 10)
```

可控 fuzzy 来源：

```text
bead near activation threshold
bead partly occluded
rod boundary jitter
ambiguous bead grouping
```

### 4.3 Rod-Length Train Attribute

视觉形式：

```text
quantity = total length of a train of rods measured in unit ticks
rod lengths are not labeled by numbers
n = sum_i length(rod_i)
```

例子：

```text
58 = five long rods of length 10 + one rod of length 8
or = rods [20, 20, 10, 8] if larger rods are allowed
```

为什么适合 symbolic：

```text
长度是连续视觉量，但可被离散 unit ticks 复算；
天然适合 number-line、ratio、scale、division by segment；
和 dot-count family 的视觉统计完全不同。
```

Fuzzy 设计：

```text
mu_len_i(k): rod_i has length k
mu_value(n) = max_{sum k_i=n} product_i mu_len_i(k_i)
```

可控 fuzzy 来源：

```text
endpoint uncertainty
tick marks missing or faint
rod overlap
perspective/scale jitter
```

适合 splits：

```text
length OOD, scale OOD, continuous-to-discrete grounding stress.
```

### 4.4 Area-Tile Attribute

视觉形式：

```text
quantity = number of filled unit cells in a rectangular or polyomino patch
n = area in unit cells
```

例子：

```text
48 = 6 x 8 filled cell rectangle
or = irregular polyomino with 48 cells
```

为什么适合 symbolic：

```text
支持 1..99；
天然连接 multiplication/division；
可表达 factorization: 48 as 6x8, 4x12, etc.
```

Fuzzy 设计：

```text
mu_cell_j(filled): each cell occupancy confidence
mu_value(n) = probability/membership that exactly n cells are filled
```

为避免组合爆炸，实现时可使用近似：

```text
mean_count = sum_j mu_cell_j
mu_value(n) = triangular_or_gaussian(n; mean_count, sigma_from_boundary_noise)
```

可控 fuzzy 来源：

```text
partial cell fill
blurred patch boundary
holes/noise cells
ambiguous grid alignment
```

适合 splits：

```text
multiplication/division stress, factorization OOD, area-vs-count confound audit.
```

### 4.5 Graph-Path Count Attribute

视觉形式：

```text
quantity = path length, node count, or token count along a graph route
n = number of selected edges/nodes
```

例子：

```text
31 = highlighted route of 31 edges
```

为什么适合 symbolic：

```text
它把 quantity 与 structure 绑定；
非常适合测试 long-range visual binding；
和 expression graph 自身同构，能做 path-as-value 和 graph-as-program 的双层任务。
```

Fuzzy 设计：

```text
mu_edge(e selected): edge selection confidence
mu_value(n): membership over possible path lengths under uncertain selected edges
```

可控 fuzzy 来源：

```text
edge gaps
branch ambiguity
near-overlapping routes
node occlusion
```

适合 splits：

```text
long-range binding, branch ambiguity, graph topology OOD.
```

风险：

```text
如果 path 太长，视觉拥挤会让任务退化为 low-level tracing。
```

控制策略：

```text
用 chunked paths：每 10 edges 一个 visual segment；
限制单图拥挤度；
保存 path graph oracle。
```

## 5. 不建议作为主 family 的表示

### Raw Random Dot Array

优点：

```text
符合 non-symbolic numerosity；
适合 approximate number sense。
```

问题：

```text
1..99 精确表达需要大量点；
area, density, convex hull, perimeter 等 confounds 强；
长 expression 中多个 raw dot arrays 会非常拥挤。
```

结论：

```text
可作为 fuzzy/noisy auxiliary family 或 stress split；
不作为主 symbolic-friendly family。
```

### Pure Color/Texture Code

问题：

```text
颜色到数值的映射太像人工密码；
不具备数学教具或数量感基础；
容易被认为是 arbitrary lookup table。
```

结论：

```text
不用作主数值属性。
```

## 6. 视觉 rule 设计：先由图像生成 program，再执行数学表达式

这里必须区分两个概念：

```text
mathematical expression:
  Add, Sub, Mul, Div 这些可执行算术程序。

visual rule:
  图像中的 grouping、ordering、connection、containment、motion/flow、panel correspondence
  如何生成 quantity objects、operator gates、binding edges 和 expression tree。
```

如果只有 mathematical expression，没有 visual rule，数据集就退化成“把文本表达式画成电路图”。
如果只有 visual rule，没有 mathematical expression，就会接近 RAVEN/ARC 风格的 pattern induction。
VQ-Expr 的核心是：

```text
visual rule induces executable expression;
expression result determines the candidate.
```

### 6.1 Visual rule 的分层

每个样本保存两套程序：

```text
visual_rule_program:
  image primitives -> quantity carriers / gates / edges / panel correspondences

arithmetic_program:
  quantity values + operator graph -> output value
```

也就是说，模型需要先隐式执行 visual rule，再执行 arithmetic program。metadata 中二者分开保存，方便诊断：

```json
{
  "visual_rule_program": {
    "grouping_rules": [],
    "binding_rules": [],
    "layout_rules": [],
    "cross_panel_rules": []
  },
  "arithmetic_program": {
    "root": "g4",
    "operators": ["Add", "Mul", "Sub", "Div"]
  }
}
```

### 6.2 五类主 visual rule

#### Rule A: Grouping-to-Quantity

作用：

```text
决定哪些 primitive 属于同一个 quantity carrier。
```

可用视觉依据来自 Gestalt 风格原则：proximity、similarity、common region、continuity、closure。
这些原则本身只给 perceptual grouping；我们把它们程序化成可控生成规则。

例子：

```text
nearby beads inside the same faint hull -> one quantity carrier
same-color rods aligned on one baseline -> one rod train
tiles enclosed by the same boundary -> one area-tile object
continuous highlighted edges -> one graph-path object
```

形式化：

```text
Group(p_i, q_j) = sigma(
    w1 * proximity(p_i, q_j)
  + w2 * common_region(p_i, q_j)
  + w3 * appearance_similarity(p_i, q_j)
  + w4 * continuity(p_i, q_j)
)
```

输出：

```text
quantity_carrier q
  primitives = {p_i}
  fuzzy_value = decode_attribute(primitives)
```

它解决的问题：

```text
数值不是单个 glyph；
数值来自视觉元素如何被 grouping 成一个对象。
```

#### Rule B: Place/Scale-to-Value

作用：

```text
把一个 grouped quantity carrier 解码成 1..99 的 fuzzy value distribution。
```

不同 attribute family 使用不同 visual rule：

```text
base10_bundle:
  value = 10 * count(bundle) + count(loose_unit)

abacus_place:
  value = 10 * active_count(left_column) + active_count(right_column)

rod_length_train:
  value = sum(unit_length(rod_i))

area_tile:
  value = count(filled_unit_cells)

graph_path_count:
  value = count(selected_edges_or_nodes)
```

形式化：

```text
mu_value(n | q, family) =
  max over latent decompositions z:
    Tnorm(mu_components(z | q), indicator(value_family(z) = n))
```

这类 rule 是视觉 rule，不是数学 expression。它只回答：

```text
这个视觉对象代表哪个数？
```

#### Rule C: Port-Binding

作用：

```text
决定哪个 quantity/operator output 连接到哪个 operator input port。
```

这是 VQ-Expr 最关键的 visual-math 联动点。数学表达式的结构不是给定的，而是由连线、端口、方向、距离、连续性决定。

例子：

```text
wire from q1 attaches to left port of g1 -> input_1(g1)=q1
wire from q2 attaches to right port of g1 -> input_2(g1)=q2
output of g1 flows into left port of g2 -> input_1(g2)=g1
```

Fuzzy 形式：

```text
Bind(src, gate, port) =
  softmax_port(
    alpha * endpoint_closeness(src_wire, port)
  + beta  * tangent_continuity(src_wire, port)
  + gamma * occlusion_penalty
  + eta   * port_type_compatibility
  )
```

对于非交换运算，port-binding 直接改变语义：

```text
Sub(a, b) != Sub(b, a)
Div(a, b) != Div(b, a)
```

这就是“视觉联动”最硬的地方：

```text
同样的 quantity values，
同样的 gates，
只改变 visual binding edges，
expression semantics 就改变。
```

#### Rule D: Layout-to-Execution Order

作用：

```text
决定 expression tree / DAG 的执行顺序。
```

不能只靠固定从左到右，否则会变成位置模板。应提供几种 layout rules：

| Layout rule | 视觉依据 | 生成的程序结构 |
| --- | --- | --- |
| Pipeline flow | directed wires from source to sink | left-to-right or top-to-bottom DAG |
| Nested machines | gates inside larger containers | inner gate evaluated before outer gate |
| Junction priority | wire merges at junction nodes | local subexpression first |
| Color/texture lanes | same lane marks a computation stream | lane-specific subprogram |
| Depth layering | visual depth/shadow/overlap | foreground/background precedence |

Fuzzy 形式：

```text
Precedes(u, v) =
  Tnorm(
    path_exists(u -> v),
    direction_confidence(u -> v),
    layer_order_confidence(u, v)
  )
```

这类 visual rule 解决长 expression 的关键问题：

```text
不是把很长的表达式写出来；
而是用视觉结构生成一个深度 3..6 的 program graph。
```

#### Rule E: Cross-Panel Correspondence

作用：

```text
定义 context panels 和 candidate panels 中哪些 visual objects / gates / subprograms 是同一个抽象角色。
```

MNR 的 protocol 是 3 context + 8 candidate。VQ-Expr 保留这个外壳，但 context 不再只是给数字模板，而是给出 shared visual rule。

例子：

```text
context panels show the same visual rule schema:
  two quantities enter a merge gate,
  output enters a repeater gate,
  final stream enters splitter gate.

candidates vary:
  output quantity carrier
  binding edge
  gate type
  tree layout
```

视觉 correspondence 可由以下属性定义：

```text
same gate morphology
same port topology
same lane color/texture
same relative graph structure
same quantity attribute family or allowed family-substitution
```

Fuzzy 形式：

```text
Correspond(o_i^context, o_j^candidate) =
  Tnorm(
    morphology_similarity,
    topology_position_similarity,
    lane_identity_similarity,
    family_compatibility
  )
```

这类 rule 决定：

```text
candidate 是否遵循 context 给出的 visual computation rule。
```

### 6.3 三种 visual-rule family

为了不把所有东西堆到一个图里，v1 应该先做三种 family。每种 family 都能生成长表达式，但 visual rule 不同。

#### Family 1: Circuit-Flow

视觉对象：

```text
quantity carriers + gates + wires + typed ports
```

Visual rules：

```text
grouping-to-quantity: carrier hull / local grouping
port-binding: wire endpoint to port
layout-to-order: directed acyclic flow
cross-panel: same gate topology
```

适合：

```text
最小 killer sample；
绑定交换、端口交换、operator swap、tree swap 都容易实现。
```

风险：

```text
太像电路符号系统。
```

控制：

```text
operator gate 不使用标准数学符号；
quantity family 混合；
style OOD；
用 fuzzy port attachment 避免固定模板。
```

#### Family 2: Container-Machine

视觉对象：

```text
nested machines / trays / bins / conveyors
```

Visual rules：

```text
common region defines scope;
objects entering same tray are grouped as operands;
nested tray means inner subexpression;
output slot of one tray feeds another tray.
```

例子：

```text
two quantity carriers placed into a merge tray;
the tray output is placed into a repeat machine;
the repeat output is partitioned by a splitter tray.
```

适合：

```text
把 precedence/scope 做得更视觉化；
避免纯线缆图。
```

#### Family 3: Path-Transformation

视觉对象：

```text
tokens move along a path; gates are stations on the path
```

Visual rules：

```text
route continuity defines execution order;
station morphology defines operation;
side branches inject operands;
fork/merge defines local expression tree.
```

例子：

```text
a quantity token flows through Add station, then Mul station, then Div station;
side input quantities attach at each station.
```

适合：

```text
长序列 expression；
OOD length split；
graph-path quantity attribute 与 program graph 联动。
```

### 6.4 Visual rule counterfactuals

候选负例不能只是数学结果错，要明确是哪条 visual rule 被破坏：

| Counterfactual | 改变的 visual rule | 数学表达式如何变 |
| --- | --- | --- |
| grouping split | Rule A | 一个 quantity 被分成两个 carrier，value 改变 |
| grouping merge | Rule A | 两个 carriers 被合并，value 改变 |
| place-value swap | Rule B | tens/ones 或 row/column 解码改变 |
| port swap | Rule C | `Sub(a,b)` 变成 `Sub(b,a)` |
| edge rewire | Rule C | operand binding 改变 |
| scope shift | Rule D | `(a+b)*c` 变成 `a+(b*c)` |
| flow reversal | Rule D | execution order 改变 |
| correspondence break | Rule E | candidate 不再遵循 context visual schema |

每个负例记录：

```json
{
  "counterfactual_type": "scope_shift",
  "violated_visual_rule": "Layout-to-Execution Order",
  "arithmetic_delta": {
    "before": "Mul(Add(q1,q2),q3)",
    "after": "Add(q1,Mul(q2,q3))"
  }
}
```

### 6.5 视觉 rule 的验收标准

一个 visual rule family 可以进入 benchmark，必须通过：

```text
1. No visible digits/operators:
   evaluation image 中无 0-9, +, -, *, /, =。

2. Recomputable:
   visual_rule_program + scene primitives 可以复算 quantity/gate/edge/tree。

3. Non-template:
   同一数学 expression 可以有多种 layout/style；
   同一 layout 可以承载不同 expression。

4. Counterfactual-local:
   每个 wrong candidate 主要只破坏一条 visual rule。

5. Long-expression capable:
   支持 depth 3..6, leaves 3..9。

6. Fuzzy-aware:
   grouping、value、port、order、correspondence 至少一层有连续 membership。
```

## 7. 数学表达式语法：quantity 与 visual rule 之后才有 +-*/

VQ-Expr 的表达式不是文本序列，而是 visual program graph。

### 7.1 Program grammar

```text
Q ::= quantity_carrier
E ::= Q
    | Add(E, E)
    | Sub(E, E)
    | Mul(E, E)
    | Div(E, E)
```

生成约束：

```text
leaf quantity values in 1..99
intermediate values may be bounded, e.g. 1..199 for v1 debug
candidate answer values in 1..99
expression depth = 2..6 for v1
number of leaves = 3..9 for v1
operator mix controlled by split
avoid degenerate identities: +0, *1, /1 unless in diagnostic split
division must be exact in crisp oracle for v1
subtraction either requires a >= b or uses AbsDiff gate explicitly
```

### 7.2 Operator gates

| Operation | Visual gate | Semantics | Input order matters? | Fuzzy source |
| --- | --- | --- | --- | --- |
| `Add(a,b)` | merge funnel / union basin | combine quantities | no | edge-to-port membership, carrier value uncertainty |
| `Sub(a,b)` | cancellation tray | remove b from a | yes | left/right port ambiguity, incomplete cancellation |
| `Mul(a,b)` | array/repeater gate | repeat a by b or area a x b | yes/typed | row/column ambiguity, lane count uncertainty |
| `Div(a,b)` | splitter/partition gate | partition a into b equal groups | yes | bin membership, remainder ambiguity |

关键点：

```text
图像里不出现 + - * / =。
门形状、端口类型、连线方向决定操作。
debug metadata 才保存 operator name。
```

### 7.3 Expression graph schema

```json
{
  "program_graph": {
    "nodes": [
      {"id": "q1", "type": "quantity", "family": "base10_bundle"},
      {"id": "q2", "type": "quantity", "family": "rod_length_train"},
      {"id": "g1", "type": "operator_gate", "op": "Add"},
      {"id": "g2", "type": "operator_gate", "op": "Mul"}
    ],
    "edges": [
      {"src": "q1", "dst": "g1", "port": "left", "membership": 0.96},
      {"src": "q2", "dst": "g1", "port": "right", "membership": 0.91},
      {"src": "g1", "dst": "g2", "port": "left", "membership": 0.88}
    ],
    "root": "g2"
  }
}
```

### 7.4 Fuzzy arithmetic over values

每个 quantity carrier 给出一个 fuzzy set：

```text
mu_A(x), x in {1, ..., 99}
```

用 fuzzy extension principle 定义运算：

```text
mu_{A+B}(z) = max_{x+y=z} Tnorm(mu_A(x), mu_B(y))

mu_{A-B}(z) = max_{x-y=z} Tnorm(mu_A(x), mu_B(y))

mu_{A*B}(z) = max_{x*y=z} Tnorm(mu_A(x), mu_B(y))

mu_{A/B}(z) = max_{x=y*z} Tnorm(mu_A(x), mu_B(y))
```

再把 gate/edge 的视觉置信度乘进去：

```text
truth(program, candidate) =
  Tnorm(
    mu_program_output(candidate_value),
    all edge_port_memberships,
    all gate_type_memberships,
    all quantity_visibility_confidences
  )
```

v1 可以先用 crisp oracle 生成 label，用 fuzzy oracle 做 stress split：

```text
crisp label = exact evaluation over numeric_value
fuzzy score = evaluation over mu_value and visual memberships
```

## 8. 一个真正的长 expression 样例

图像里没有数字、没有运算符，只有五个 quantity carriers、四个 gates、若干连线：

```text
E = Div(
      Mul(
        Add(q1, q2),
        q3
      ),
      Sub(q4, q5)
    )
```

canonical values:

```text
q1 = 14  via base10_bundle
q2 =  7  via rod_length_train
q3 =  4  via abacus_place
q4 = 15  via area_tile
q5 =  3  via graph_path_count
```

evaluation:

```text
Add(14, 7) = 21
Mul(21, 4) = 84
Sub(15, 3) = 12
Div(84, 12) = 7
```

候选答案是 8 个 output quantity carriers，不显示数字：

```text
correct candidate: quantity carrier encoding 7
wrong candidates: encoding values produced by controlled counterfactuals
```

同一组 quantity objects 可以通过换连线产生另一个 expression：

```text
E' = Div(
       Mul(
         Add(q1, q3),
         q2
       ),
       Sub(q4, q5)
     )
```

这时：

```text
Add(14, 4) = 18
Mul(18, 7) = 126
Sub(15, 3) = 12
Div(126, 12) is not exact
```

这个 counterfactual 只改变 binding edge，不改变 quantity visual attributes。它直接测试：

```text
模型是否真的读了 visual program binding，而不是只读了数量 multiset。
```

## 9. 8-way candidates 应如何生成

每个 sample 的候选不应该随机凑。建议固定 8 类：

| Candidate | Counterfactual | 保持不变 | 破坏内容 |
| --- | --- | --- | --- |
| 0 | correct | all | none |
| 1 | value off-by-one | program graph | output quantity value |
| 2 | binding swap | quantities + gates | two input edges swapped |
| 3 | port swap | quantities + gates + edges mostly | left/right port for non-commutative op |
| 4 | operator swap | quantities + graph topology | one gate type, e.g. Add -> Sub |
| 5 | precedence/tree swap | quantities + local gates | expression tree structure |
| 6 | attribute-family decoy | final value maybe close | visual attribute family/style shortcut |
| 7 | confound-balanced decoy | area/density/color/count stats | executable program truth |

每个 candidate 都记录：

```json
{
  "counterfactual_operator": "binding_swap",
  "operator_log": [
    {"type": "swap_edges", "edge_a": "e_q2_g1", "edge_b": "e_q3_g2"}
  ],
  "verifier": {
    "only_primary_constraint_changed": true,
    "crisp_truth": 0,
    "fuzzy_truth": 0.18
  }
}
```

## 10. Splits

| Split | 目的 | 例子 |
| --- | --- | --- |
| IID | 基本学习 | seen values, seen families, depth 2..4 |
| Value-OOD | 数值泛化 | train 1..49, test 50..99 |
| Family-OOD | 表征迁移 | train base10/abacus/rod, test area/path |
| Style-OOD | 渲染迁移 | 同 family 换材质、间距、角度、颜色 |
| Length-OOD | 长表达式 | train depth <= 3, test depth 5..6 |
| Operator-OOD | 组合泛化 | train no `Div(Mul(...))`, test includes it |
| Fuzzy-OOD | 模糊程度泛化 | train low ambiguity, test port/value ambiguity high |
| Shortcut-Balanced | 候选偏置检查 | balance quantity multiset, pixel stats, family counts |

## 11. Baselines

必须包含的 diagnostic baselines：

| Baseline | 输入 | 预期作用 |
| --- | --- | --- |
| quantity-only oracle | extracted numeric values only, no graph | 测 quantity multiset shortcut |
| graph-only oracle | graph/gates only, no values | 测 structure-only shortcut |
| OCR/glyph baseline | raw image OCR | 应接近随机，因为无阿拉伯数字 |
| hard graph parser | argmax edge/gate/quantity only | 测 fuzzy collapse |
| fuzzy symbolic oracle | metadata fuzzy values + graph | upper bound |
| candidate-only model | candidates only | 测 RAVEN-FAIR style bias |

## 12. 实现阶段

### Phase 0: Quantity attribute probe

先不做完整 MNR protocol，只生成单 panel：

```text
5 attribute families x values 1..99 x fuzzy severity levels
```

验证：

```text
renderer can represent every n in 1..99
metadata has no Arabic digit text
fuzzy_value support contains true value
confound audit does not make value trivially recoverable from area/density alone
```

### Phase 1: Short expression circuit

支持：

```text
Add, Sub
depth 2..3
3..5 leaves
all values/result in 1..99
```

产物：

```text
one killer sample PNG
one debug metadata JSON
one verifier recomputation script
```

### Phase 2: Full operator set

增加：

```text
Mul via area/repeater gate
Div via partition gate
depth 3..6
mixed attribute families
8-way counterfactual candidates
```

### Phase 3: Benchmark splits

生成：

```text
IID
Value-OOD
Family-OOD
Length-OOD
Operator-OOD
Fuzzy-OOD
Shortcut-Balanced
```

## 13. 最小实现合同：模块、接口和生成流程

这一节把前面的研究设计压成实现接口。v1 不应先做大规模数据集，而应先做一个可复算的
`killer sample` generator。只要这个 generator 能工作，后续扩展就是工程问题；如果这个
generator 都需要偷用数字 glyph 或固定模板，方向就错了。

### 13.1 数据流

```text
SeedSpec
  -> sample quantities and arithmetic tree
  -> choose quantity attribute families
  -> render quantity carriers
  -> choose visual-rule family
  -> layout gates and binding edges
  -> render evaluation image
  -> derive visual_rule_program
  -> derive arithmetic_program
  -> verify crisp/fuzzy output
  -> generate 8 counterfactual candidates
  -> export PNG + JSON/NPZ metadata
```

核心原则：

```text
图像不是 arithmetic_program 的插图。
图像 primitives 通过 visual_rule_program 生成 arithmetic_program。
```

### 13.2 SeedSpec

一个 seed 应声明“想要的数学结构”和“允许的视觉规则”，但不直接决定像素细节：

```json
{
  "seed_id": "vqexpr_seed_0001",
  "target_expression": {
    "template": "Div(Mul(Add(q1,q2),q3),Sub(q4,q5))",
    "depth": 4,
    "num_leaves": 5,
    "allowed_ops": ["Add", "Sub", "Mul", "Div"],
    "result_range": [1, 99],
    "require_exact_division": true
  },
  "quantity_families": [
    "base10_bundle",
    "rod_length_train",
    "abacus_place",
    "area_tile",
    "graph_path_count"
  ],
  "visual_rule_family": "circuit_flow",
  "fuzzy_severity": "medium",
  "counterfactuals": [
    "value_off_by_one",
    "binding_swap",
    "port_swap",
    "operator_swap",
    "scope_shift",
    "family_style_decoy",
    "confound_balanced_decoy"
  ]
}
```

### 13.3 QuantityRenderer 接口

每个 quantity attribute family 必须实现同一个接口：

```text
render_quantity(value, style_seed, fuzzy_severity)
  -> image_primitives
  -> component_parse
  -> fuzzy_value_distribution over 1..99
  -> visual_confound_stats
  -> leakage_flags
```

例子：

```json
{
  "id": "q1",
  "family": "base10_bundle",
  "numeric_value": 14,
  "component_parse": {
    "tens_bundles": 1,
    "ones_units": 4
  },
  "fuzzy_value": {
    "support": [13, 14, 15, 24],
    "membership": {"13": 0.32, "14": 1.0, "15": 0.41, "24": 0.12}
  },
  "visual_confounds": {
    "primitive_count": 5,
    "ink_area": 284.0,
    "convex_hull_area": 391.0,
    "density": 0.73
  },
  "leakage_flags": {
    "contains_arabic_digit": false,
    "contains_operator_symbol": false,
    "text_ocr_risk": "none"
  }
}
```

每个 renderer 的单元测试必须覆盖：

```text
for n in 1..99:
  render_quantity(n).numeric_value == n
  true n in fuzzy_value.support
  contains_arabic_digit == false
```

### 13.4 VisualRuleCompiler 接口

VisualRuleCompiler 负责从布局和 primitive 关系中恢复程序结构：

```text
compile_visual_rules(primitives, family)
  -> quantity_groups
  -> operator_gates
  -> binding_edges
  -> execution_order
  -> visual_rule_program
```

它不执行算术，只生成 program graph：

```json
{
  "visual_rule_program": {
    "family": "circuit_flow",
    "grouping": [
      {"rule": "common_region", "target": "q1", "confidence": 0.96}
    ],
    "binding_edges": [
      {"src": "q1", "dst": "g_add", "port": "left", "confidence": 0.94},
      {"src": "q2", "dst": "g_add", "port": "right", "confidence": 0.92}
    ],
    "execution_order": [
      ["g_add"],
      ["g_mul"],
      ["g_sub"],
      ["g_div"]
    ]
  }
}
```

编译器必须能复现：

```text
same rendered primitives -> same visual_rule_program
```

但 generator 要支持：

```text
same arithmetic_program -> many visual_rule_program/layout variants
same visual_rule_family -> many arithmetic_programs
```

否则数据集会退化成模板匹配。

### 13.5 ArithmeticVerifier 接口

ArithmeticVerifier 只读两个东西：

```text
quantity_carriers.numeric_value or fuzzy_value
visual_rule_program-derived operator graph
```

输出：

```json
{
  "crisp_trace": [
    {"node": "g_add", "op": "Add", "inputs": [14, 7], "output": 21},
    {"node": "g_mul", "op": "Mul", "inputs": [21, 4], "output": 84},
    {"node": "g_sub", "op": "Sub", "inputs": [15, 3], "output": 12},
    {"node": "g_div", "op": "Div", "inputs": [84, 12], "output": 7}
  ],
  "output_value": 7,
  "fuzzy_output": {
    "support": [6, 7, 8],
    "membership": {"6": 0.25, "7": 1.0, "8": 0.31}
  }
}
```

禁止：

```text
从文件名、label、debug text、operator name text、visible digit text 读取答案。
```

### 13.6 CandidateFactory 接口

CandidateFactory 从 correct sample 生成 8-way candidates。每个 candidate 都必须由
counterfactual operator 产生，而不是随机采样：

```text
make_candidates(correct_scene)
  -> [
       correct,
       value_off_by_one,
       binding_swap,
       port_swap,
       operator_swap,
       scope_shift,
       family_style_decoy,
       confound_balanced_decoy
     ]
```

每个 counterfactual 必须声明它改了哪一层：

| Counterfactual | Layer | 示例 |
| --- | --- | --- |
| value_off_by_one | quantity attribute | q_out 的 area tile 多/少一个 cell |
| binding_swap | visual rule C | q2 和 q3 的 wire endpoint 交换 |
| port_swap | visual rule C | Sub/Div 的 left/right port 交换 |
| operator_swap | gate semantics | merge gate 替换为 cancellation gate |
| scope_shift | visual rule D | nested machine 边界移动，改变 tree |
| family_style_decoy | renderer style | 同 value 换 family/style，测外观 shortcut |
| confound_balanced_decoy | bias control | ink/area/count 近似匹配但 program false |

### 13.7 Killer sample 的具体参数

第一张样例不要太复杂，但必须过线：

```text
quantity values:
  q1 = 14 base10_bundle
  q2 = 7  rod_length_train
  q3 = 4  abacus_place
  q4 = 15 area_tile
  q5 = 3  graph_path_count

visual_rule_family:
  circuit_flow

expression:
  Div(Mul(Add(q1, q2), q3), Sub(q4, q5)) = 7

visual rules:
  grouping: each carrier's primitives form one quantity object
  binding: wires attach quantities/gate outputs to typed ports
  order: directed flow from left/top sources to right/bottom sink
  correspondence: all candidates preserve context topology except one declared counterfactual

image constraints:
  no Arabic digits
  no + - * / =
  no gate labels
  no answer labels
  no debug text in presentation image
```

这张图必须有两个输出：

```text
presentation.png:
  只显示图像任务本身。

debug.png:
  可显示 q ids、gate ids、trace、fuzzy memberships，用于论文解释和内部 QA。
```

### 13.8 文件落点

建议实现时新建模块，不继续扩 FVNB/AFVNB 旧文件：

```text
mnr_dataset/vqexpr_quantity.py
mnr_dataset/vqexpr_visual_rules.py
mnr_dataset/vqexpr_program.py
mnr_dataset/vqexpr_generator.py
mnr_dataset/vqexpr_candidates.py
mnr_dataset/vqexpr_visualize.py
tests/test_vqexpr.py
outputs/vqexpr_killer_sample/
```

最低测试：

```text
test_each_quantity_family_covers_1_to_99_without_digits
test_visual_rule_compiler_recovers_binding_edges
test_expression_trace_matches_expected_long_program
test_counterfactuals_change_only_declared_layer
test_presentation_image_has_no_digit_or_operator_glyphs
```

## 14. Yangshi-style proxy split

| Item | VQ-Expr 定义 |
| --- | --- |
| Proxy A | 模型能读显式数字或浅层视觉数量，并在短模板中匹配答案。 |
| Construct B | 模型能从多种非符号视觉属性恢复 fuzzy number，并沿视觉 program graph 执行长 arithmetic expression。 |
| Regime R | 数量 multiset、像素统计、candidate appearance 被平衡，但 binding edges / operator tree / fuzzy value distributions 改变。 |
| Mechanism H | Quantity-Program Binding Collapse：模型把视觉数量和程序结构分离处理，不能把 fuzzy numeric attributes 传播到长表达式执行。 |
| Artifact O | 一个由 4-5 种视觉数量 attribute、visual operator graph、fuzzy arithmetic oracle 和 counterfactual candidates 组成的数据集。 |

一句话 thesis：

```text
Current visual arithmetic benchmarks can still reduce reasoning to reading explicit numerals or matching short spatial templates;
VQ-Expr removes numeral glyphs and tests whether models can recover fuzzy quantities from symbolic-friendly visual attributes
and execute long arithmetic programs defined by visual operator graphs.
```

## 15. 当前结论

下一步不应该继续实现 `inner/boundary/outlier`。正确的实现入口是：

```text
1. 实现 quantity attribute renderers:
   base10_bundle, abacus_place, rod_length_train, area_tile, graph_path_count

2. 每个 renderer 输出:
   image primitives + numeric_value + fuzzy_value distribution + visual confound stats

3. 实现 visual expression graph:
   quantity nodes + operator gates + binding edges + output candidates

4. 实现 crisp/fuzzy verifier:
   crisp oracle 产生 deterministic label
   fuzzy oracle 产生 stress metrics

5. 做一个 killer sample:
   no Arabic digits, 5 quantity carriers, 4 gates, expression depth >= 4,
   8 candidate outputs, with binding/operator/tree counterfactuals.
```

如果这个 killer sample 立不住，后续扩数据集没有意义。

## 16. 当前 prototype 状态

已实现一个最小可复算 VQ-Expr killer sample：

```text
modules:
  mnr_dataset/vqexpr_quantity.py
  mnr_dataset/vqexpr_visual_rules.py
  mnr_dataset/vqexpr_program.py
  mnr_dataset/vqexpr_generator.py

tests:
  tests/test_vqexpr.py

outputs:
  outputs/vqexpr_killer_sample/vqexpr_killer_presentation.png
  outputs/vqexpr_killer_sample/vqexpr_killer_debug.png
  outputs/vqexpr_killer_sample/vqexpr_killer_metadata.json
```

当前 prototype 覆盖：

```text
5 quantity families:
  base10_bundle
  abacus_place
  rod_length_train
  area_tile
  graph_path_count

long expression:
  Div(Mul(Add(q1, q2), q3), Sub(q4, q5)) = 7

visual rule family:
  circuit_flow

verification command:
  python -m unittest tests.test_vqexpr -v
```

2026-06-21 fuzzy-visible 更新：

```text
metadata now includes:
  fuzzy_visual_atoms.value_decoding
  fuzzy_visual_atoms.port_binding
  visual_rule_program.alternative_binding_edges

presentation image now shows:
  pale/ghost quantity primitives for nearby possible values
  pale dashed alternative binding edges for lower-membership port assignments

added test:
  test_killer_sample_contains_visible_fuzzy_ambiguity_contract
```

这一步的判断：

```text
fuzzy 不能只是 fuzzy_value JSON 字段；
它必须在图像中表现为多个 plausible visual parses：
  nearby count/value parses
  competing port-binding parses
  later can add competing scope/order parses
```

当前 prototype 仍只是 research probe，不是完整 benchmark：

```text
not yet implemented:
  scalable random seed sampler
  full 8-way rendered candidate panels with local counterfactual images
  OCR/pixel-level leakage detector
  confound balancing optimizer
  Family-OOD/Length-OOD/Operator-OOD split generator
  model evaluation pipeline
```

但是它已经证明一件关键事情：

```text
不用阿拉伯数字，也不用 + - * / =，
我们可以用可控视觉数量属性 + visual rule graph 表示一个长 arithmetic expression。
```
