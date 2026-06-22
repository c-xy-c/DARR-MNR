# VQ-Expr 1-99 版本：Calibrated Fuzzy Attributes 与 Composition AoT Rule

日期：2026-06-21

状态说明：

```text
本文是 1-99 calibrated attribute 的细化设计文档。
当前主计划见 doc/plan.md：研究定义已经改为原生支持 1-99；
1-9 只能作为 smoke test 子集，不能作为数据集目标。
```

硬约束：

```text
holistic / analytical / analytical_part 不再作为 VQ-Expr rule。
它们属于 MNR visible-number arrangement ontology；
VQ-Expr 的 rule 是 calibration、quantity construction、visual-AoT binding、
operator gate、scope/execution 和 candidate-scoring semantics。
```

本文是对上一版 1-9 prototype 的重新收敛。新的目标是：

```text
1. 设计 4-5 种 fuzzy attributes，能够表示 1-99。
2. 设计 5 种 composition AoT rule，用 answer AoT 生成正样本和 negatives。
3. 让问题本身不要绑定到全局 attribute mapping：
   同一个属性值，例如 black，可以在一个 sample 中表示 99，
   在另一个 sample 中表示 0 或其他值。
```

核心变化：

```text
VQ-Expr 不再把 attribute 当作全局数字表。
每个 sample 有自己的 calibration context / local decoder。
模型必须从当前 sample 的 visual calibration 中恢复数值，再执行 AoT。
```

## 0. 设计原则

### 0.1 为什么可以回到 1-99

直接用点、格子、路径去表示 1-99 会太重；但如果每种 attribute 都有可视化的局部标定机制，
1-99 是可行的：

```text
value = local decoder(sample, attribute observation)
```

其中 decoder 不是数据集全局固定的。例如：

```text
sample A: black swatch -> digit 9
sample B: black swatch -> digit 0
```

这意味着黑色本身不携带数字含义；含义来自本题的 calibration context。

### 0.2 每个 sample 必须有三层

```text
Calibration layer:
  当前 sample 的 attribute-to-value decoder。

Quantity layer:
  visual object q 经 decoder 得到 fuzzy value distribution over 1..99。

Composition AoT layer:
  answer AoT 读取 quantity values，执行 arithmetic expression，生成 correct/negative choices。
```

形式化：

```text
phi_s^F: attribute observation -> fuzzy digit/value distribution
mu_q^s(n) = phi_s^F(obs(q))[n]
answer = Eval_AoT({mu_q^s})
```

`s` 是 sample id。没有 `s`，attribute 没有稳定语义。

## 1. 参考工作与借鉴点

| 领域 | 工作 | 借鉴点 | 对本方案的约束 |
| --- | --- | --- | --- |
| executable visual reasoning | [CLEVR](https://arxiv.org/abs/1612.06890) | 每个问题有 scene graph 和 executable functional program。 | VQ-Expr 必须保存 calibration graph、quantity graph、answer AoT，可复算 label。 |
| compositional scene graph QA | [GQA](https://arxiv.org/abs/1902.09506) | scene graph 生成 functional programs，并报告 consistency/grounding。 | 我们要报告 attribute grounding、decoder grounding、AoT consistency。 |
| program-guided visual generation | [Bongard-LOGO](https://arxiv.org/abs/2010.00763) | program-guided generation，且强调 context-dependent perception。 | attribute 不能全局绑定，必须有 sample-local interpretation。 |
| abstract visual reasoning | [RAVEN](https://arxiv.org/abs/1903.02741) / [PGM](https://arxiv.org/abs/1807.04225) | 结构化规则、关系推理和 generalization regimes。 | 我们要有 Attribute-OOD、Rule-OOD、Calibration-OOD、AoT-depth-OOD。 |
| fuzzy semantics | [Zadeh fuzzy sets](https://doi.org/10.1016/S0019-9958(65)90241-X) | membership in `[0,1]`。 | 每个 attribute decoder 输出 `mu(n)` 而不是 hard label。 |
| neural-symbolic fuzzy logic | [Logic Tensor Networks](https://arxiv.org/abs/1606.04422), [LTN AIJ](https://arxiv.org/abs/2012.13635) | many-valued first-order formulas / Real Logic。 | visual predicate、decoder confidence、edge binding 都进入 truth score。 |
| color fuzzy attributes | [fuzzy color naming dataset](https://www.cvc.uab.es/color_naming/), [parametric fuzzy color naming](https://opg.optica.org/abstract.cfm?uri=josaa-25-10-2582) | 颜色类别可建模为 fuzzy membership。 | color 可以做 attribute，但必须 sample-local calibration，不能全局 black=某数。 |
| numerosity confounds | [NASCO dot arrays](https://jnc.psychopen.eu/index.php/jnc/article/view/5893/5893.html), [numerosity visual-property database](https://pmc.ncbi.nlm.nih.gov/articles/PMC9840615/) | 数量判断受 area、size、convex hull、density 等影响。 | raw dots 不能直接做主 family；必须记录 visual confounds。 |
| long compositional rules | [ListOps/LRA](https://arxiv.org/abs/2011.04006), [CLUTRR](https://arxiv.org/abs/1908.06177) | 长树结构与 systematic generalization。 | AoT 要有 depth、tree-shape、held-out composition split。 |
| math program generation | [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset) | 程序化生成可验证数学题。 | answer AoT 与 negative AoT 都必须由 generator/verifier 产生。 |

## 2. 五种 1-99 Fuzzy Attribute

### 2.1 Local Place-Value Bundle

**表示范围：** 1-99。

视觉形式：

```text
object q = two compartments:
  high slot: tens component
  low slot: ones component

value(q) = 10 * digit(high) + digit(low)
```

关键不是固定 `tens/ones` 外观，而是每个 sample 的 calibration layer 给出：

```text
which local unit glyph corresponds to each digit 0..9
which slot is high / low
```

例如同样的黑色 bundle：

```text
sample A calibration: black -> 9
sample B calibration: black -> 0
```

Fuzzy decoder：

```text
mu_high(d) = membership that high-slot visual code is digit d
mu_low(d)  = membership that low-slot visual code is digit d

mu_q(n) = max_{10a+b=n} Tnorm(mu_high(a), mu_low(b))
```

为什么 symbolic-friendly：

```text
1-99 被拆成两个 digit-level symbolic variables；
AoT 可以读取 whole value，也可以诊断 tens/ones grounding；
可视化不需要 99 个物体。
```

Fuzzy 来源：

```text
slot boundary ambiguity；
digit-code ambiguity；
partial occlusion；
calibration swatch / token 相似度接近。
```

### 2.2 Calibrated Length Scale

**表示范围：** 1-99。

视觉形式：

```text
sample has a local ruler / axis with calibration marks.
quantity q is a rod or segment projected onto this local ruler.
```

重要的是 ruler 的 orientation、scale、zero/high direction 可以每题变化：

```text
sample A: darker endpoint is high
sample B: darker endpoint is low
sample C: axis is vertical or curved
```

Decoder：

```text
project q onto local scale axis
normalize by local calibration endpoints
mu_q(n) = membership around projected value n
```

Fuzzy 形式：

```text
mu_q(n) = exp(-(proj(q) - n)^2 / (2 sigma_scale^2))
```

为什么 symbolic-friendly：

```text
它是 continuous-to-symbolic discretization；
适合可视化 fuzzy membership；
可显示为局部尺度，而不是固定 pixel length。
```

### 2.3 Calibrated Area Grid

**表示范围：** 1-99。

视觉形式：

```text
sample has local unit-cell prototype and local grid orientation.
quantity q is an area patch / polyomino.
value(q) = number of local unit cells covered.
```

它不能用全局 cell size：

```text
sample A: small cell prototype
sample B: large cell prototype
sample C: skewed cell prototype
```

Decoder：

```text
mu_cell_i(filled) from patch overlap with local cell lattice
mu_q(n) = membership that exactly n local cells are filled
```

近似实现：

```text
mean_count = sum_i mu_cell_i
sigma = boundary_uncertainty + grid_alignment_uncertainty
mu_q(n) = exp(-(n - mean_count)^2 / (2 sigma^2))
```

为什么 symbolic-friendly：

```text
area 与 multiplication/division 天然相关；
1-99 可以用 9x11、10x10 以内局部 patch；
局部 grid 让属性不绑定全局 pixel area。
```

### 2.4 Chunked Path / Topological Count

**表示范围：** 1-99。

视觉形式：

```text
value = 10 * number_of_chunks + residual_edges
```

其中 chunk 的视觉符号由 sample calibration 决定，不是固定形状：

```text
sample A: thick arc means a 10-edge chunk
sample B: dotted capsule means a 10-edge chunk
sample C: same thick arc may mean residual group, not chunk
```

Decoder：

```text
mu_chunk(k) = membership that q contains k calibrated chunks
mu_residual(r) = membership that q contains r residual edges
mu_q(n) = max_{10k+r=n} Tnorm(mu_chunk(k), mu_residual(r))
```

Fuzzy 来源：

```text
branch ambiguity；
chunk boundary uncertainty；
edge gaps；
route crossing；
alternative path membership。
```

为什么 symbolic-friendly：

```text
它把 visual topology 与 numeric value 联动；
非常适合测试 long-range binding；
也能和 expression graph 形成双层图结构。
```

### 2.5 Local Color / Texture Codebook

**表示范围：** 1-99。

视觉形式：

```text
sample contains a local codebook:
  color/texture token -> digit 0..9

quantity q has two code tokens:
  high token -> tens digit
  low token  -> ones digit
```

关键约束：

```text
颜色/纹理没有全局数字意义。
同一个 black token 在 sample A 可以映射到 9，
在 sample B 可以映射到 0。
```

如何让模型知道 mapping：

```text
calibration context 显示 code token 与非颜色数量属性的对应关系。
例如:
  black swatch paired with 9-unit local grouped-units prototype in sample A;
  black swatch paired with empty/zero marker in sample B.
```

Decoder：

```text
mu_digit(d | token, codebook_s)
mu_q(n) = max_{10a+b=n} Tnorm(mu_digit(a|high), mu_digit(b|low))
```

Fuzzy 来源：

```text
hue boundary；
texture mixture；
codebook swatch ambiguity；
high/low token boundary ambiguity。
```

为什么保留它：

```text
颜色是最自然的 fuzzy category；
但必须通过 local codebook 避免全局 attribute shortcut。
```

## 3. Attribute 不绑定问题：Local Calibration Contract

每个 sample 必须有：

```json
{
  "calibration_context": {
    "sample_id": "s_000001",
    "decoders": {
      "color_texture_codebook": {
        "black": {"digit": 9, "membership": {"8": 0.21, "9": 1.0}},
        "blue": {"digit": 0, "membership": {"0": 1.0, "1": 0.18}}
      },
      "length_scale": {
        "low_endpoint": "marker_a",
        "high_endpoint": "marker_b",
        "orientation": "right_to_left"
      }
    }
  }
}
```

同一个 attribute observation 在不同 sample 中可以有不同 decoder：

```text
mu_q^sampleA(99 | black-black) = high
mu_q^sampleB(0  | black-black) = high
```

注意：0 可以作为 digit component 出现，但 whole quantity values 用 1-99。

### 3.1 为什么这很重要

如果没有 local calibration：

```text
模型可以学全局 shortcut：black -> 9, long -> large, red -> 3。
```

有了 local calibration：

```text
模型必须读取当前 sample 的 decoder，再把 decoder 绑定到 quantity object。
```

这就是新的 hidden variable：

```text
Attribute-Calibration Binding
```

## 4. 五种 Composition AoT Rule

旧 MNR 的 holistic/analytical 不能用。VQ-Expr 的 rule 都应该是 composition rule，
也就是由多个 decode / operator / binding / scope 节点组合成 answer AoT。

### Rule 1: Serial Composition

结构：

```text
T = Op3(Op2(Op1(q1, q2), q3), q4)
```

用途：

```text
测试顺序执行、edge flow、intermediate value。
```

可视化：

```text
pipeline / circuit-flow / path-transformation
```

Negative 重点：

```text
operator swap, edge rewire, output perturbation
```

### Rule 2: Parallel Composition

结构：

```text
T_left  = Op1(q1, q2)
T_right = Op2(q3, q4)
T       = Op3(T_left, T_right)
```

用途：

```text
测试两个分支的并行计算和最终合成。
```

可视化：

```text
two lanes merge into a final gate
```

Negative 重点：

```text
branch swap, gate swap, cross-branch binding
```

### Rule 3: Nested Scope Composition

结构：

```text
T = Op1(q1, Op2(q2, Op3(q3, q4)))
```

或：

```text
T = Op1(Op2(q1, q2), Op3(q3, q4))
```

用途：

```text
测试 visual scope / container / nesting 是否决定 AoT tree。
```

可视化：

```text
nested machines, trays, containers, brackets without text
```

Negative 重点：

```text
scope rotation:
  Op1(Op2(a,b),c) -> Op1(a,Op2(b,c))
```

### Rule 4: Inverse Composition

结构：

```text
T = Div(Mul(Op1(q1, q2), q3), Op2(q4, q5))
```

或：

```text
T = Sub(Add(Op1(q1,q2), q3), q4)
```

用途：

```text
在 1-99 范围内构造长表达式，同时保持 final answer 合法。
```

可视化：

```text
repeater gate + splitter gate,
additive tray + cancellation tray
```

Negative 重点：

```text
port swap for Div/Sub,
fuzzy ambiguity trap,
operator inverse mismatch
```

### Rule 5: Calibration-Composition

结构：

```text
T = Op(
      Decode(q1, decoder_s^A),
      Decode(q2, decoder_s^B)
    )
```

或者更长：

```text
T = Op2(
      Op1(Decode(q1, color_codebook_s), Decode(q2, length_scale_s)),
      Decode(q3, area_grid_s)
    )
```

用途：

```text
测试 attribute 不绑定问题。
同一个 observation 在不同 sample 里由不同 decoder 解释。
```

可视化：

```text
calibration panel / local legend / local scale / codebook
```

Negative 重点：

```text
wrong decoder,
calibration swap,
global attribute shortcut trap
```

## 5. Answer AoT 如何生成正负样本

每个 sample 有一个 answer AoT：

```text
A* = Compose(
       Decode nodes,
       Visual-binding nodes,
       Arithmetic operator nodes,
       Scope/order nodes
     )
```

生成流程：

```text
1. sample a composition rule family R in {Serial, Parallel, Nested, Inverse, Calibration}.
2. sample answer AoT tree shape T*.
3. sample local calibration decoders phi_s.
4. reverse-sample leaf latent values so final answer y* in 1..99.
5. render quantity objects under phi_s.
6. render prompt/context visual rule.
7. render correct candidate C0 as quantity carrier for y*.
8. render C1..C7 by applying one counterfactual mutation each.
9. verifier recomputes crisp/fuzzy score for every candidate.
```

### 5.1 Correct choice

```text
C0:
  uses same calibration context phi_s
  encodes y* as candidate quantity
  satisfies answer AoT A*
```

### 5.2 Seven negative choices

| Choice | Counterfactual | Mutated layer | What it tests |
| --- | --- | --- | --- |
| C1 | Output perturbation | answer quantity | Can model decode final value? |
| C2 | Leaf decode perturbation | one quantity attribute | Can model decode input attribute? |
| C3 | Calibration swap | local decoder | Does model use sample-specific mapping? |
| C4 | Operator swap | gate semantics | Does model parse operator gates? |
| C5 | Port/binding swap | edge/port binding | Does model respect non-commutative input order? |
| C6 | Scope/tree rotation | AoT structure | Does model recover visual scope? |
| C7 | Fuzzy ambiguity trap | membership/truth score | Does model propagate fuzzy uncertainty instead of hard parse? |

Each negative must satisfy:

```text
exactly one primary mutation;
candidate visual statistics balanced where possible;
candidate value unique;
fuzzy score below correct by margin;
metadata logs before/after AoT.
```

### 5.3 Candidate score

```text
score(C_i) =
  Tnorm(
    candidate_value_match,
    calibration_consistency,
    attribute_decode_membership,
    gate_membership,
    binding_membership,
    scope_membership
  )
```

Correctness:

```text
correct = argmax_i score(C_i)
valid iff:
  score(C0) >= theta_pos
  score(C0) - max_{i>0} score(C_i) >= delta
```

## 6. 为什么这不等于 attribute lookup

如果一个属性有全局意义：

```text
black -> 9
blue -> 0
long -> large
```

模型可以只学 attribute shortcut。

VQ-Expr 的 contract 是：

```text
black has no dataset-level numeric meaning.
black only has meaning under the current sample decoder phi_s.
```

因此，最重要的 split 是：

```text
Calibration-OOD:
  train: black often maps to high digit
  test: black often maps to low digit
```

如果模型在这里失败，就说明它学的是全局 attribute shortcut，而不是 sample-local calibration binding。

## 7. 可视化建议

### 7.1 Presentation view

只能显示：

```text
calibration context
quantity carriers
visual operator graph
candidate quantity carriers
```

不能显示：

```text
Arabic numerals
operator symbols
debug labels
decoder table in text
answer index
```

### 7.2 Debug view

可以显示：

```text
local decoder phi_s
mu_q(n) top-k
answer AoT
mutation log for negatives
candidate scores
```

### 7.3 最适合 symbolic 展现的属性

第一优先级：

```text
Local Place-Value Bundle
Calibrated Area Grid
Chunked Path / Topological Count
```

原因：

```text
它们有清楚的 symbolic decomposition。
它们可以画成 scene graph。
它们能生成 explainable metadata。
```

第二优先级：

```text
Calibrated Length Scale
Local Color / Texture Codebook
```

原因：

```text
它们 fuzzy 感更强，但必须依赖 calibration context 避免 shortcut。
```

## 8. Yangshi-style proxy split

| Item | Definition |
| --- | --- |
| Proxy A | 模型能识别全局 attribute-value shortcut，例如黑色总是大数、长条总是大数。 |
| Construct B | 模型能读取 sample-local calibration，把 fuzzy attributes 绑定到 AoT leaf values，并执行 composition rule。 |
| Regime R | 同一 visual attribute 在不同 sample 中映射到不同数值；相同 quantity multiset 在不同 AoT rule 中得到不同答案。 |
| Mechanism H | Calibration-AoT Binding Collapse：模型把 attribute decode、local calibration 和 AoT execution 分开 harden 或 shortcut。 |
| Artifact O | 5 类 calibrated fuzzy attributes + 5 类 composition AoT rules + answer-AoT counterfactual candidate generator。 |

一句话 thesis：

```text
VQ-Expr evaluates whether models can bind sample-local fuzzy visual attributes to arithmetic composition trees,
rather than exploiting globally fixed attribute-to-number shortcuts.
```

## 9. 下一步实现

不要先扩当前 1-9 prototype。下一步应该先实现：

```text
1. calibration_context schema
2. local decoder phi_s for each attribute family
3. answer AoT schema with Decode nodes and Arithmetic nodes
4. five composition rule samplers
5. C0-C7 candidate mutation operators
```

最小 acceptance tests：

```text
test_same_color_maps_to_different_digits_across_samples
test_each_attribute_family_represents_1_to_99_with_local_decoder
test_answer_aot_recomputes_correct_candidate
test_each_negative_has_single_primary_mutation
test_calibration_ood_breaks_global_attribute_shortcut
```
