# 顶会 Benchmark 对齐调研与 AFVNB-MNR 重构方案

日期：2026-06-18

本文档记录一次从顶会 benchmark 标准出发，对 FVNB-MNR 的重新推敲。核心修正是：**数字不应该像贴纸一样放在几何图上，再被外部 rule 读取；数字应该是视觉对象的 attribute value，和颜色、大小、位置、区域隶属度一样属于 scene graph 的一部分。** 这会把任务从“数字锚点 + fuzzy role”提升为“对象属性 + fuzzy predicate + executable program”的 benchmark。

## 0. 结论先行

上一版 FVNB-MNR 的方向是对的：

```text
visual structure -> role membership -> fuzzy rule truth
```

但它仍有一个 ontology 问题：数字被建模成 `anchor` 上的 glyph，而不是对象自身的属性。于是图像看起来像“几何背景 + 数字贴片”，rule 看起来像外部算式，而不是 scene graph 内部的属性关系。

新的 benchmark 研究对象应改成：

```text
object o_i:
  geometry attributes: position, region, boundary distance, containment relation
  visual attributes: shape, fill, outline, size
  numeric attribute: value(o_i)
  fuzzy predicates: mu_outer(o_i), mu_inner(o_i), mu_boundary(o_i), ...

rule program:
  exists distinct x, y, z:
    Outer(x) AND Inner(y) AND Boundary(z)
    AND Eq(value(x) - value(y), value(z))
```

建议把下一版称为：

```text
AFVNB-MNR: Attribute-Fuzzy Visual-Number Binding for Machine Number Reasoning
```

其中 `Attribute-Fuzzy` 比单纯 `Fuzzy` 更准确：数字不是“图中可 OCR 的文本”，而是参与 fuzzy first-order rule 的对象属性。

## 1. 当前设计的问题诊断

### 1.1 表面问题：可视化像 debug 图

上一版可视化里，数字 token、边框、分数、negative type 全部挤在一个 panel grid 中。它可以用于调试，但不适合做 benchmark teaser。这个问题能通过更好的 rendering 缓解，但这不是根因。

### 1.2 根本问题：数字与对象 ontology 没有融合

当前代码里的核心对象接近：

```text
anchor = {
  position,
  value,
  membership
}
```

这会让 `value` 像附着在位置上的 OCR 文本。更合理的 benchmark ontology 应是：

```text
object = {
  id,
  geometry,
  numeric_value,
  visual_attributes,
  fuzzy_predicates
}
```

也就是说，数字不是独立的视觉元素，而是对象的一项 attribute；rule 不是“读三个位置的数字再算”，而是“选择满足 fuzzy predicate 的对象，再比较它们的 numeric attribute”。

### 1.3 Yangshi proxy split

| 项 | 旧 FVNB 表述 | 修正后的 AFVNB 表述 |
| --- | --- | --- |
| Proxy A | 模型识别数字 glyph，并把位置/角色 membership 送进算式 | 模型识别对象上的数字文本，并把它当作外部可读符号 |
| Construct B | 模型理解视觉结构对算式角色的 fuzzy 影响 | 模型理解对象属性系统：numeric value 是对象属性，fuzzy role 是对象谓词，rule 是属性-谓词联合 program |
| Regime R | 同数字/同坐标/hard role 不变但 membership 变化 | 同一组对象的 numeric attributes 可不变，但对象谓词 membership 改变；或谓词不变但 attribute assignment 改变 |
| Mechanism H | fuzzy rule binding collapse | attribute-predicate binding collapse |
| Minimal artifact O | fuzzy role counterfactual dataset | executable object-attribute fuzzy program benchmark |

更强的机制名建议：

```text
Attribute-Predicate Binding Collapse:
models decouple numeric attributes from the fuzzy visual predicates that define which objects the rule quantifies over.
```

## 2. 顶会 Benchmark 标准：我们应该学什么

### 2.1 CLEVR：diagnostic benchmark 的底线是 executable semantic program

CLEVR 的关键不是合成图像，而是每个问题都有 scene graph 和 functional program；program 可以在 scene graph 上执行得到答案。它还强调减少数据偏置，避免模型不用推理也答对。

对 AFVNB-MNR 的要求：

```text
每个样本必须有 object-level scene graph；
每个 rule 必须是 executable fuzzy program；
每个答案必须能由 program 在 scene graph 上复算；
metadata 要能支持按 reasoning type 分解错误。
```

不能只保存：

```text
image + answer index + truth score
```

必须保存：

```text
objects, attributes, fuzzy predicates, program, assignments, counterfactual operator
```

来源：[CLEVR CVPR 2017](https://arxiv.org/abs/1612.06890), [CLEVR project](https://cs.stanford.edu/people/jcjohns/clevr/)

### 2.2 GQA：真实视觉推理 benchmark 的价值在 semantic control 和 metrics

GQA 用 scene graph 生成 compositional questions，并引入 consistency、grounding、plausibility 等指标。它的启发是：benchmark 不能只看 accuracy，必须能说明模型错在哪个 semantic axis。

对 AFVNB-MNR 的要求：

```text
除了 8-way accuracy，还要报告：
  attribute binding accuracy,
  predicate grounding accuracy,
  program consistency,
  counterfactual consistency,
  shortcut reliance rate.
```

其中 `attribute binding accuracy` 应回答：

```text
模型是否把 object 的 numeric_value 与正确 fuzzy predicate 绑定？
```

来源：[GQA CVPR 2019](https://arxiv.org/abs/1902.09506), [GQA project](https://cs.stanford.edu/people/dorarad/gqa/about.html)

### 2.3 RAVEN / PGM：结构推理必须有明确的 generalization regimes

RAVEN 把视觉结构表示引入 RPM 风格推理。PGM 强调不同 generalization regimes，训练和测试在 rule、attribute、object combinations 上系统不同。RAVEN-FAIR 进一步指出候选构造如果不平衡，会让模型只看候选也能答题。

对 AFVNB-MNR 的要求：

```text
必须设计不止 IID split：
  Attribute-OOD: numeric value range 或 value distribution 未见过；
  Predicate-OOD: membership band/tau/sigma 未见过；
  Program-OOD: rule template 未见过；
  Family-OOD: containment -> partition/set/topology；
  Candidate-bias audit: candidate-only 必须接近随机。
```

候选不能随机生成，必须是 controlled near-miss，每个负例只破坏一个 semantic constraint。

来源：[RAVEN CVPR 2019](https://arxiv.org/abs/1903.02741), [PGM ICML 2018](https://proceedings.mlr.press/v80/barrett18a.html), [RAVEN-FAIR CVPR 2021](https://arxiv.org/abs/2009.09405)

### 2.4 Bongard-LOGO：程序生成要保持人类可解释概念

Bongard-LOGO 的价值在于 program-guided generation，同时保持 concept 对人类可解释。它也强调 context-dependent perception：同一视觉元素在不同上下文中解释不同。

对 AFVNB-MNR 的要求：

```text
rule program 必须足够短，能由人类读懂；
visual family 必须让同一 numeric attribute 在不同 predicate context 中改变作用；
不要用装饰复杂度替代概念复杂度。
```

这直接支持用户的判断：不要加奇怪装饰。我们需要的是概念结构自然，不是画面复杂。

来源：[Bongard-LOGO NeurIPS 2020](https://arxiv.org/abs/2010.00763)

### 2.5 PTR：对象-部件层级说明 attribute schema 可以丰富，但必须受控

PTR 将视觉场景解析为对象和部件，并提供 part-level annotations、attributes、relations 和多种 reasoning question。它说明：如果 benchmark 的 claims 是“结构化视觉推理”，metadata 必须跟上。

对 AFVNB-MNR 的要求：

```text
object hierarchy 可以从 simple objects 开始：
  container
  attribute token
  boundary band
  region cell

但 metadata 必须明确：
  which object carries numeric_value,
  which region predicates apply,
  which object variables are selected by the program.
```

来源：[PTR NeurIPS 2021](https://arxiv.org/abs/2112.05136)

## 3. 视觉数学 Benchmark 标准：我们的定位不能跑偏

### 3.1 MathVista / MATH-Vision：广覆盖不是我们的优势

MathVista 和 MATH-Vision 的路线是 broad coverage：多数学科、多视觉上下文、真实竞赛题或混合来源。它们适合评估 general-purpose LMM，但不适合隔离一个机制变量。

对 AFVNB-MNR 的定位：

```text
不和 MathVista/MATH-Vision 比题量、真实度、学科覆盖；
我们只 claim 可控机制诊断：
  fuzzy predicate 与 numeric attribute 的绑定。
```

来源：[MathVista ICLR 2024](https://arxiv.org/abs/2310.02255), [MATH-Vision NeurIPS 2024 D&B](https://arxiv.org/abs/2402.14804)

### 3.2 MathVerse：视觉依赖必须用信息版本或反事实证明

MathVerse 的关键是把同一视觉数学题改写成不同信息含量版本，问模型是否真的看图。

对 AFVNB-MNR 的要求：

```text
每个 seed program 应产生至少三种 paired variants：
  Full visual: image contains object attributes and fuzzy predicates；
  Attribute-only: value attributes exposed but predicate geometry removed；
  Predicate-only: geometry exposed but numeric attributes masked；
  Counterfactual: same objects/values but predicate memberships changed。
```

这样才能证明模型不是只读数字，也不是只读几何。

来源：[MathVerse ECCV 2024](https://arxiv.org/abs/2403.14624)

### 3.3 DynaMath：seed program 和 dynamic variants 是 robustness 证据

DynaMath 把每个 seed question 表示成 Python program，能自动生成视觉和文本变体，并报告 robustness。

对 AFVNB-MNR 的要求：

```text
一个 benchmark unit 不应是单张题；
而应是一个 seed program family：
  seed_id
  object schema
  fuzzy predicate functions
  numeric attribute sampler
  counterfactual operators
  deterministic oracle
```

指标上应加入：

```text
seed worst-case accuracy:
  a seed is solved only if all variants are solved.
```

来源：[DynaMath 2024](https://arxiv.org/abs/2411.00836)

### 3.4 VC-Bench / VisuLogic：显式视觉依赖与 vision-centric reasoning

VC-Bench 明确提出 explicit visual dependency；VisuLogic 关注避免文本 shortcut 的 vision-centric reasoning。

对 AFVNB-MNR 的要求：

```text
不要让自然语言描述泄露 rule；
图像必须携带不可由文字/数字列表替代的 predicate membership；
metadata 可以有 program，但 evaluation 输入应保持 image-only 或 minimal-language。
```

特别要避免：

```text
题干写出 outer/inner/boundary；
candidate label 泄露 negative type；
数字列表足以解题。
```

来源：[VC-Bench 2025](https://arxiv.org/html/2504.18589v1), [VisuLogic 2025](https://arxiv.org/abs/2504.15279)

### 3.5 VisioMath / TACIT：图像候选和 near-miss 必须结构化

VisioMath 强调 image-based answer choices 和 visually similar candidates。TACIT 更进一步：near-miss distractor 每个只违反一个结构约束，并可由 deterministic verifier 检查。

对 AFVNB-MNR 的要求：

```text
8 个候选都应该是图像；
每个 negative 必须对应一个 counterfactual operator；
每个 negative 只违反一个 semantic predicate 或 attribute binding；
候选之间视觉复杂度、值分布、位置分布要平衡。
```

来源：[VisioMath 2025 / ICLR 2026](https://arxiv.org/abs/2506.06727), [TACIT 2026](https://arxiv.org/abs/2603.00206)

## 4. Fuzzy / Neuro-symbolic 标准：rule 应该是什么形式

### 4.1 Fuzzy set 的核心是 membership，不是视觉模糊装饰

Zadeh 的 fuzzy set 定义强调对象属于集合的程度。对我们来说，`Outer(o)`、`Boundary(o)`、`Inside(o)` 都应是对象谓词，返回 `[0,1]`。

正确迁移：

```text
mu_boundary(o) = degree that object o belongs to boundary region
```

错误迁移：

```text
把边框画模糊；
把数字画灰一点；
把图像加噪声。
```

来源：[Zadeh 1965 Fuzzy Sets](https://doi.org/10.1016/S0019-9958(65)90241-X)

### 4.2 LTN / Real Logic：first-order variables 应绑定到对象

Logic Tensor Networks 的 Real Logic 思路是：一阶逻辑公式在真实向量域上求 `[0,1]` truth value。对 AFVNB-MNR 最重要的是 first-order language：

```text
objects are variables;
attributes are functions;
relations/predicates return truth degrees.
```

所以我们的 rule 不应写成：

```text
outer - inner = boundary
```

而应写成：

```text
exists x, y, z in Objects:
  Outer(x) AND Inner(y) AND Boundary(z)
  AND Eq(Value(x) - Value(y), Value(z))
```

来源：[Logic Tensor Networks](https://arxiv.org/abs/1606.04422), [LTN 2020](https://arxiv.org/abs/2012.13635)

## 5. 新 Benchmark 定义：AFVNB-MNR

### 5.1 单个 panel 的 scene graph

一个 panel 应保存：

```json
{
  "objects": [
    {
      "id": "o1",
      "class": "value_token",
      "numeric_value": 7,
      "geometry": {
        "center": [x, y],
        "radius": r,
        "signed_distance_outer": 14.0,
        "signed_distance_inner": -8.0,
        "distance_to_boundary": 8.0
      },
      "visual_attributes": {
        "shape": "circle",
        "fill": "white",
        "outline": "black"
      },
      "fuzzy_predicates": {
        "Outer": 0.97,
        "Inner": 0.01,
        "Boundary": 0.02,
        "Outside": 0.00
      }
    }
  ],
  "regions": [
    {"id": "R_outer", "class": "container"},
    {"id": "R_inner", "class": "subcontainer"},
    {"id": "R_boundary", "class": "fuzzy_band"}
  ]
}
```

这里 `numeric_value` 是对象属性；数字 glyph 只是这个属性的可视化方式。

### 5.2 Rule program

正式 rule schema：

```text
Program Diff:
  variables: x, y, z over value_token objects
  constraints:
    distinct(x, y, z)
    Outer(x)
    Inner(y)
    Boundary(z)
    Equal(Value(x) - Value(y), Value(z))
  truth:
    max_{x,y,z distinct} Tnorm(
      Outer(x),
      Inner(y),
      Boundary(z),
      Equal(Value(x) - Value(y), Value(z))
    )
```

其他 schema：

```text
Program Sum:
  Outer(x) AND Inner(y) AND Boundary(z)
  AND Equal(Value(x), Value(y) + Value(z))

Program Ratio:
  Outer(x) AND Inner(y) AND Boundary(z)
  AND Equal(Value(x) / Value(y), Value(z))
```

### 5.3 视觉设计原则

用户说“不要奇怪装饰”是对的。视觉风格应该遵循 minimal semantic rendering：

| 视觉元素 | 是否允许 | 原因 |
| --- | --- | --- |
| value token | 允许 | object carrier，承载 numeric attribute |
| container / region boundary | 允许 | 定义 fuzzy predicates |
| boundary band | 允许 | 表示 fuzzy membership 的语义区域 |
| 阴影、渐变、装饰色 | 不允许 | 不改变 program，只增加视觉噪声 |
| 多颜色 | 只有成为 attribute 才允许 | 如果颜色不进 rule，就不应出现 |
| 文本题干 | v0 不允许 | 避免语言 shortcut |
| 候选 negative type overlay | evaluation 图不允许，debug 图可允许 | 不应泄露答案 |

## 6. Counterfactual operators

每个 negative 应由一个 operator 产生：

| Operator | 保持不变 | 改变 | 打击 shortcut |
| --- | --- | --- | --- |
| Value-only perturbation | fuzzy predicates | `numeric_value` | arithmetic-only 或 value-only failure |
| Predicate-only perturbation | object values | object region / predicate membership | number-only shortcut |
| Attribute-predicate swap | value multiset | value-object binding | bag-of-values shortcut |
| Hard-predicate invariant fuzzy flip | hard predicate argmax | predicate membership strength | crisp parser shortcut |
| Program near-miss | all but one predicate | one program atom | random negative bias |
| Candidate-only decoy | candidate appearance distribution | context-rule consistency | RAVEN-FAIR style candidate bias |

每个负例只能破坏一个主要约束，否则错误不可归因。

## 7. Metrics

建议指标：

| Metric | 定义 | 对齐标准 |
| --- | --- | --- |
| AFRA | Attribute-Fuzzy Rule Accuracy | main score |
| APCC | Attribute-Predicate Counterfactual Consistency | MathVerse/DynaMath paired variant |
| AGA | Attribute Grounding Accuracy | GQA grounding |
| PGA | Predicate Grounding Accuracy | object predicate grounding |
| HG | Hardening Gap | fuzzy vs crisp parser |
| VDI | Visual Dependency Index | MathVerse/VC-Bench visual reliance |
| SRR | Shortcut Reliance Rate | RAVEN-FAIR candidate bias |
| Seed-Worst Accuracy | seed solved iff all variants solved | DynaMath robustness |

## 8. Splits

| Split | Train/Test 关系 | 测什么 |
| --- | --- | --- |
| IID | same family/rule/value range | basic learnability |
| Attribute-OOD | new value range or digit distribution | 是否学到 value function |
| Predicate-OOD | new fuzzy band/tau/sigma/region scale | 是否学到 predicate membership |
| Program-OOD | new formula composition | 是否把 predicate 和 value 解耦 |
| Family-OOD | containment -> partition/set/topology | 是否迁移 object-attribute principle |
| Style-OOD | same metadata, different minimal renderer | 是否依赖渲染纹理 |
| Counterfactual Stress | paired interventions | 是否真正跟随 program truth |

## 9. 第一版应该怎么改代码

当前 `fvnb_generator.py` 不应继续只加视觉细节，而应改 schema：

```text
Anchor -> Object
value -> numeric_value attribute
membership -> fuzzy_predicates
truth_assignment -> object variable assignment
rule -> executable program with variables/functions/predicates
negative_type -> counterfactual_operator
```

建议新增模块：

```text
mnr_dataset/afvnb_scene.py
  AFVNBObject, AFVNBRegion, AFVNBScene

mnr_dataset/afvnb_program.py
  FuzzyProgram, Atom, AttributeFunction, Predicate

mnr_dataset/afvnb_generator.py
  seed program family + counterfactual variants

mnr_dataset/afvnb_visualize.py
  presentation view and debug view split
```

旧 `fvnb_*` 可以保留为 v0 proof-of-concept；新 `afvnb_*` 做顶会对齐版本。

## 10. Teaser 应该长什么样

不要展示“数字贴在图上”。第一页 teaser 应展示：

```text
Scene graph view:
  objects with numeric_value attributes
  fuzzy predicate memberships

Program view:
  exists x,y,z:
    Outer(x) AND Inner(y) AND Boundary(z)
    AND Value(x)-Value(y)=Value(z)

Counterfactual view:
  same values, same objects, different predicate membership
  -> program truth changes
```

视觉上应该像 CLEVR/GQA 的 program figure，而不是普通 puzzle grid。

## 11. Claim-evidence ledger

| Claim | 当前状态 | 允许写法 | 需要证据 |
| --- | --- | --- | --- |
| 当前 FVNB 可证明 fuzzy role truth，但 numeric value 仍像外部 glyph | 代码支持 | 可以写在内部文档 | object schema audit |
| 顶会 diagnostic benchmark 通常需要 executable program/scene graph | 文献支持 | 可以写 | CLEVR/GQA/RAVEN/PGM references |
| AFVNB-MNR 定义新的 attribute-predicate binding object | 设计 claim | 可以说 designed to | program/oracle implementation |
| 模型存在 attribute-predicate binding collapse | hypothesis | 不能强说 | shortcut baselines + strong models |
| AFVNB 比 FVNB 更顶会 | conditional | 不能直接说 | reviewer-facing evidence stack |

## 12. 下一步一周计划

```text
Day 1:
  写 AFVNB object/program schema；
  把 current anchor metadata 映射成 object metadata。

Day 2:
  实现 executable fuzzy first-order program；
  rule assignment 返回 object ids，而不是 role names。

Day 3:
  改 generator：counterfactual operators 基于 object attribute/predicate intervention。

Day 4:
  改 visualization：presentation view 不显示 negative type；debug view 才显示。

Day 5:
  生成 1k AFVNB probe；
  跑 number-only、attribute-only、predicate-only、hard-program baselines。

Day 6:
  做 candidate-only audit 和 paired counterfactual consistency。

Day 7:
  更新 doc/plan.md 和 paper story；
  画 teaser：object-attribute scene graph + executable program + counterfactual.
```

## 13. 当前成熟度

```text
Maturity level:
  L2/L3 between diagnostic design and mechanism hypothesis.

One-sentence thesis:
  Although visual mathematics benchmarks increasingly test whether models use diagrams,
  current abstract visual arithmetic can still treat numbers as external glyphs rather than object attributes;
  AFVNB-MNR exposes this attribute-predicate binding gap through executable fuzzy scene-graph programs and controlled counterfactuals.

Evidence held:
  repo-level FVNB prototype;
  literature-backed benchmark standards;
  clear ontology flaw in current design.

Evidence missing:
  AFVNB executable program implementation;
  human-readable teaser;
  candidate-only and strong-model baselines;
  OOD split results.

Forbidden claims:
  “AFVNB proves models lack human-like reasoning.”
  “AFVNB is harder than MathVista/MNR.”
  “AFVNB is the first visual math benchmark with fuzzy logic.”

Next action:
  implement AFVNB object/program schema and regenerate one clean seed family.
```

## 14. References

- [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning, CVPR 2017](https://arxiv.org/abs/1612.06890)
- [GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering, CVPR 2019](https://arxiv.org/abs/1902.09506)
- [RAVEN: A Dataset for Relational and Analogical Visual rEasoNing, CVPR 2019](https://arxiv.org/abs/1903.02741)
- [Measuring abstract reasoning in neural networks / PGM, ICML 2018](https://proceedings.mlr.press/v80/barrett18a.html)
- [RAVEN-FAIR / Scale-Localized Abstract Reasoning, CVPR 2021](https://arxiv.org/abs/2009.09405)
- [Bongard-LOGO, NeurIPS 2020](https://arxiv.org/abs/2010.00763)
- [PTR, NeurIPS 2021](https://arxiv.org/abs/2112.05136)
- [MathVista, ICLR 2024](https://arxiv.org/abs/2310.02255)
- [MathVerse, ECCV 2024](https://arxiv.org/abs/2403.14624)
- [MATH-Vision, NeurIPS 2024 Datasets and Benchmarks](https://arxiv.org/abs/2402.14804)
- [DynaMath, 2024](https://arxiv.org/abs/2411.00836)
- [VC-Bench, 2025](https://arxiv.org/html/2504.18589v1)
- [VisuLogic, 2025](https://arxiv.org/abs/2504.15279)
- [VisioMath, 2025 / ICLR 2026](https://arxiv.org/abs/2506.06727)
- [TACIT, 2026](https://arxiv.org/abs/2603.00206)
- [Zadeh, Fuzzy Sets, 1965](https://doi.org/10.1016/S0019-9958(65)90241-X)
- [Logic Tensor Networks, 2016](https://arxiv.org/abs/1606.04422)
- [Logic Tensor Networks, 2020](https://arxiv.org/abs/2012.13635)
