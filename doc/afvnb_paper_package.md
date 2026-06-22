# AFVNB-MNR Paper Package

日期：2026-06-18

本文档把 AFVNB-MNR 包装成一篇顶会 benchmark/evaluation paper 的叙事骨架。它不是最终论文文本；它是写作边界、贡献边界、图表边界和 claim-evidence 边界。

核心约束：

```text
Every paper surface must repeat the same mechanism:
Attribute-Predicate Binding Collapse.

不要把论文写成：
  We build a harder visual math dataset.

要把论文写成：
  Existing abstract visual arithmetic can treat numeric symbols as external glyphs;
  AFVNB makes numeric values object attributes and tests whether fuzzy visual predicates
  are propagated into executable arithmetic rule truth.
```

## 1. Paper Vehicle

推荐 paper vehicle：

```text
Benchmark / evaluation standard paper
```

次级 vehicle：

```text
Conceptual reframing + reusable diagnostic artifact
```

不推荐 vehicle：

```text
Method paper
SOTA visual math leaderboard paper
Human cognition paper
```

理由：AFVNB 的核心贡献不是一个新模型，而是定义一个缺失的评价对象：`numeric_value` object attributes 与 fuzzy visual predicates 的绑定是否进入 rule truth。

## 2. Title Options

推荐标题：

1. `AFVNB-MNR: A Benchmark for Attribute-Fuzzy Visual-Number Binding in Abstract Arithmetic`
2. `Decoupling Numeric Attributes from Visual Predicates in Machine Number Reasoning`
3. `When the Same Numbers Mean Different Rules: Attribute-Predicate Binding in Visual Arithmetic`

更强但需证据后再用：

4. `Attribute-Predicate Binding Collapse in Visual Arithmetic Reasoning`
5. `A Fuzzy Scene-Graph Benchmark for Visual-Number Rule Binding`

不推荐标题：

```text
A Harder Dataset for Visual Math
Fuzzy Logic for Better Visual Reasoning
Improving Machine Number Reasoning with Fuzzy Rules
```

问题：这些标题要么只强调难度，要么像方法改进，要么把 fuzzy logic 当装饰词。

## 3. One-Sentence Thesis

当前安全版本：

```text
Although visual mathematics and abstract reasoning benchmarks increasingly test whether models use diagrams,
current abstract visual arithmetic can still treat numbers as external glyphs read from fixed visual templates;
AFVNB-MNR defines attribute-predicate binding as a controlled diagnostic object by representing numeric values as object attributes,
visual roles as fuzzy predicates, and labels as executable fuzzy scene-graph programs.
```

实验完成后的更强版本：

```text
Although visual mathematics and abstract reasoning benchmarks increasingly test whether models use diagrams,
current abstract visual arithmetic can still treat numbers as external glyphs read from fixed visual templates;
AFVNB-MNR exposes this attribute-predicate binding gap with executable fuzzy scene-graph programs and shows that number-only,
coordinate-only, and hard-role baselines fail under controlled counterfactual stress.
```

禁止版本：

```text
AFVNB-MNR proves that current VLMs cannot do human-like fuzzy mathematical reasoning.
```

## 4. Story Spine

| Step | Message | Evidence object |
| --- | --- | --- |
| Progress | Visual math and abstract reasoning benchmarks increasingly test multimodal reasoning. | MathVista, MathVerse, MATH-Vision, RAVEN, CLEVR/GQA references. |
| Promise | MNR-style tasks suggest a clean way to study visual arithmetic. | DARR/MNR framing and 3-context/8-candidate format. |
| Failure | Current abstract visual arithmetic can let models treat numbers as glyphs in fixed templates. | FVNB v0 ontology audit; MNR generator discussion; candidate-level visual counterfactual gap. |
| Proxy split | Reading numbers plus discrete roles is used as if it were visual-number rule understanding. | Number-only, coordinate-only, hard-program baselines. |
| Mechanism | Attribute-Predicate Binding Collapse: numeric attributes are decoupled from fuzzy predicates that define rule variables. | APCC, HG, VDI, SRR. |
| Lens | Fuzzy first-order scene-graph programs make the hidden variable executable. | AFVNB metadata schema and program verifier. |
| Artifact | AFVNB-MNR dataset with object attributes, fuzzy predicates, counterfactual operators, verifier, and baseline audits. | Release artifact. |
| Evidence | Data validity, shortcut failure, strong model analysis, OOD robustness, human sanity. | Experiment protocol tables. |

## 5. Contribution Bullets

当前设计阶段可写：

```text
1. We identify attribute-predicate binding as a missing diagnostic object in abstract visual arithmetic: numeric values should be bound as object attributes to visual predicates before arithmetic rules are evaluated.
2. We formalize the task using executable fuzzy scene-graph programs, where object variables bind to value-token objects, visual roles are fuzzy predicates, and arithmetic atoms read numeric_value attributes.
3. We design AFVNB-MNR, a programmatic benchmark specification with counterfactual candidate operators, metadata contracts, verifier requirements, and shortcut-diagnostic metrics.
```

实验完成后可加：

```text
4. We instantiate the benchmark with controlled families and show through oracle recomputation, candidate-bias audits, shortcut baselines, and model evaluations that AFVNB isolates attribute-predicate binding rather than generic visual difficulty.
```

不应写：

```text
We propose a more difficult dataset.
We outperform existing benchmarks.
We introduce fuzzy logic to visual math for the first time.
```

## 6. Teaser Figure Plan

### 6.1 Teaser message

Teaser 必须在一眼内表达：

```text
Same numeric values.
Different fuzzy visual predicates.
Different executable rule truth.
Hard/discrete parsing misses the difference.
```

### 6.2 Teaser layout

Panel A: Old proxy

```text
Image -> OCR digits -> hard role labels -> crisp expression
```

Message:

```text
This proxy treats numeric values as external symbols.
```

Panel B: AFVNB ontology

```text
value-token objects:
  o_A.numeric_value = 8
  o_B.numeric_value = 3
  o_C.numeric_value = 5

fuzzy predicates:
  Outer(o_A)=0.96
  Inner(o_B)=0.94
  Boundary(o_C)=0.91
```

Message:

```text
Numbers are attributes of objects; visual roles are fuzzy predicates.
```

Panel C: Executable program

```text
exists x,y,z:
  Outer(x) AND Inner(y) AND Boundary(z)
  AND Value(x)-Value(y)=Value(z)
```

Message:

```text
Rule truth is computed by binding predicates and attributes.
```

Panel D: Counterfactual

```text
Same objects and values.
Boundary/region geometry changes.
Hard roles may remain unchanged.
Fuzzy truth drops below threshold.
```

Message:

```text
The arithmetic semantics changes because predicate strength changes.
```

### 6.3 Visual discipline

Do not include:

```text
decorative colors
gradient backgrounds
busy puzzle grids
candidate negative labels
truth scores on evaluation panels
```

Allowed:

```text
minimal scene graph callouts
simple object ids in method figure
predicate bars in debug/instructional figure
clean candidate images in dataset examples
```

## 7. Abstract Skeleton

Do not use as final abstract until evidence exists. Use this as a claim scaffold.

### Current safe abstract draft

```text
Visual mathematics benchmarks increasingly ask whether models use diagrams rather than text or numeric shortcuts, but abstract visual arithmetic can still treat numbers as external glyphs read from fixed visual templates. We study this gap as attribute-predicate binding: the requirement that numeric values be bound as object attributes to visual predicates before arithmetic rules are evaluated. We propose AFVNB-MNR, a benchmark design for Attribute-Fuzzy Visual-Number Binding, where each sample contains value-token objects with numeric_value attributes, fuzzy visual predicates induced by scene geometry, and executable fuzzy scene-graph programs that determine answer labels. AFVNB-MNR specifies counterfactual answer operators, metadata contracts, verifier requirements, shortcut baselines, and evaluation metrics for testing whether models propagate fuzzy visual evidence into rule truth. This design positions AFVNB-MNR as a controlled diagnostic benchmark rather than a broad visual mathematics dataset.
```

Claim status:

| Sentence | Status |
| --- | --- |
| visual math benchmark momentum | literature-supported |
| abstract visual arithmetic can treat numbers as external glyphs | design/ontology-supported; needs empirical audit for stronger wording |
| attribute-predicate binding definition | design-supported |
| AFVNB-MNR benchmark design | supported by docs |
| model testing results | not claimed yet |

### Evidence-ready abstract extension

Add only after experiments:

```text
On generated probe and paper splits, fuzzy oracle recomputation is exact by construction, while number-only, coordinate-only, and hard-program baselines fail on the intended counterfactual stress regimes. Strong vision-language models show errors aligned with shortcut reliance under APCC and SRR diagnostics, indicating that AFVNB exposes a capability not captured by crisp role parsing.
```

Required evidence:

```text
1k+ probe
baseline table
strong model table
APCC/SRR metrics
candidate-only audit
```

## 8. Introduction Reverse Outline

### Paragraph 1: Field momentum

Message:

```text
Visual mathematical and abstract reasoning benchmarks have made diagram use and structured reasoning central evaluation concerns.
```

Evidence:

```text
MathVista, MathVerse, MATH-Vision, RAVEN, CLEVR/GQA.
```

Avoid:

```text
claiming existing benchmarks are flawed in general.
```

### Paragraph 2: Missing diagnostic object

Message:

```text
For abstract visual arithmetic, a model can still solve many tasks by reading digits and applying fixed or hard visual roles, without testing whether visual structure changes arithmetic semantics.
```

Evidence:

```text
DARR/MNR generator analysis;
FVNB v0 anchor-centric audit;
candidate-level counterfactual motivation.
```

Allowed wording:

```text
can still allow
does not explicitly isolate
leaves room for
```

Avoid:

```text
does not use vision
is invalid
fails completely
```

### Paragraph 3: Proxy split

Message:

```text
The proxy is number reading plus hard role assignment; the desired construct is attribute-predicate binding under fuzzy rule truth.
```

Evidence:

```text
formal definition and teaser.
```

### Paragraph 4: AFVNB formalization

Message:

```text
AFVNB represents numbers as object attributes and roles as fuzzy predicates evaluated by executable first-order programs.
```

Evidence:

```text
metadata schema, program schema, verifier.
```

### Paragraph 5: Dataset artifact

Message:

```text
The benchmark is generated through controlled counterfactual operators with metadata sufficient for recomputation and shortcut diagnosis.
```

Evidence:

```text
counterfactual operators, release checklist, experiment protocol.
```

### Paragraph 6: Evidence summary

Current design-stage version:

```text
We provide the benchmark specification, construction protocol, validation plan, and release requirements.
```

Evidence-ready version:

```text
We report oracle validation, shortcut baselines, OOD robustness, model evaluations, and human sanity checks.
```

## 9. Related Work Positioning

| Area | What to say | What not to say |
| --- | --- | --- |
| CLEVR/GQA diagnostic VQA | We inherit executable scene graph/program and semantic audit principles. | We are not building another language VQA dataset. |
| RAVEN/PGM/RAVEN-FAIR | We inherit structured abstract reasoning and candidate-bias lessons. | We do not claim RAVEN lacks structure; our difference is graded predicate strength entering arithmetic truth. |
| Bongard-LOGO/PTR | We inherit program-guided concepts and object/attribute annotations. | We do not expand into general concept learning or part reasoning. |
| MathVista/MATH-Vision | We acknowledge broad visual math evaluation. | We do not compete on coverage or real-world diversity. |
| MathVerse/VC-Bench/VisuLogic | We align with explicit visual dependency and vision-centric reasoning. | We do not use language variants as the primary construction axis. |
| VisioMath/TACIT | We align with image candidates, near-miss distractors, and deterministic verification. | We do not claim image-option evaluation is new. |
| Fuzzy sets/LTN | We use fuzzy membership and many-valued first-order semantics as operational language. | We do not claim to introduce fuzzy logic itself. |

## 10. Method Section Outline

### 10.1 Object-attribute scene graph

Key equation:

```text
o_i = (class_i, numeric_value_i, geometry_i, visual_attributes_i, fuzzy_predicates_i)
```

Message:

```text
`numeric_value` is a first-class object attribute.
```

### 10.2 Fuzzy predicate induction

Key equation:

```text
mu_R(o_i) = normalize(raw_R(geometry_i, regions))
```

Message:

```text
Fuzzy predicates are deterministic functions of visual structure.
```

### 10.3 Executable fuzzy program

Key equation:

```text
T(scene, program) =
max_{x,y,z distinct} Tnorm(
  Outer(x), Inner(y), Boundary(z),
  Eq(Value(x)-Value(y), Value(z))
)
```

Message:

```text
Rule truth is the joint binding of visual predicates and numeric attributes.
```

### 10.4 Counterfactual operators

Message:

```text
Each negative candidate violates one semantic atom or one binding factor.
```

### 10.5 Verification and metadata

Message:

```text
Labels are recomputable from metadata; images are renderings, not label sources.
```

## 11. Experiments Section Outline

| Experiment | Reviewer question | Required table |
| --- | --- | --- |
| Data validity | Are labels clean and candidate sets balanced? | Dataset audit table. |
| Shortcut baselines | Does old proxy fail where expected? | Mechanism table. |
| Counterfactual consistency | Does model follow truth ranking? | APCC/HG table. |
| OOD robustness | Does benchmark overfit to one renderer/family? | split matrix. |
| Human sanity | Is the task legible? | human by margin bin. |
| Strong models | Does the gap matter for current models? | strong model table and SRR. |
| Ablations | Are counterfactual operators necessary? | ablation table. |

## 12. Claim-Evidence Ledger

| Claim | Evidence required | Current status | Allowed wording now |
| --- | --- | --- | --- |
| AFVNB defines attribute-predicate binding | schema and formalization | supported by docs | “we define/design” |
| AFVNB labels are executable | verifier implementation | needs implementation | “is designed to make labels executable” |
| Hard-role parsing is insufficient | hard-program stress failure | needs probe | “we test whether hard-role parsing is sufficient” |
| Candidate bias is controlled | candidate-only audit | needs probe | “we include a candidate-bias audit” |
| Strong VLMs exhibit binding collapse | strong model results | needs experiment | cannot claim yet |
| AFVNB is reusable | release artifact | needs release | “the release is designed to include...” |
| AFVNB is Oral-aspiring | broad evidence stack | needs evidence | internal only |

## 13. Reviewer Risk Checklist

| Risk | Response in paper |
| --- | --- |
| Synthetic toy dataset | Position as diagnostic lineage like CLEVR/RAVEN; show controlled hidden variable and falsifiers. |
| Fuzzy labels arbitrary | Deterministic fuzzy programs, margin filtering, t-norm sensitivity, human sanity. |
| OCR shortcut | Attribute-only/number-only controls and predicate-only perturbations. |
| Coordinate shortcut | Predicate-OOD, Style-OOD, coordinate-order baseline. |
| Candidate leakage | Candidate-only audit, correct-index entropy, visual/statistical balance. |
| RAVEN overlap | Emphasize discrete structure vs graded predicate truth. |
| No model relevance | Strong model suite after construction is validated. |
| Overclaiming novelty | Novelty audit forbids “first fuzzy visual math benchmark.” |

## 14. Reverse Outline Gate

| Surface | One message | Status |
| --- | --- | --- |
| Title | Attribute-predicate binding in visual arithmetic | ready |
| Teaser | Same numbers, changed fuzzy predicates, changed rule truth | ready as design |
| Abstract | Define benchmark object, avoid result claims until evidence | design-ready |
| Introduction | progress -> proxy split -> mechanism -> artifact -> evidence | design-ready |
| Method | object attributes + fuzzy predicates + executable program | ready |
| Experiments | mechanism validation, not leaderboard chasing | protocol-ready |
| Dataset card | controlled diagnostic benchmark with limitations | ready |

If any future paragraph does not point back to Attribute-Predicate Binding Collapse, cut or rewrite it.

## 15. Project Page Plan

Project page sections:

```text
1. Teaser animation/static figure: same values, changed predicates, changed truth.
2. What is attribute-predicate binding?
3. AFVNB sample format: 3 context + 8 candidates.
4. Metadata and executable program.
5. Counterfactual operators.
6. Metrics and audits.
7. Download links: images, metadata, verifier, baselines.
8. Known limitations.
```

Do not make the project page a leaderboard-first page. The first viewport should show the mechanism, not a table of model scores.

## 16. Current Maturity Decision

```text
Maturity level:
  L4 design-complete minimal artifact specification.

One-sentence thesis:
  AFVNB-MNR defines attribute-predicate binding as a controlled diagnostic object for abstract visual arithmetic by representing numeric values as object attributes, visual roles as fuzzy predicates, and labels as executable fuzzy scene-graph programs.

Evidence held:
  literature-aligned benchmark standards;
  AFVNB blueprint;
  implementation plan;
  experiment protocol;
  metadata schema;
  dataset card checklist;
  FVNB v0 proof-of-concept showing fuzzy rule truth can differ from hard-role truth.

Evidence missing:
  AFVNB v1 implementation;
  oracle recomputation report;
  candidate-bias audit;
  baseline results;
  human sanity;
  strong model evaluation;
  family/OOD expansion.

Forbidden claims:
  AFVNB proves VLMs lack human-like reasoning.
  AFVNB is the first fuzzy visual math benchmark.
  AFVNB is harder or better than MathVista/MATH-Vision/DARR.

Next one-week action:
  implement AFVNB scene/program/verifier in an isolated branch or approved current workspace, generate a 20-sample smoke probe, and run schema/oracle/candidate audits.
```
