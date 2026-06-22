# AFVNB-MNR 实验协议与评测矩阵

日期：2026-06-18

本文档补齐 `afvnb_oral_benchmark_blueprint.md` 中的证据层：如果 AFVNB-MNR 要作为顶会 benchmark paper，而不是一个生成器 demo，必须明确哪些实验可以证明 benchmark 的构造有效，哪些结果会反过来推翻当前设计。

字段级 metadata 合同见 [`doc/afvnb_metadata_schema.md`](./afvnb_metadata_schema.md)，发布前 dataset card 和 release gate 见 [`doc/afvnb_dataset_card_and_release_checklist.md`](./afvnb_dataset_card_and_release_checklist.md)，投稿叙事与 claim-evidence 边界见 [`doc/afvnb_paper_package.md`](./afvnb_paper_package.md)。

核心原则：

```text
不要只报告模型 accuracy。
要报告：数据是否干净、视觉是否必要、fuzzy predicate 是否进入 rule truth、shortcut 是否失效、强模型错在哪里。
```

## 1. Paper-Level Evidence Stack

AFVNB-MNR 的证据应按 Yangshi 机制链排列：

| 证据层 | 回答的问题 | 必须有的结果 |
| --- | --- | --- |
| Data validity | 生成器是否可靠？ | fuzzy oracle 从 metadata 复算标签为 100%；低 margin 样本被拒绝；candidate 统计平衡。 |
| Proxy failure | 旧 proxy 是否真的不够？ | number-only、coordinate-only、hard-program 在 stress split 上显著低于 fuzzy oracle。 |
| Mechanism metric | Attribute-Predicate Binding Collapse 是否可测？ | APCC/HG/VDI/SRR 能稳定区分 shortcut 与 fuzzy-program following。 |
| Model relevance | 这个 gap 是否影响强模型？ | 至少一组开源 VLM 和闭源 VLM 在 counterfactual stress 上出现可归因错误。 |
| Robustness | 结果是否依赖单一渲染/数值/家庭？ | Attribute-OOD、Predicate-OOD、Style-OOD、Family-OOD 中趋势保持。 |
| Human sanity | 人类是否能理解任务？ | 人类在 clean presentation view 上显著高于随机，错误主要集中低 margin 样本。 |
| Falsifier | 什么结果会推翻论文故事？ | hard-program oracle 若接近 fuzzy oracle，则降级 claim 或重构数据。 |

## 2. Dataset Releases

### 2.1 Probe release

用于一周内验证机制，不追规模。

```text
families: containment-band
programs: diff, sum, ratio
num_samples: 1k-5k
variants per seed: full, attribute-only, predicate-only, counterfactual
purpose: prove object-attribute fuzzy program is executable and shortcut baselines fail.
```

可接受 claim：

```text
AFVNB probe demonstrates a controlled diagnostic regime for attribute-predicate binding.
```

不可接受 claim：

```text
AFVNB is a definitive visual math benchmark.
```

### 2.2 Paper release

用于投稿主实验。

```text
families: containment-band, partition-cell, set-overlap
programs: diff, sum, ratio, two-step composition
num_samples: 50k-200k
splits: IID, Attribute-OOD, Predicate-OOD, Program-OOD, Family-OOD, Style-OOD, Counterfactual-Stress
metadata: full object scene graph, program JSON, operator log, verifier report
```

可接受 claim：

```text
AFVNB provides a reusable benchmark for controlled attribute-predicate binding in abstract visual arithmetic.
```

不可接受 claim：

```text
AFVNB proves general mathematical reasoning ability.
```

## 3. Split Matrix

| Split | Train | Test | Primary question | Minimum evidence |
| --- | --- | --- | --- | --- |
| IID | same family/program/value/predicate distribution | held-out seeds | Can models learn the basic task? | fuzzy oracle 100%; human high; simple learned model above random. |
| Attribute-OOD | values 1-9, 3 value-token objects | unseen value range or 4-5 value-token objects | Does model learn object value function? | number-only does not dominate; AGA remains meaningful. |
| Predicate-OOD | fixed tau/sigma/band width | unseen fuzzy width/region scale | Does model learn predicate membership instead of coordinates? | coordinate-only drops; PGA/APCC diagnose failures. |
| Program-OOD | diff and sum | ratio or two-step composition | Does model decouple predicate binding from arithmetic schema? | hard-coded expression baselines fail. |
| Family-OOD | containment-band | partition-cell or set-overlap | Does object-attribute principle transfer? | performance drop is reported, not hidden; fuzzy oracle remains clean. |
| Style-OOD | one font/line width | different minimal renderer | Does model overfit rendering texture? | human high; style-only changes should not change oracle. |
| Counterfactual-Stress | all ordinary seeds | paired interventions per seed | Does model follow truth ranking under controlled changes? | APCC and HG are headline metrics. |
| Candidate-Bias Audit | no context input | candidates only | Can candidate statistics leak answers? | candidate-only near random; correct-index distribution flat. |

## 4. Metrics With Operational Definitions

### 4.1 AFRA: Attribute-Fuzzy Rule Accuracy

Main 8-way accuracy:

```text
AFRA = mean_i 1[pred_i = argmax_c T(program_i, scene_{i,c})]
```

Use:

```text
main leaderboard
split-level capability summary
```

Do not use alone as proof of mechanism.

### 4.2 APCC: Attribute-Predicate Counterfactual Consistency

For each seed pair `(a, b)` where the same object values are preserved but predicate memberships change:

```text
APCC = mean_pairs 1[
  sign(model_score(a) - model_score(b))
  =
  sign(oracle_truth(a) - oracle_truth(b))
]
```

For black-box VLMs without scores, use answer consistency:

```text
APCC_choice = mean_pairs 1[
  model chooses candidate whose oracle truth increases under the intervention
]
```

Reviewer-facing meaning:

```text
Does the model track the direction of fuzzy rule truth under counterfactual visual changes?
```

### 4.3 HG: Hardening Gap

Let `H(scene)` be the scene where each object keeps only its max predicate among `Outer/Inner/Boundary`.

```text
HG_oracle = AFRA_fuzzy_oracle - AFRA_hard_program
HG_model = AFRA_model_full - AFRA_model_hardened_input
```

For dataset validation, `HG_oracle` should be large on hard-predicate invariant splits. If it is small, AFVNB is not testing fuzzy predicate strength.

### 4.4 VDI: Visual Dependency Index

Use information variants:

```text
Full: image contains value tokens and fuzzy predicate geometry.
Attribute-only: numeric values exposed, predicate geometry removed or neutralized.
Predicate-only: geometry exposed, numeric values masked.
Counterfactual: values held fixed, predicate memberships changed.
```

Define:

```text
VDI_attr = AFRA_full - AFRA_attribute_only
VDI_pred = AFRA_full - AFRA_predicate_only
VDI_cf = AFRA_full - AFRA_counterfactual_blind
```

Interpretation:

```text
High VDI_attr means predicates matter.
High VDI_pred means numeric attributes matter.
Both must matter for AFVNB to be valid.
```

### 4.5 SRR: Shortcut Reliance Rate

For each model error, tag which shortcut would have selected the same wrong answer:

```text
SRR_number = wrong predictions aligned with number-only / all wrong predictions
SRR_coordinate = wrong predictions aligned with coordinate-order / all wrong predictions
SRR_hard = wrong predictions aligned with hard-program / all wrong predictions
```

Use:

```text
error attribution table
model comparison
ablation diagnosis
```

### 4.6 AGA and PGA

AGA requires either model rationales, object-level probes, or a structured baseline output.

```text
AGA = fraction of selected program variables whose object ids carry the oracle numeric_value roles
PGA = fraction of selected program variables whose predicate evidence matches oracle assignment
```

If evaluating black-box VLMs without object rationales, report AGA/PGA only for structured baselines and open models with probing; do not fabricate grounding metrics from final answers.

### 4.7 SWA: Seed-Worst Accuracy

For each seed with variants:

```text
seed_solved = min_{variant in seed} 1[pred_variant = label_variant]
SWA = mean_seed seed_solved
```

This prevents models from getting credit for solving only easy variants of a seed.

### 4.8 ARR: Ambiguity Rejection Rate

During generation:

```text
ARR = rejected_candidate_or_seed_count / attempted_candidate_or_seed_count
```

Report ARR by reason:

```text
low_top_margin
non_unique_answer
non_isolated_counterfactual
degenerate_arithmetic
candidate_bias_violation
oracle_recompute_failure
```

High ARR is not automatically bad; it is evidence that the generator is filtering ambiguous cases. But extremely high ARR may indicate the family is poorly parameterized.

## 5. Baseline Suite

### 5.1 Symbolic and shortcut baselines

| Baseline | Input | Expected role | Pass/fail criterion |
| --- | --- | --- | --- |
| Fuzzy oracle | full metadata | verifier upper bound | 100% on all accepted samples |
| Hard-program oracle | metadata with hardened predicates | crisp parser falsifier | low on hard-predicate invariant split |
| Number-only | numeric multiset only | tests value shortcut | near random on predicate-only perturbations |
| Coordinate-order | fixed object order / center order | tests layout shortcut | drops on Predicate-OOD and Style-OOD |
| Candidate-only | candidates without context | candidate bias audit | near random; no stable correct slot |
| Attribute-only | values without predicate geometry | visual dependency control | low when predicates decide label |
| Predicate-only | geometry without values | arithmetic dependency control | low when values decide label |

### 5.2 Learning baselines

First paper should include at least:

```text
small CNN over image grid
ResNet-style image encoder
ViT-style image encoder
structured scene-graph oracle
hard-program oracle
```

If resources allow:

```text
open-source VLMs with image-option prompts
closed-source VLMs with identical image-only or minimal-language prompt
```

Do not let VLM leaderboard dominate the paper. The main result is mechanism diagnosis, not model ranking.

### 5.3 Prompt protocol for VLMs

Use a minimal prompt:

```text
You are given three context panels and eight answer panels.
Choose the answer panel that follows the same hidden visual-number rule as the context panels.
Return only the answer index from 0 to 7.
```

Do not include:

```text
outer
inner
boundary
fuzzy
rule
negative type
numeric list
```

Reason: the benchmark tests whether the image conveys attribute-predicate binding. Language should not reveal the ontology.

## 6. Data Quality Audits

### 6.1 Oracle recomputation

For every sample:

```text
load metadata
reconstruct AFVNBScene objects
reconstruct FuzzyProgram
recompute candidate truth scores
assert saved score difference <= 1e-9
assert correct_index equals argmax score
```

Failure means generator or serializer bug. Reject dataset release.

### 6.2 Candidate balance

Report per split:

```text
correct index histogram
operator slot histogram
numeric value histogram by candidate type
object count histogram
center coordinate histogram by candidate type
average foreground pixel count by candidate type
average line length / boundary area by candidate type
```

RAVEN-FAIR lesson:

```text
If candidate-only accuracy is high, the dataset is not ready.
```

### 6.3 Counterfactual isolation

Each counterfactual operator must declare:

```text
kept_constant
changed
targets_shortcut
single_primary_intervention
```

The verifier must reject samples where a supposed predicate-only perturbation also changes numeric value distribution, object count, or candidate appearance outside tolerance.

### 6.4 Margin distribution

Report:

```text
top1 truth
top2 truth
top1-top2 margin
max negative truth
hard-program truth of correct and stress negatives
```

Use margin bins:

```text
easy: margin >= 0.50
medium: 0.30 <= margin < 0.50
hard: 0.20 <= margin < 0.30
reject: margin < 0.20
```

Do not train/test on rejected bins unless the paper explicitly studies ambiguity.

## 7. Human Sanity Protocol

Human evaluation is not to prove human-like reasoning. It is to prove task legibility.

### 7.1 Participants

Minimum probe:

```text
3-5 lab members
50 samples
presentation view only
no debug overlays
```

Paper version:

```text
20+ participants or a controlled annotation platform
200-500 samples stratified by split and margin bin
```

### 7.2 Instructions

Use:

```text
You will see three examples and eight answer choices.
Choose the answer that follows the same hidden visual-number rule.
Some objects are closer to regions or boundaries than others; use the visual structure.
```

Do not use:

```text
The rule is outer - inner = boundary.
Membership is fuzzy.
Boundary score is multiplied.
```

### 7.3 Report

Report:

```text
human AFRA
human APCC on paired counterfactuals
accuracy by margin bin
inter-annotator agreement
free-text failure themes
```

If humans fail clean high-margin samples, simplify renderer or ontology before running model experiments.

## 8. Main Tables for a Paper

### 8.1 Dataset audit table

| Split | Samples | Rejection rate | Fuzzy oracle | Candidate-only | Correct-index entropy | Max negative truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IID | report | report | 100 | near random | high | <= theta_neg |
| Counterfactual-Stress | report | report | 100 | near random | high | <= theta_neg |
| Predicate-OOD | report | report | 100 | near random | high | <= theta_neg |

### 8.2 Mechanism table

| Split | Number-only | Coordinate-only | Hard-program | Full model | APCC | HG | SRR_hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Predicate-only perturbation | report | report | report | report | report | report | report |
| Attribute-predicate swap | report | report | report | report | report | report | report |
| Hard-predicate invariant flip | low | low | low | report | high desired | high desired | report |

### 8.3 Strong model table

| Model | IID AFRA | Counterfactual AFRA | APCC | VDI_attr | VDI_pred | SRR top failure | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Small CNN | report | report | report | report | report | report | image-only baseline |
| ViT | report | report | report | report | report | report | image-only baseline |
| Open VLM | report | report | report | report | report | report | minimal prompt |
| Closed VLM | report | report | report | report | report | report | minimal prompt |

### 8.4 Ablation table

| Ablation | Expected effect | Interpretation |
| --- | --- | --- |
| Remove predicate-only negatives | number-only and hard-program scores rise | visual dependency weakened |
| Remove hard-invariant flips | hard-program oracle improves | fuzzy predicate strength no longer tested |
| Use random negatives | candidate-only audit may improve artificially | near-miss structure is necessary |
| Use text prompt with role names | VLM score rises without visual grounding | language leakage |
| Use single renderer only | Style-OOD unknown | rendering shortcut risk |

## 9. Reviewer Risk Gates

| Reviewer concern | Evidence that answers it |
| --- | --- |
| “This is just synthetic toy data.” | Show proxy split, controlled counterfactuals, and failure of hard-program oracle; compare to CLEVR/RAVEN as diagnostic benchmark lineage. |
| “Fuzzy labels are arbitrary.” | Deterministic fuzzy program, margin filtering, t-norm sensitivity, human sanity by margin bin. |
| “Models can solve by OCR and arithmetic.” | Attribute-only and number-only baselines fail on predicate-controlled splits. |
| “Models can solve by location template.” | Predicate-OOD and Style-OOD plus coordinate-order baseline fail. |
| “Candidate construction leaks answer.” | Candidate-only near random, correct-index entropy high, appearance/value histograms balanced. |
| “RAVEN already does visual rules.” | RAVEN is discrete structure; AFVNB isolates graded predicate strength entering arithmetic truth. |
| “No real-world relevance.” | Position as controlled diagnostic benchmark, not replacement for MathVista/MATH-Vision; connect to explicit visual dependency. |
| “No evidence of model failure.” | Strong VLM suite and SRR/APCC error analysis. |

## 10. Falsification Policy

AFVNB is a scientific benchmark only if it can fail its own tests.

Downgrade or redesign if:

```text
fuzzy oracle < 100% on accepted metadata
candidate-only > 25% on 8-way task across stress splits
hard-program oracle > 70% on hard-predicate invariant flip
number-only > 40% on predicate-only perturbation
human high-margin accuracy < 80%
APCC does not separate hard-program from fuzzy-program behavior
```

These thresholds are starting gates, not final claims. They should be reported as construction criteria in the dataset card.

## 11. Dataset Card Requirements

The released dataset card should include:

```text
generation date and code commit
families and program schemas
accepted sample counts per split
rejection rates and reasons
thresholds: theta_pos, theta_neg, delta
t-norm variants
renderer variants
known limitations
forbidden uses
license and release format
baseline audit results
human sanity summary
```

Known limitations should explicitly say:

```text
AFVNB is not a broad visual mathematics benchmark.
AFVNB is not a human cognition model.
AFVNB is a controlled diagnostic benchmark for attribute-predicate binding.
```

## 12. Claim-Evidence Lock

Before writing Abstract or Introduction:

| Claim tier | Allowed only if |
| --- | --- |
| Dataset validity | oracle recomputation, candidate balance, and human sanity pass. |
| Mechanism claim | APCC/HG/SRR show shortcut separation. |
| Model failure claim | strong model suite shows errors under stress splits. |
| Benchmark usefulness | release artifact, verifier, baseline code, and dataset card exist. |
| Oral-level claim | evidence holds across families/splits and reviewer risk gates are answered. |

Safe current wording:

```text
AFVNB-MNR is designed to test whether models bind numeric object attributes to fuzzy visual predicates before evaluating arithmetic rules.
```

Unsafe current wording:

```text
AFVNB-MNR proves current VLMs cannot perform fuzzy visual mathematical reasoning.
```

## 13. Next Evidence Milestones

| Milestone | Artifact | Success criterion |
| --- | --- | --- |
| M1 | 20-sample AFVNB smoke probe | metadata recomputation passes; no `anchors` key in candidate scene graphs. |
| M2 | 1k containment-band probe | fuzzy oracle 100%; candidate-only near random; hard-program fails stress. |
| M3 | human sanity mini-study | humans solve high-margin samples; failure themes are renderer/ambiguity, not ontology confusion. |
| M4 | open-model baseline panel | number-only, coordinate-only, hard-program, CNN/ViT results reported by split. |
| M5 | family expansion | partition-cell and set-overlap reproduce mechanism trends. |
| M6 | strong VLM evaluation | minimal-prompt VLM errors analyzed with APCC/SRR. |
| M7 | paper package | teaser, dataset card, experiment tables, claim-evidence ledger complete. |

Only after M2 should we call AFVNB a working benchmark probe. Only after M5-M7 should we call it paper-ready.
