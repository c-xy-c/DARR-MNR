# AFVNB-MNR Dataset Card And Release Checklist

日期：2026-06-18

本文档规定 AFVNB-MNR 发布时必须随数据一起给出的 dataset card 内容和 release gate。目标是让 benchmark 像顶会数据集/评测论文一样可审计，而不是只发布一批图片和答案。

## 1. Dataset Card Summary

Dataset card 开头应使用保守表述：

```text
AFVNB-MNR is a controlled diagnostic benchmark for attribute-predicate binding in abstract visual arithmetic. Each sample contains object-level scene graphs, numeric object attributes, fuzzy visual predicates, executable fuzzy first-order programs, and counterfactual answer candidates. The benchmark is designed to test whether models bind numeric values to visually induced fuzzy predicates before evaluating arithmetic rules.
```

禁止开头表述：

```text
AFVNB-MNR is a general visual mathematics benchmark.
AFVNB-MNR proves current VLMs cannot reason like humans.
AFVNB-MNR is the first fuzzy visual reasoning dataset.
```

## 2. Required Release Artifacts

Public release must include:

| Artifact | Required | Purpose |
| --- | --- | --- |
| `images/` | yes | Evaluation images without answer leakage. |
| `metadata.jsonl` | yes | Canonical sample metadata following `afvnb_v1`. |
| `programs.json` | yes | Program definitions and aggregation semantics. |
| `splits.json` | yes | IID/OOD/stress split membership. |
| `counterfactual_pairs.json` | yes | Seed-level paired variants for APCC/SWA. |
| `generation_config.json` | yes | Reproducibility configuration. |
| `dataset_card.md` | yes | Dataset scope, limits, audits, and intended use. |
| `verifier.py` | yes | Recompute labels and reject invalid samples. |
| `baseline_audit.py` | yes | Number-only, hard-program, candidate-only audits. |
| `render_debug.py` | recommended | Reconstruct debug views from metadata. |
| `human_sanity_report.json` | paper release yes | Human legibility evidence. |
| `model_results.json` | paper release yes | Baseline and strong model predictions by split. |

Do not release evaluation images with:

```text
truth score overlays
correct-answer highlights
counterfactual operator labels
debug object ids
program text
```

## 3. Dataset Card Sections

### 3.1 Intended use

Required content:

```text
AFVNB-MNR is intended for evaluating controlled visual-number binding, shortcut reliance, and fuzzy predicate-to-rule propagation in abstract visual arithmetic. It is suitable for diagnostic model analysis, controlled training, and benchmark construction studies.
```

Disallowed use claims:

```text
Use AFVNB as a proxy for general mathematical ability.
Use AFVNB as evidence of human-like reasoning.
Use AFVNB as a real-world educational math benchmark.
```

### 3.2 Dataset construction

Must report:

```text
visual families
program schemas
value range
object counts
predicate functions
t-norm variants
thresholds: theta_pos, theta_neg, delta
candidate operators
accepted sample counts
rejected sample counts
rejection reasons
```

### 3.3 Metadata and reproducibility

Must state:

```text
Labels are recomputed from scene_graph + program.
metadata.jsonl is canonical.
Images are renderings of metadata, not the source of labels.
Verifier code is released.
Generation config and seed ids are released.
```

### 3.4 Splits

Must list sample counts for:

```text
IID
Attribute-OOD
Predicate-OOD
Program-OOD
Family-OOD
Style-OOD
Counterfactual-Stress
Candidate-Bias Audit
```

Each split entry must include:

```text
train/test factor held fixed
train/test factor changed
primary metric
baseline expected to fail
```

### 3.5 Data quality audit

Must report:

```text
fuzzy oracle recomputation accuracy
correct-index entropy
candidate-only accuracy
max negative truth distribution
top margin distribution
operator isolation pass rate
appearance/value balance statistics
```

Release gate:

```text
fuzzy oracle recomputation accuracy must equal 100% on accepted samples.
candidate-only accuracy should be near random for 8-way choice.
operator isolation pass rate must equal 100% for released samples.
```

### 3.6 Human sanity

Probe release may include a lab sanity check. Paper release should include a larger human evaluation.

Must report:

```text
number of participants
number of samples
sampling strategy by split and margin bin
human AFRA
human APCC
accuracy by margin bin
inter-annotator agreement
common failure themes
```

Interpretation rule:

```text
Human sanity proves task legibility, not human cognitive equivalence.
```

### 3.7 Baselines

Must report:

```text
fuzzy oracle
hard-program oracle
number-only
coordinate-order
candidate-only
attribute-only
predicate-only
small visual model
open VLM if evaluated
closed VLM if evaluated
```

For each baseline:

```text
input modality
prompt if any
split-level AFRA
APCC where applicable
SRR error attribution where applicable
```

### 3.8 Known limitations

Required limitations:

```text
AFVNB-MNR is synthetic and controlled.
AFVNB-MNR does not cover broad visual mathematics.
AFVNB-MNR does not model human fuzzy logic.
AFVNB-MNR focuses on abstract visual arithmetic and object-attribute binding.
AFVNB-MNR may not reflect natural diagram conventions without further validation.
```

### 3.9 Ethical and misuse notes

Must state:

```text
The dataset contains synthetic images and no personal data.
The benchmark should not be used to rank general intelligence.
The benchmark should not be used as sole evidence for educational readiness.
The benchmark should not be used to claim model interpretability without grounding evidence.
```

### 3.10 License and maintenance

Must include:

```text
license
code release location
dataset version
schema version
generation date
contact or issue tracker
known bugs
planned version changes
```

## 4. Release Gates

### 4.1 Probe release gate

A probe release is allowed when:

```text
20-sample smoke probe passes metadata validation.
1k containment-band probe passes oracle recomputation.
No candidate metadata contains anchors.
All candidates have counterfactual_operator and operator_log.
Candidate-only audit is near random.
Hard-program oracle fails hard-predicate invariant cases.
Presentation and debug views are separated.
Dataset card states probe limitations.
```

Allowed claim:

```text
The probe release validates the AFVNB data contract and tests the initial containment-band mechanism.
```

### 4.2 Paper release gate

A paper release is allowed when:

```text
At least three visual families are included or the paper explicitly claims only a probe benchmark.
All advertised splits have nontrivial sample counts.
All data quality audits pass.
Human sanity is reported.
Strong model results are reported or the paper is framed as a construction-only artifact.
Verifier and baseline audit scripts are released.
Known limitations are explicit.
Claim-evidence ledger has no unsupported abstract/introduction claims.
```

Allowed claim:

```text
The paper release provides a reusable diagnostic benchmark for controlled attribute-predicate binding.
```

## 5. Red-Line Failures

Do not release if:

```text
metadata cannot recompute labels.
evaluation images leak answers.
candidate-only accuracy is high and unexplained.
hard-program oracle solves the claimed fuzzy stress split.
number-only baseline solves predicate-controlled splits.
human high-margin accuracy is near random.
any released sample uses anchor/value as canonical metadata.
```

If any red-line failure occurs, record it as a design failure and revise generator, renderer, or rule family before adding more data.

## 6. Dataset Card Audit Table

Use this table in the dataset card:

| Audit item | Result field | Release gate |
| --- | --- | --- |
| Schema validation | `schema_valid_rate` | 100% |
| Oracle recomputation | `oracle_recompute_accuracy` | 100% |
| Unique answer | `unique_answer_rate` | 100% accepted samples |
| Operator isolation | `operator_isolation_pass_rate` | 100% accepted samples |
| Candidate-only shortcut | `candidate_only_accuracy` | near random for 8-way |
| Hard-program falsifier | `hard_program_stress_accuracy` | substantially below fuzzy oracle |
| Number-only falsifier | `number_only_predicate_stress_accuracy` | substantially below fuzzy oracle |
| Human sanity | `human_high_margin_accuracy` | high enough for legibility |
| Margin safety | `margin_reject_rate` and margin histogram | reported |

## 7. Versioning Policy

Version names:

```text
afvnb_probe_v1
afvnb_paper_v1
afvnb_paper_v1_1
```

Schema version:

```text
afvnb_v1
```

Breaking changes include:

```text
renaming metadata fields
changing program semantics
changing t-norm default
changing candidate operator set
changing split definitions
```

Non-breaking changes include:

```text
adding extra diagnostics fields
adding renderer variants
adding model prediction files
adding human sanity files
```

## 8. Paper Claim Checklist

Before submission, every paper claim must map to a release artifact:

| Claim | Required artifact |
| --- | --- |
| AFVNB labels are deterministic | verifier output and oracle recomputation table |
| AFVNB tests visual dependency | attribute-only, predicate-only, and counterfactual variants |
| AFVNB avoids candidate bias | candidate-only audit and correct-index entropy |
| AFVNB tests fuzzy predicate strength | hard-program falsifier and HG |
| AFVNB is reusable | dataset card, schema, verifier, generation config |
| Models exhibit binding collapse | strong model APCC/SRR results |

Claims not backed by artifacts must be removed or weakened.

## 9. Reviewer-Facing Release Summary

When the artifact is ready, summarize it as:

```text
We release AFVNB-MNR with object-level metadata, executable fuzzy programs, counterfactual candidate operators, verifier scripts, split definitions, and baseline audits. The release is designed so that labels can be recomputed from scene graphs and programs, and so that shortcut baselines can be evaluated without using the images.
```

Do not summarize it as:

```text
We release a harder visual math dataset.
```

The benchmark contribution is not generic difficulty. It is the controlled measurement of attribute-predicate binding.
