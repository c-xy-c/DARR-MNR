# AFVNB-MNR Goal Completion Audit

日期：2026-06-18

本文档对当前 AFVNB-MNR 研究目标做逐项完成度审计。它不是新的方案，而是检查：用户提出的原始目标哪些已经由当前文件证明，哪些还只是设计级完成，哪些仍缺实现或实验。

## 1. 原始目标拆解

用户目标可拆为 8 个可验收要求：

| ID | Requirement | Evidence needed |
| --- | --- | --- |
| R1 | 不要通过奇怪装饰解决问题。 | 视觉设计原则明确排除无关装饰，并把视觉元素绑定到 rule semantics。 |
| R2 | 修正 rule 呈现和 ontology。 | 从 `digit anchor + value` 改为 `value_token.numeric_value + fuzzy_predicates + executable program`。 |
| R3 | 数字自然融入 fuzzy system 作为 attribute value。 | Metadata schema、program schema、implementation plan 都要求 `numeric_value` 是 object attribute。 |
| R4 | 全面调研顶会相关 benchmark。 | 文献覆盖 CLEVR/GQA/RAVEN/PGM/RAVEN-FAIR/Bongard-LOGO/PTR/MathVista/MathVerse/MATH-Vision/DynaMath/MV-MATH/VC-Bench/VisuLogic/VisioMath/TACIT/fuzzy logic/LTN。 |
| R5 | 根据 benchmark 标准构建我们的 benchmark。 | AFVNB blueprint、experiment protocol、metadata schema、release checklist、paper package 形成 benchmark artifact specification。 |
| R6 | 过程推敲仔细，跟顶会对齐。 | Yangshi proxy split、claim-evidence ledger、Oral self-check、novelty audit、reviewer risk gates、falsification policy。 |
| R7 | 在 `doc/` 记录 idea 和调研。 | `doc/` 中完整 AFVNB 文档体系。 |
| R8 | 达到可继续实现和验证的状态。 | TDD implementation plan、metadata contract、release gates、实验 milestones。 |

## 2. 当前证据地图

| Artifact | Role | Status |
| --- | --- | --- |
| `doc/afvnb_oral_benchmark_blueprint.md` | 主蓝图：研究对象、顶会 benchmark 标准、AFVNB ontology、splits、metrics、baselines、claim ledger。 | complete as design |
| `doc/afvnb_implementation_plan.md` | TDD 执行计划：scene/program/generator/verifier/baseline/CLI/visualization。 | complete as plan |
| `doc/afvnb_experiment_protocol.md` | 实验协议：data validity、proxy failure、mechanism metrics、human sanity、falsification policy。 | complete as protocol |
| `doc/afvnb_metadata_schema.md` | 字段级 metadata 合同：`afvnb_v1` schema、validator、serializer rules。 | complete as schema |
| `doc/afvnb_dataset_card_and_release_checklist.md` | 发布与 dataset card 规范：release artifacts、red-line failures、claim checklist。 | complete as release policy |
| `doc/afvnb_paper_package.md` | 投稿包装：title、teaser、story spine、abstract skeleton、related work、claim-evidence。 | complete as paper package |
| `doc/top_benchmark_alignment_attribute_fuzzy_vnb.md` | 从 benchmark 标准到 AFVNB 重构的调研记录。 | complete as supporting research |
| `doc/plane.md` | FVNB v0 计划，顶部已标注由 AFVNB supersede。 | historical/proof-of-concept |
| `doc/plan.md` | 更早的长篇 fuzzy rule benchmark 思考。 | historical/supporting |
| `doc/visual_number_binding_dataset_proposal.md` | VNB-MNR 初始问题分析和 proposal。 | historical/supporting |

## 3. Requirement-by-Requirement Audit

### R1: 不要通过奇怪装饰解决问题

Evidence:

```text
doc/afvnb_oral_benchmark_blueprint.md:
  visual discipline forbids decorative colors, gradients, busy puzzle grids, negative labels, truth scores on evaluation panels.

doc/afvnb_paper_package.md:
  teaser figure plan states that visual elements must show object attributes, fuzzy predicates, executable program, and counterfactual truth change.
```

Status:

```text
Achieved at design/spec level.
Not yet verified by rendered AFVNB v1 images.
```

Remaining evidence:

```text
AFVNB presentation view and debug view implementation.
Human sanity check on presentation view.
```

### R2: 修正 rule 呈现和 ontology

Evidence:

```text
doc/afvnb_oral_benchmark_blueprint.md:
  object = {id, class, numeric_value, visual_attributes, geometry, fuzzy_predicates}
  rule = exists x,y,z: Outer(x) AND Inner(y) AND Boundary(z) AND Equal(Value(x)-Value(y),Value(z))

doc/afvnb_metadata_schema.md:
  canonical metadata forbids value/role/intended_role as replacements for numeric_value/fuzzy_predicates.
```

Status:

```text
Achieved at benchmark definition level.
Not yet implemented in mnr_dataset/afvnb_*.py.
```

Remaining evidence:

```text
AFVNBObject, AFVNBScene, FuzzyProgram implementation and tests.
```

### R3: 数字自然融入 fuzzy system 作为 attribute value

Evidence:

```text
doc/afvnb_metadata_schema.md:
  value_token.numeric_value is required.
  non-value objects must have numeric_value = null.
  arithmetic atoms read numeric_value through Value(object).

doc/afvnb_implementation_plan.md:
  non-negotiable invariant: Program evaluation reads object.numeric_value only.
```

Status:

```text
Achieved at schema/plan level.
Not yet proven by generated metadata.
```

Remaining evidence:

```text
20-sample smoke probe where candidate scene graphs contain no anchors.
Validator output confirming no canonical `anchors` key.
```

### R4: 全面调研顶会相关 benchmark

Evidence:

```text
doc/afvnb_oral_benchmark_blueprint.md:
  structured diagnostic benchmark table:
    CLEVR, GQA, RAVEN, PGM, RAVEN-FAIR, Bongard-LOGO, PTR.
  visual math benchmark table:
    MathVista, MathVerse, MATH-Vision, DynaMath, MV-MATH, VC-Bench, VisuLogic, VisioMath, TACIT.
  fuzzy/neuro-symbolic table:
    Zadeh fuzzy sets, Logic Tensor Networks.

doc/afvnb_metadata_schema.md:
  maps schema features back to CLEVR/GQA, RAVEN-FAIR, MathVerse/VC-Bench, TACIT, fuzzy set/Real Logic.
```

Status:

```text
Achieved for the design phase.
```

Remaining evidence:

```text
Before paper submission, refresh literature search and check whether 2026/2027 visual math benchmark work overlaps AFVNB.
```

### R5: 根据 benchmark 标准构建我们的 benchmark

Evidence:

```text
doc/afvnb_oral_benchmark_blueprint.md:
  defines benchmark object, families, splits, metrics, baselines.

doc/afvnb_experiment_protocol.md:
  defines evidence stack and main experiment tables.

doc/afvnb_metadata_schema.md:
  defines exact sample schema and validator contract.

doc/afvnb_dataset_card_and_release_checklist.md:
  defines release artifacts and release gates.
```

Status:

```text
Achieved as a benchmark specification.
Not achieved as an implemented/generated benchmark artifact.
```

Remaining evidence:

```text
mnr_dataset/afvnb_scene.py
mnr_dataset/afvnb_program.py
mnr_dataset/afvnb_generator.py
mnr_dataset/afvnb_verifier.py
mnr_dataset/afvnb_baselines.py
mnr_dataset/afvnb_visualize.py
mnr_dataset/afvnb_main.py
tests/test_afvnb_mnr.py
```

### R6: 过程推敲仔细，跟顶会对齐

Evidence:

```text
Yangshi objects:
  proxy split
  mechanism H = Attribute-Predicate Binding Collapse
  minimal artifact O = executable object-attribute fuzzy program benchmark
  claim-evidence ledger
  forbidden claims
  novelty overlap audit

Top-conference benchmark standards:
  executable metadata
  candidate-bias audit
  counterfactual variants
  OOD splits
  human sanity
  release checklist
```

Status:

```text
Achieved at research design level.
```

Remaining evidence:

```text
Mechanism-first experiments in the order specified by experiment protocol.
```

### R7: 在 doc/ 记录 idea 和调研

Evidence:

```text
doc/ contains 10 markdown files and 7078 lines before this audit.
The AFVNB-specific core contains blueprint, implementation plan, experiment protocol, metadata schema, dataset card, and paper package.
```

Status:

```text
Achieved.
```

### R8: 达到可继续实现和验证的状态

Evidence:

```text
doc/afvnb_implementation_plan.md:
  TDD steps, test names, target modules, CLI, visualization, full verification gate.

doc/afvnb_experiment_protocol.md:
  M1-M7 evidence milestones.

doc/afvnb_metadata_schema.md:
  validator checklist.
```

Status:

```text
Achieved as implementation-ready plan.
Not yet executed because implementation branch/worktree permission remains unresolved.
```

Remaining evidence:

```text
User approval to create isolated worktree or implement in current main worktree.
```

## 4. Current Maturity Level

```text
Maturity level:
  L4: minimal artifact specification is complete.

Why not L5:
  L5 requires evidence stack: main score, mechanism metric, stress test, ablation, negative control.
  AFVNB currently has a complete specification and FVNB v0 proof-of-concept, but not AFVNB v1 implementation or experiments.
```

## 5. Completion Decision

Design/documentation objective:

```text
Complete enough for research planning and implementation handoff.
```

Full benchmark objective:

```text
Not complete.
```

Reason:

```text
The benchmark is fully specified but not yet implemented, generated, validated, or evaluated.
```

## 6. Next Action Gate

The next meaningful action is one of:

```text
Option A:
  Create an isolated worktree/branch and implement AFVNB v1 using doc/afvnb_implementation_plan.md.

Option B:
  User explicitly permits implementation in current main worktree despite existing uncommitted changes.
```

Recommended:

```text
Option A.
```

Rationale:

```text
Current main worktree contains many untracked and modified files. Implementing AFVNB v1 directly there risks mixing research docs, FVNB v0 proof-of-concept, and AFVNB v1 code in one review surface.
```

## 7. No-Overclaim Reminder

Do not say yet:

```text
AFVNB-MNR is finished.
AFVNB-MNR is paper-ready.
AFVNB-MNR demonstrates strong VLM failures.
AFVNB-MNR is an Oral-level benchmark.
```

Allowed now:

```text
AFVNB-MNR is fully specified at the benchmark-design level.
AFVNB-MNR has an implementation-ready TDD plan, metadata contract, experiment protocol, release checklist, and paper package.
The next step is to implement and validate AFVNB v1 in an isolated worktree or explicitly approved current workspace.
```
