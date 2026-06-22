# VQ-Expr 1x3 RAVEN-like Visual Goal Completion Audit

日期：2026-06-21

目标原文：

```text
有点不像正常的数据集，你还是得仔细调研和仔细比对一下，另外应该是1x3。
```

本文逐项审计当前 worktree 是否已经满足该阶段目标。

## 1. Requirement Ledger

| Requirement | Evidence | Status |
| --- | --- | --- |
| 使用 Yangshi-style problem analysis，而不是只做外观微调。 | `doc/raven_visual_grammar_reset.md` 和 `doc/vqexpr_minimal_visual_gate_audit.md` 均使用 proxy A / construct B / regime R / mechanism H / artifact O 框架。 | satisfied |
| 仔细调研 RAVEN 视觉特点。 | `doc/raven_visual_grammar_reset.md` 总结 RAVEN 的灰度几何对象、7 figure configurations、A-SIG 层级、Layout/Entity attributes、full-panel candidates。 | satisfied |
| 仔细对照 RAVEN 代码。 | `doc/plan.md` 和 `doc/avr_benchmark_visual_grammar_review.md` 记录 `assets/README.md`、`src/dataset/main.py`、`src/dataset/rendering.py` 的 context/choice split：`image = imgs[0:8] + answers`。 | satisfied |
| 修正为 1x3。 | `PRESENTATION_LAYOUT = "1x3_context_row"`；`context_images.shape[0] == 3`；overview 上方只画 3 个 context panels。 | satisfied |
| 8 个候选必须是 full-panel choices。 | `answer_set_images.shape[0] == 8`；`presentation_constraints.full_panel_candidates = True`；tests cover this. | satisfied |
| 视觉不应像彩色机制图/编码表。 | `VISUAL_SURFACE_STYLE = "minimal_grayscale_object_attribute"`；renderer 只画外框和同一种圆形 mark；无 color token、bar、dial、path、internal structure lines、role-specific shape。 | satisfied |
| 不引入奇怪 attribute 以外的装饰。 | `minimal_surface = True`、`single_geometric_primitive = True`、`no_internal_structure_lines = True`、`no_right_side_output_node = True`；tests check flags and grayscale RGB channel equality。 | satisfied |
| 生成可视化样例。 | `outputs/vqexpr_minimal_1x3/vqexpr_000000_overview.png`。 | satisfied |
| 生成 RAVEN 对照审查图。 | `outputs/vqexpr_minimal_1x3_visual_audit/raven_vs_vqexpr_minimal_visual_audit.png`。 | satisfied |
| 生成多 surface 对照图。 | `outputs/vqexpr_minimal_1x3_variants/minimal_contact_sheet.png`。 | satisfied |
| 语义 oracle 仍可复算。 | 200-sample audit: `metadata_oracle_accuracy = 1.0`, `score_argmax_accuracy = 1.0`。 | satisfied |
| 候选偏置没有明显恶化。 | 200-sample learned candidate-only probe: `test_accuracy = 0.13` vs chance `0.125`；视觉统计差异很小。 | preliminary satisfied |
| 测试通过。 | `python -m unittest -q` -> `Ran 25 tests OK`。 | satisfied |

## 2. Current Artifacts

核心代码：

```text
mnr_dataset/vqexpr_generator.py
tests/test_vqexpr.py
```

核心文档：

```text
doc/plan.md
doc/raven_visual_grammar_reset.md
doc/avr_benchmark_visual_grammar_review.md
doc/vqexpr_minimal_visual_gate_audit.md
```

核心输出：

```text
outputs/vqexpr_minimal_1x3/vqexpr_000000_overview.png
outputs/vqexpr_minimal_1x3_variants/minimal_contact_sheet.png
outputs/vqexpr_minimal_1x3_visual_audit/raven_vs_vqexpr_minimal_visual_audit.png
outputs/vqexpr_minimal_1x3_probe200/audit_report.json
```

## 3. Final State

Current schema:

```text
schema_version = vqexpr_1_9_avr_1x3_v4_minimal
presentation_layout = 1x3_context_row
strip_semantics = three_context_panels_plus_eight_full_panel_candidates
visual_surface.style = minimal_grayscale_object_attribute
```

Current visual constraints:

```text
grayscale_only = True
minimal_surface = True
single_geometric_primitive = True
no_internal_structure_lines = True
no_right_side_output_node = True
full_panel_candidates = True
```

## 4. Claim Boundary

Allowed:

```text
The current VQ-Expr prototype now uses a 1x3 context row with 8 full-panel choices
and a minimal grayscale object-attribute visual surface closer to RAVEN-style panels.
```

Forbidden:

```text
The dataset is fully RAVEN-equivalent.
Candidate-only shortcut is impossible.
The visual grammar is publication-ready without human sanity review.
```

## 5. Remaining Work Beyond This Goal

These are future research tasks, not blockers for the current objective:

```text
1. Human visual sanity review.
2. 1k/10k stronger audit.
3. CNN/ViT candidate-only baseline.
4. Role-permutation and Rule-OOD splits.
5. Hard-decoder vs fuzzy oracle gap.
```

## 6. Completion Decision

The current goal is satisfied:

```text
1. RAVEN was researched and compared at paper/code/visual levels.
2. The generated presentation is 1x3.
3. The visual renderer was changed away from colored mechanism diagrams.
4. The current output is a minimal grayscale object-panel prototype.
5. Tests and audit evidence support the implementation.
```
