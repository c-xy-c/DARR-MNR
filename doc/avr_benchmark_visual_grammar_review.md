# AVR Benchmark Visual Grammar Review for VQ-Expr

日期：2026-06-21

本文只回答一个问题：VQ-Expr 的展示为什么要从“混合符号图”大修成 **1x3 context row + 8 full-panel candidates**。

更新说明：本文最初只解决 presentation organization。进一步对照 RAVEN 论文 Figure 4 和代码后，当前实现仍不满足 RAVEN-like visual grammar；完整视觉重置见 [`doc/raven_visual_grammar_reset.md`](./raven_visual_grammar_reset.md)。

## 1. 结论

AVR/RPM benchmark 的可读性来自三件事：

```text
1. one panel grammar:
   同一题内对象共享同一套 visual alphabet。

2. attribute-level variation:
   推理发生在 object attributes 上，而不是显式文字、数字、算符或流程图上。

3. full-panel candidates:
   候选项必须和缺失 panel 同构，不能是孤立答案 token。
```

之前的 VQ-Expr 图像失败点正好相反：一个 panel 里同时出现 color token、bar、grid、path、dial，看起来像解释性符号表，不像 AVR puzzle。修复原则是：

```text
one sample -> one visual_quantity_family
context/query/candidates share the same family
role layout encodes operand binding
object attribute encodes quantity
metadata/verifier encodes AoT; presentation never draws AoT
```

## 2. Benchmark Lessons

| Benchmark | 主要对象 | 对 VQ-Expr 的约束 |
| --- | --- | --- |
| RAVEN | structured RPM with scene grammar and object attributes | 图像必须像 puzzle，不像程序图；结构只通过对象布局和属性呈现。 |
| I-RAVEN / RAVEN-style fixes | debiased structure generation and rule annotations | 需要把 structure metadata 保存下来，防止图像和 label 不一致。 |
| RAVEN-FAIR | answer candidate bias audit | 8 个候选必须 full-panel、single-mutation；必须保留 candidate-only audit。 |
| PGM | synthetic matrices with explicit generalization regimes | 不能只做 IID；必须设计 Rule-OOD、Attribute-OOD、Calibration-OOD。 |
| A-I-RAVEN / I-RAVEN-Mesh | held-out attribute generalization and transfer-oriented RPM variants | 单题单 visual alphabet 之后，还要报告 visual family 分布，支持 Attribute-OOD。 |
| RuleMatch / rule-aware AVR work | context images should be required; answer-only shortcuts violate AVR spirit | generation report 必须暴露 correct-index distribution 和 candidate audit hooks。 |
| RAISE / generative AVR | arbitrary-position answer generation reduces dependence on answer candidate bias | VQ-Expr 保留 full-panel candidates，但必须做 post-hoc candidate-only audit。 |
| rule-omission studies | high RPM scores may reflect shortcuts rather than using all rules | VQ-Expr 要记录 rule family、AoT、mutation type，并报告 oracle/audit。 |
| Bongard-LOGO | program-guided visual concept generation | label 必须由 generator/verifier 产生，而不是人工解释图像。 |

## 2.1 Literature-to-Implementation Matrix

| AVR lesson | Source evidence | VQ-Expr design decision | Current evidence |
| --- | --- | --- | --- |
| A readable RPM item uses one consistent object grammar, not a legend of mixed symbols. | RAVEN uses attributed stochastic image grammar and multiple figure configurations. | `one sample -> one visual_quantity_family` | `metadata.visual_quantity_family`; `test_each_sample_uses_one_visual_quantity_family` |
| Candidate sets can leak the answer even when context is ignored. | RAVEN-FAIR reports the original RAVEN can be solved in a context-blind setting because of biased negatives. | full-panel candidates, balanced correct slots, candidate-only audits | `correct_index_counts`; `candidate_only_heuristics`; `learned_candidate_only_probe` |
| Generalization regimes matter more than IID accuracy alone. | PGM formalizes held-out factor/attribute/style generalization; A-I-RAVEN/I-RAVEN-Mesh continue this direction. | Rule-OOD, Attribute-OOD, Calibration-OOD are planned from v1. | `rule_counts`; `visual_family_counts`; sample-local calibration metadata |
| Labels need executable generation, not visual intuition. | Bongard-LOGO and program-guided AVR lines emphasize generated concepts/rules. | Answer AoT + verifier + mutation logs | `metadata.query_panel.answer_aot`; `metadata.candidates[].mutation_log`; `score_argmax_accuracy` |
| Multiple-choice AVR is vulnerable; generative/open-answer variants reduce candidate bias. | RAISE argues for generative AVR; RAVEN-FAIR fixes candidate construction. | We keep MC for compatibility, but audit it aggressively. | post-hoc `vqexpr_audit.py`; learned candidate-only linear probe |
| High scores may reflect partial or omitted-rule shortcuts. | Rule omission studies warn that RPM success need not imply all rules are used. | Store rule family, visual family, AoT, negative mutation type and require counterfactual isolation. | `metadata.rule_family`; `metadata.visual_quantity_family`; `validity.counterfactual_isolation` |

Current 200-sample sanity result:

```text
metadata_oracle_accuracy = 1.0
score_argmax_accuracy = 1.0
learned_candidate_only_probe.train_accuracy = 0.15
learned_candidate_only_probe.test_accuracy = 0.13
chance_accuracy = 0.125
```

## 3. What VQ-Expr Borrows

VQ-Expr 借用 AVR benchmark 的 presentation grammar：

```text
object set
  -> visual attributes
  -> layout / grouping
  -> missing panel candidate choice
```

但 VQ-Expr 的 hidden semantics 不同：

```text
AVR/RAVEN:
  object attributes -> discrete relation rule

VQ-Expr:
  fuzzy quantity attributes -> role binding -> arithmetic AoT
```

因此，VQ-Expr 不应该画：

```text
operator gate
wire / circuit
calibration strip
visible digits
visible + - * / =
debug labels
```

这些东西会把任务变成“读机制图”，而不是 abstract visual reasoning。

## 4. RAVEN Code-Level Constraint

RAVEN 的关键不是“必须 3x3”，而是它把 problem context 和 answer choices 清楚分离。代码里有三个很直接的事实：

```text
assets/README.md:
  image shape = (16, 160, 160)
  first 8 figures compose the problem matrix
  last 8 figures are choices

src/dataset/main.py:
  imgs = [row_1_1, ..., row_3_2, blank]
  context = [row_1_1, ..., row_3_2]
  answers = render_panel(candidate) for 8 candidates
  image = imgs[0:8] + answers

src/dataset/rendering.py:
  generate_matrix(context panels)
  generate_answers(8 choices)
  merge_matrix_answer(matrix, answer)
```

因此 VQ-Expr 不应该把 query missing panel 混进 context row 里。我们采用的是 RAVEN 的展示原则，而不是 RAVEN 的 3x3 尺寸：

```text
RAVEN:
  8 context matrix panels + 8 full-panel choices

VQ-Expr:
  3 complete context panels + 8 full-panel choices
```

上方 1x3 只放 context；候选区的每个 candidate 才是 query inputs + candidate output 的完整 panel。

## 5. Concrete Renderer Changes

当前实现已经按这份 review 改成了 presentation shell：

```text
schema_version: vqexpr_1_9_avr_1x3_v4_minimal
presentation_layout: 1x3_context_row
strip_semantics: three_context_panels_plus_eight_full_panel_candidates
visual_quantity_family: one of five families, fixed per sample
visual_surface: minimal_grayscale_object_attribute
generation_report:
  presentation_layout
  rule_counts
  visual_family_counts
  correct_index_counts
  candidate_visual_stats
  candidate_only_heuristics
  audit_notes

post-hoc audit:
  python -m mnr_dataset.vqexpr_audit <dataset_dir>
  metadata_oracle_accuracy
  score_argmax_accuracy
  candidate_visual_stats
  candidate_only_heuristics
  learned_candidate_only_probe
```

其中 `rule_counts`、`visual_family_counts`、`correct_index_counts` 在 dataset writer 层面轮转均衡；不是只在文档里承诺。

当前极简视觉层已经删除彩色 token / bar / path / dial / role-specific shapes / internal structure lines。主视觉只保留单对象几何 primitive，数值只通过必要连续 attribute 变化表达：

```text
size_level
color_lightness
stroke_width
aspect_ratio
```

也就是说：

```text
已完成: context/choice split
已完成: minimal grayscale object-attribute prototype
仍需: human visual review and stronger 1k+ audit
```

1x3 context row 语义：

```text
context panel 1: complete input-output example
context panel 2: complete input-output example
context panel 3: complete input-output example
candidate: query inputs + one candidate output, full panel
```

这比旧版本更接近 AVR：

```text
旧版本:
  left side = mixed visual symbols + implicit mechanism

新版本:
  left side = same object grammar repeated across examples
```

## 6. Remaining Risks

| Risk | How to test |
| --- | --- |
| 1-9 过于简单 | increase AoT depth and role-binding OOD before expanding numeric range |
| 候选位置泄漏 | 已轮转均衡 `correct_index_counts` |
| visual family 分布偏置 | 已轮转均衡 `visual_family_counts` |
| 候选视觉统计泄漏 | 已支持 generation-time 与 post-hoc `candidate_visual_stats` / `candidate_only_heuristics`，并加入 learned linear candidate-only probe；publication 前仍建议跑更强 CNN/ViT candidate-only classifier |
| 单题单 family 导致模型学 family-specific shortcut | Attribute-OOD split |
| fuzzy 没有真正进入 label | hard-decoder oracle vs fuzzy oracle gap |
| 1x3 被看成普通 function induction | emphasize no visible symbols and sample-local decoder; compare to MNR visible-number baseline |

## 7. References

- RAVEN: A Dataset for Relational and Analogical Visual rEasoNing. https://arxiv.org/abs/1903.02741
- Measuring Abstract Reasoning in Neural Networks / PGM. https://arxiv.org/abs/1807.04225
- RAVEN-FAIR / Scale-Localized Abstract Reasoning. https://arxiv.org/abs/2009.09405
- A-I-RAVEN and I-RAVEN-Mesh / Generalization and Knowledge Transfer in Abstract Visual Reasoning Models. https://arxiv.org/abs/2406.11061
- RuleMatch: Matching Abstract Rules for Semi-supervised Learning in Abstract Visual Reasoning. https://www.ijcai.org/proceedings/2023/0179.pdf
- RAISE / Towards Generative Abstract Reasoning. https://iclr.cc/virtual/2024/poster/18960
- Rule omission in Raven's Progressive Matrices. https://arxiv.org/html/2510.03127v1
- Bongard-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning. https://arxiv.org/abs/2010.00763
- I-RAVEN and related RAVEN debiasing work should be treated as candidate-bias and structure-consistency warnings, not as a template to copy directly.
