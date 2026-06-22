# VQ-Expr Minimal 1x3 Visual Gate Audit

日期：2026-06-21

本文是当前极简视觉版本的 gate audit。目标不是证明数据集已经 paper-ready，而是回答一个具体问题：

```text
当前 VQ-Expr 是否已经从“彩色机制图/编码表”移动到
“1x3 RAVEN-like minimal object panel”的方向？
```

## 1. Yangshi Proxy Split

| 项 | 定义 |
| --- | --- |
| Proxy A | `1x3 + 8 candidates` 的外部格式。 |
| Construct B | 正常 AVR/RAVEN-like 数据集视觉语法。 |
| Regime R | 不读 metadata，只看 rendered panels。 |
| Mechanism H | visual primitive 是否引入额外解释性属性。 |
| Minimal artifact O | `minimal_grayscale_object_attribute` renderer。 |

诊断 thesis：

```text
如果 H 是问题根源，那么只改 layout 不足以让图像像 RAVEN；
必须删除彩色 token、内部结构线、role-specific shape、右侧 output node，
让数值只通过少量 object attributes 变化。
```

## 2. 当前视觉样例

核心样例：

```text
outputs/vqexpr_minimal_1x3/vqexpr_000000_overview.png
```

五种 surface 对照：

```text
outputs/vqexpr_minimal_1x3_variants/minimal_contact_sheet.png
```

RAVEN 对照审查图：

```text
outputs/vqexpr_minimal_1x3_visual_audit/raven_vs_vqexpr_minimal_visual_audit.png
```

该对照图上方是 RAVEN paper Figure 4 的视觉配置 crop；中间是当前 VQ-Expr 1x3 样例；下方是当前四类极简 attribute surface。

## 3. Visual Gate Checklist

| Gate | 当前状态 | Evidence |
| --- | --- | --- |
| `1x3` context row | pass | `context_images.shape[0] == 3`; overview 上方三格均为 context。 |
| 8 个 full-panel candidates | pass | `answer_set_images.shape[0] == 8`; candidate 是完整 panel。 |
| 无 query/missing 第四格 | pass | `presentation_constraints.no_query_panel_in_context_row = True`。 |
| 无显式数字/运算符 | pass | tests over quantity primitives；metadata constraint。 |
| 灰度图像 | pass | RGB channel delta = 0；`grayscale_only = True`。 |
| 单一几何 primitive | pass | renderer 只调用 `_draw_mark` 画圆形 mark；`single_geometric_primitive = True`。 |
| 无内部结构线 | pass | `_draw_expression_panel` 只画外框和 marks；`no_internal_structure_lines = True`。 |
| 无 role-specific shape | pass | 所有 role 使用同一 circle primitive。 |
| 无右侧 output node | pass | target 是固定 role slot，不是右侧单独答案节点；`no_right_side_output_node = True`。 |
| 无彩色 token/bar/dial/path 装饰 | pass | surface 只保留 `gray_level / size_level / count / position_set`。 |
| RAVEN-like visual normality | partial | 对照图显示已接近 minimal object panels；仍需人工审查是否过稀疏。 |
| Candidate-only shortcut | preliminary pass | 200-sample learned candidate-only probe `0.13` vs chance `0.125`。 |

## 4. 当前 Metadata Contract

```text
schema_version = vqexpr_1_9_avr_1x3_v4_minimal
presentation_layout = 1x3_context_row
strip_semantics = three_context_panels_plus_eight_full_panel_candidates
visual_surface.style = minimal_grayscale_object_attribute
visual_surface.attribute_family in {gray_level, size_level, count, position_set}
```

Presentation constraints:

```text
grayscale_only = True
minimal_surface = True
single_geometric_primitive = True
no_internal_structure_lines = True
no_right_side_output_node = True
answer_set_separate_from_context = True
full_panel_candidates = True
```

## 5. Verification Commands

```bash
python -m unittest -q
```

Result:

```text
Ran 25 tests
OK
```

Post-hoc 200-sample audit:

```bash
python -m mnr_dataset.vqexpr_main \
  --num_prob 200 \
  --output_dir outputs/vqexpr_minimal_1x3_probe200 \
  --seed 99 \
  --rule_schema mixed

python -m mnr_dataset.vqexpr_audit outputs/vqexpr_minimal_1x3_probe200 \
  --output outputs/vqexpr_minimal_1x3_probe200/audit_report.json
```

Result:

```text
metadata_oracle_accuracy = 1.0
score_argmax_accuracy = 1.0
learned_candidate_only_probe.test_accuracy = 0.13
chance_accuracy = 0.125
correct_index_counts = 25 per slot
visual_family_counts = 40 per family
```

## 6. Claim-Evidence Ledger

| Claim | Status | Allowed wording |
| --- | --- | --- |
| VQ-Expr uses a 1x3 context/choice split. | supported | 可以写。 |
| Current visual surface is minimal grayscale. | supported | 可以写。 |
| Current visual surface is fully RAVEN-equivalent. | forbidden | 只能说 closer to RAVEN-style minimal panels。 |
| Candidate-only shortcut is absent. | needs stronger probe | 只能说 linear visual-stat probe is near chance。 |
| VQ-Expr is not merely RAVEN. | needs final dataset evidence | 可以说 intended distinction is fuzzy quantity + arithmetic AoT。 |

## 7. Remaining Risks

| Risk | Next action |
| --- | --- |
| 图像过稀疏，难以承载足够视觉结构。 | 只调 spacing/size/density，不新增装饰属性。 |
| role slot 太固定，模型可能学 slot-template shortcut。 | 做 Rule-OOD / role-permutation split。 |
| 1-9 太简单。 | 先跑 hard-decoder oracle 和 fuzzy-hardening gap，再决定是否扩展。 |
| candidate-only probe 太弱。 | 跑 CNN/ViT candidate-only baseline。 |

## 8. Maturity Decision

```text
Maturity level: L4 minimal artifact
Research object: visual primitive normality under RAVEN-like grammar
Evidence object: rendered audit sheet + metadata constraints + oracle/candidate-only probes
Writing object: "same 1x3 shell is not enough; visual primitive must be minimal object-attribute grammar"
Next one-week action: human visual review + 1k/10k audit + role-permutation stress split
```
