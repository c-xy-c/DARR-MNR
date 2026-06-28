# VQ-Expr v1 计划：1x3 Context-Choice AVR + Fuzzy Quantity AoT

日期：2026-06-21

AVR 可视化语法调研详见 [`doc/avr_benchmark_visual_grammar_review.md`](./avr_benchmark_visual_grammar_review.md)。RAVEN 视觉语法重置稿详见 [`doc/raven_visual_grammar_reset.md`](./raven_visual_grammar_reset.md)。当前极简视觉 gate audit 详见 [`doc/vqexpr_minimal_visual_gate_audit.md`](./vqexpr_minimal_visual_gate_audit.md)。本文只保留和 VQ-Expr 设计直接相关的结论。

本文是当前主计划。目标不是继续修补 MNR 的 `holistic / analytical / analytical_part`，也不是做一个 RAVEN 换皮，而是提出一个更窄、更可诊断的新数据集：

```text
VQ-Expr: Visual Quantity Expression Reasoning

模型必须从无数字、无算符、无文字说明的视觉属性中恢复 fuzzy quantity，
再把这些 quantity 按视觉结构绑定到可执行算术表达式树中，
最后从 8 个完整图像候选中选择唯一正确 panel。
```

当前 v1 明确收敛到 **1-9**，不是 1-99。理由是：我们现在最关键的研究对象不是“大数读数”，而是 **attribute -> fuzzy number -> role binding -> arithmetic AoT** 这条语义链。1-99 在无显式数字条件下很容易变成编码表或密集计数，反而让视觉展示不可读。1-9 足够承载 `+ - * /`、scope、port binding、fuzzy ambiguity 和 counterfactual negatives。

## 1. 核心判断

### 1.1 当前 MNR 的问题

DARR-MNR 的价值在于多 panel、多候选、程序化算术规则。但是它的 number 与 vision 结合方式主要仍是：

```text
visible number / visible operator
  -> layout or grouping
  -> arithmetic rule matching
```

这导致视觉更像“读数地址”或“分组模板”，而不是会真正改变表达式语义的变量。VQ-Expr 要补的不是更多装饰，而是：

```text
同一个视觉 quantity，在不同 visual role structure 中绑定到不同 operand port，
从而改变 hidden expression 的语义和答案。
```

### 1.2 为什么不是 RAVEN

RAVEN 的贡献是把视觉结构、对象属性和抽象规则联系起来；PGM 强调用可控生成器测试 systematic generalization；RAVEN-FAIR/I-RAVEN 一类工作提醒我们，多选候选若有偏置，会被 candidate-only shortcut 利用。

VQ-Expr 吸收这些约束，但研究对象不同：

```text
RAVEN / PGM:
  object attributes -> discrete relational rule

VQ-Expr:
  fuzzy visual quantity attributes -> executable arithmetic AoT
```

所以 VQ-Expr 不能画显式 gate、wire、operator symbol。规则必须藏在 visual role layout 和 metadata verifier 里，图像只呈现对象属性。

### 1.2.1 RAVEN 代码给我们的具体约束

RAVEN 不是只给了一个“3x3 外观”，更重要的是它在数据结构上把 context 和 choices 分开：

```text
RAVEN assets/README.md:
  image: (16, 160, 160)
  first 8 figures = problem matrix
  last 8 figures = choices

RAVEN src/dataset/main.py:
  context = [row_1_1, row_1_2, row_1_3,
             row_2_1, row_2_2, row_2_3,
             row_3_1, row_3_2]
  answers = 8 rendered candidate panels
  image = imgs[0:8] + answers

RAVEN src/dataset/rendering.py:
  generate_matrix(context)
  generate_answers(choices)
  merge_matrix_answer(matrix, answer)
```

因此我们现在不应该在上方画“两个 context + 一个 missing”或“三个 context + 一个 query missing”。VQ-Expr 采用 RAVEN 的组织原则，但不是照搬 3x3：

```text
RAVEN:
  8 context panels + 8 choices

VQ-Expr:
  3 context panels + 8 choices
```

上方 1x3 都是 context。query inputs 不单独作为第四格出现，而是出现在 8 个 full-panel candidate 中；候选之间只在 controlled semantic mutation 上不同。

### 1.2.2 AVR benchmark 调研转成设计约束

| Benchmark line | 关键经验 | VQ-Expr 采用什么 | VQ-Expr 不采用什么 |
| --- | --- | --- | --- |
| RAVEN | RPM 图像需要 structure representation，把视觉对象、属性和推理关系连起来。 | 保存 visual rule/binding metadata；presentation 里只画结构和 attribute。 | 不把结构画成 gate/wire 流程图。 |
| PGM | 评价重点是 controlled generalization regimes，而不是单一 IID accuracy。 | 预留 Rule-OOD、Attribute-OOD、Calibration-OOD。 | 不把“生成一张好看的图”当作完整 benchmark。 |
| RAVEN-FAIR / bias audit | 多选候选很容易泄漏 shortcut。 | 8 个候选必须 full-panel、single-mutation、candidate-only audit。 | 不做随机负例，也不让正确候选在视觉统计上特殊。 |
| Bongard-LOGO | program-guided generation 能控制概念与组合结构。 | AoT/verifier 决定 label，metadata 可复算。 | 不靠人工解释或图片直觉判对错。 |
| 当前 multimodal AVR 趋势 | 模型越来越容易利用表层 pattern，因此 benchmark 要做机制诊断。 | 把核心机制定为 fuzzy quantity decoding + role binding + arithmetic AoT。 | 不宣称“更难”，只证明具体 proxy failure。 |

### 1.3 本轮大修原则：RAVEN-like object-attribute scene

RAVEN/PGM 的可读性不是来自更多符号，而是来自同一 panel 内的对象共享一套 visual alphabet：形状、大小、颜色、位置、数量这些 attribute 在同一语法中变化。之前版本把 color token、bar、grid、path、dial 同时放进一个 panel，导致左侧像混合符号说明图，而不是 AVR puzzle。

进一步检查 RAVEN 论文和代码后，结论要更强：即使每题只用一种 quantity family，`Color Token / Metric Bar / Path / Dial` 这种表面仍然像独立读数器，不像 RAVEN 的 object-attribute scene。新版 renderer 的硬规则应改为：

```text
one sample -> one RAVEN-like scene grammar
query/context/candidate all share this structure template
role binding comes from Structure/Component/Layout slots
quantity value is encoded by object attributes: count, position, size, gray, nested ratio
```

旧五类 attribute 可以保留在 metadata/audit 历史中，但不应作为最终视觉表面。

### 1.4 当前 v7：In-Out + Typed Boundary 结构化构图

当前实现已经从 `v4_minimal`、`v5_structured` 继续升级到：

```text
schema_version = vqexpr_1_9_avr_1x3_v7_typed_boundary
visual_surface.style = typed_boundary_scope_grayscale_a_sig_lite
visual_surface.scene_graph_schema = a_sig_lite_v3
```

这版的变化不是加装饰，而是把 panel 的视觉生成源头改成一个可检查的 A-SIG-lite scene graph：

```text
Scene
  Structure: 3x3Grid / 2x2Grid / Out-InGrid / Out-InCenter / Left-Right
    ComponentGroup: typed semantic boundary, branch/scope/in-out grouping
    Component: q1, q2, q3, q4, target
      Layout: bbox, scope, slot_id, number, position, uniformity
        Entity: circle, gray/size/count/position attribute
```

每个 context panel 和每个 candidate panel 都保存 `visual_scene_graph`。renderer 只读取这个 graph 画图；因此“同一个数字在不同视觉结构中绑定到不同角色”不再只是论文叙述，而是 metadata 与图像共同遵守的约束。

`v5_structured` 的主要教训是：只有 scene graph 和 role slot 还不够，`serial / parallel / calibration` 仍然容易看成简单点阵。`v7_typed_boundary` 因此加入两个结构层：

```text
1. in_out_regions:
   nested / inverse 明确记录 out_roles = [q1]，in_roles = [q2, q3, target]。

2. structure_groups:
   用 circle / square / diamond / hexagon 的灰度边界表示 scope、branch、local calibration frame、out-to-in containment。
```

边界形状只服务于结构，不编码数字值；数字值仍由 circle quantity entity 的 gray/size/count/position attribute 表达。这样可以避免回到“花哨装饰”，同时让结构差异在图像层可见。

当前 rule 到 structure 的映射是：

| Rule family | Structure family | 视觉直觉 |
| --- | --- | --- |
| `serial` | `3x3Grid` | hexagon prefix scope 绑定 q1/q2/q3，diamond tail scope 绑定 q4/target |
| `parallel` | `2x2Grid` + center target | top/bottom square branch scope，center circle merge scope |
| `nested` | `Out-InGrid` | hexagon out-to-in boundary + square inner scope；`out_roles=[q1]` |
| `inverse` | `Out-InCenter` | hexagon out-to-in boundary + diamond inner stack；`out_roles=[q1]` |
| `calibration` | `Left-Right` | diamond local calibration frame 同时包住 q1/q2/target |

样例输出：

```text
outputs/vqexpr_typed_boundary_1x3_v7_refined/vqexpr_000000_overview.png
outputs/vqexpr_typed_boundary_1x3_v7_refined/typed_boundary_five_family_contact_sheet.png
outputs/vqexpr_typed_boundary_1x3_v7_probe200/audit_report.json
```

这版已经解决的差距：

```text
不再只有一个 structure_template 字符串；
每个可见 entity 都能回到 metadata 中的 Component/Layout/Entity；
candidates 是 full-panel scene graph，不是孤立 answer token；
Out-In 类结构有可见外层 component；
serial / parallel / calibration 不再只是裸点阵，而有可见 scope / branch / local-frame；
boundary shape 进入 metadata 与 renderer，不再只用圆形大圈；
仍保持无数字、无算符、灰度、无 role label。
```

仍未解决的差距：

```text
quantity entity 仍只有 circle；Type 目前只给 boundary，不给数值实体；
boundary shape 是结构通道，未来需要验证它不会成为 rule-family shortcut；
parallel 的 center target 是折中设计，不是严格 RAVEN configuration；
candidate negatives 语义上是 single mutation，但视觉扰动还需要更强的 learned candidate-only audit；
当前只是小样本视觉 probe，还没做人类 sanity check 或大规模 split。
```

## 2. Presentation Contract

当前视觉格式固定为 **1x3 context row + 8 full-panel candidates**：

```text
[context 1] [context 2] [context 3]

candidate_i = query inputs + candidate output attribute
```

更精确地说，presentation semantics 是：

```text
three complete context panels + eight full-panel answer candidates
```

这不是 3x3 RPM 的压缩版，也不是把 query missing panel 混在 context 里。前三格必须是同一 hidden expression family 下的完整 visual function examples；candidate 区域中的每个候选才是完整 query panel，即同一 query inputs 加一个候选 output attribute。

画面硬约束：

```text
无阿拉伯数字
无 + - * / =
无文字题干、无标签、无候选编号
无 calibration strip
无 operator gate
无 wire / flowchart / debug graph
无装饰背景
```

允许出现的内容只有：

```text
panel 边框
quantity object 的视觉属性，同一题内必须同 family
role position / 必要 grouping 造成的结构绑定
candidate output 的同 family 视觉属性
```

候选项必须是完整 panel，而不是孤立答案 token。这样才能测试模型是否把 query 输入、context rule 和候选输出放在同一个视觉语义空间里比较，同时与 RAVEN 代码中的 context/choices 分离保持一致。

## 3. Quantity Attribute Families

v1 保留 1-9，但数值必须通过 RAVEN-like object attributes 表达。每一类都必须满足：

```text
render(obs | n, sample_seed)
decode(obs, calibration_s) -> mu(n), n in 1..9
no visible digit / operator leakage
single-sample consistency: all operands and candidate target roles use the same scene grammar
```

推荐新版 attribute families：

| Family | 图像中看到什么 | 数值语义 | Fuzzy 点 |
| --- | --- | --- | --- |
| Count | 1-9 个同类灰度几何对象 | Layout.Number | 遮挡/小尺寸/边界 entity 造成相邻数 membership |
| Position set | 3x3 slots 中占据若干位置 | Layout.Position pattern rank | slot 偏移、边界位置、近邻 slot membership |
| Size level | 一个或多个对象的离散大小 | Entity.Size level 1-9 | 连续尺寸落在相邻 bin 边界 |
| Gray level | 几何对象灰度深浅 | Entity.Color/gray level 1-9 | 灰度接近相邻 level |

当前主 renderer 只保留这四类极简属性。连续比例类属性可以作为后续 fuzzy 分支，但不进入主视觉，避免引入额外装饰。

注意：fuzzy 不是把图像画模糊，而是 decoder 输出分布：

```text
mu_q(n) in [0, 1]
support(mu_q) includes true n and near-neighbor alternatives
```

## 4. Visual Role Binding

每个 panel 是一个完整 scene，而不是一个公式卡片。视觉结构只通过对象位置/包含关系/相对排布表达 role，不出现显式算符，也不出现固定右侧 output 节点。

当前五类 AoT family：

| Family | Hidden expression | 视觉绑定直觉 |
| --- | --- | --- |
| serial | `Sub(Mul(Add(q1,q2),q3),q4)` | 角色来自 slot order 或 component order，不画箭头 |
| parallel | `Add(Add(q1,q2),Sub(q3,q4))` | 左右/上下 component 表示并行 role |
| nested | `Sub(q1,Add(q2,q3))` | outer/inner 或 grid center/corner 表示 scope |
| inverse | `Div(Mul(q1,q2),q3)` | inside/outside 或 paired slots 表示可抵消 role |
| calibration | `Add(q1,q2)` under local decoder | 同一灰度/大小/数量 attribute 经 sample-local decoder 影响答案 |

关键不是让人从单张图读出公式，而是通过前三个 context panels 看到同一 hidden relation，再在 8 个 candidate scenes 中选择满足 target role 的完整 panel。

## 5. Candidate / Negative Contract

每题 8 个候选，唯一正确。每个错误候选只违反一个 primary mechanism，避免随机负例。

```text
C0 correct
C1 output perturbation
C2 leaf decode perturbation
C3 calibration swap
C4 operator swap
C5 port binding swap
C6 scope tree rotation
C7 fuzzy ambiguity trap
```

验收条件：

```text
correct score >= theta_pos
max negative score <= theta_neg
correct score - max_negative >= delta
each negative has exactly one primary mutation
candidate-only baseline should not exploit visual artifact bias
```

这里的“怎么区分正确和错误”必须由 metadata verifier 完成，而不是靠人工解释：

```text
rendered image
  -> stored quantity observations
  -> decode to fuzzy values
  -> bind to Answer AoT
  -> execute arithmetic / fuzzy propagation
  -> compare each candidate score
```

## 6. Schema

每个 `.npz` 至少包含：

```text
context_images: 3 complete context panels
answer_set_images: 8 full candidate panels
correct_answer_image_index: int in [0, 7]
metadata_json:
  sample_id
  presentation_layout = 1x3_context_row
  value_range = 1..9
  visual_quantity_family
  calibration_context
  query_panel
  context_panels
  candidates
  answer_aot
  visual_rule_graph / binding metadata
  validity checks

generation_report:
  rule_counts
  visual_family_counts
  correct_index_counts
  candidate_visual_stats
  candidate_only_heuristics
  audit_notes
```

`overview.png` 只用于人工审查：上方 1x3 context row，下方 8 个候选 panel。

## 7. Evaluation Plan

最小实验不是直接跑大模型，而是先证明数据集本身没有明显捷径：

| Test | 目的 |
| --- | --- |
| metadata oracle | 从 metadata 复算 label，必须 100% |
| candidate-only baseline | 检查 RAVEN-FAIR 式候选偏置 |
| candidate-only heuristic audit | 只看候选图像统计量选答案，report 暴露 `candidate_only_heuristics` |
| learned candidate-only probe | 只看候选图像统计量和 slot one-hot 训练线性 ranker，audit 暴露 `learned_candidate_only_probe` |
| candidate visual-stat audit | correct vs negative 的 ink/mass/bbox 差异，report 暴露 `candidate_visual_stats` |
| correct-index audit | generator 按样本序号轮转正确位置，report 暴露 `correct_index_counts` |
| visual-family audit | generator 按样本序号轮转五类 visual alphabet，report 暴露 `visual_family_counts` |
| context-only ablation | 无候选输出时不可解 |
| no-vision / metadata-pruned baseline | 检查是否只靠 rule family shortcut |
| hard-decoder oracle | 把 `mu(n)` harden 后执行 AoT，测 fuzzy hardening gap |
| calibration-shuffled split | 同一 token 跨 sample 变义，检查全局 lookup |
| rule-family OOD | held-out AoT family |
| attribute-family OOD | held-out visual attribute family |

核心指标：

```text
OracleAcc: metadata verifier accuracy
CandidateBias: candidate-only accuracy
HardeningGap: fuzzy oracle - hard decoder oracle
BindingSensitivity: port/scope mutation caused score drop
```

## 8. 当前实现状态

已实现：

```text
mnr_dataset/vqexpr_quantity.py
  1-9 fuzzy quantity attribute families

mnr_dataset/vqexpr_program.py
  five Answer AoT families with valid 1-9 reverse sampling

mnr_dataset/vqexpr_generator.py
  AVR-style 1x3 context-row renderer
  minimal grayscale object-attribute surface
  single visual alphabet per sample
  full-panel candidates
  C0-C7 candidate generation

mnr_dataset/vqexpr_audit.py
  post-hoc metadata oracle audit
  candidate visual-stat audit
  candidate-only heuristic sanity check
  learned candidate-only linear probe over visual stats and slot one-hot

tests/test_vqexpr.py
  quantity coverage
  local token remapping
  AoT recomputation
  candidate uniqueness
  dataset artifact writing
  post-hoc audit recomputation
```

当前样例：

```text
outputs/vqexpr_minimal_1x3/vqexpr_000000_overview.png
outputs/vqexpr_minimal_1x3_variants/minimal_contact_sheet.png
outputs/vqexpr_minimal_1x3_visual_audit/raven_vs_vqexpr_minimal_visual_audit.png
```

生成命令：

```bash
python -m mnr_dataset.vqexpr_main \
  --num_prob 1 \
  --output_dir outputs/vqexpr_minimal_1x3 \
  --seed 321 \
  --rule_schema nested
```

测试：

```bash
python -m unittest -q
```

Post-hoc audit：

```bash
python -m mnr_dataset.vqexpr_audit outputs/vqexpr_avr_probe40 \
  --output outputs/vqexpr_avr_probe40/audit_report.json
```

上一版 200-sample sanity audit：

```text
metadata_oracle_accuracy = 1.0
score_argmax_accuracy = 1.0
learned_candidate_only_probe.test_accuracy = 0.10
chance_accuracy = 0.125
```

v7 typed-boundary 200-sample sanity audit：

```text
metadata_oracle_accuracy = 1.0
score_argmax_accuracy = 1.0
learned_candidate_only_probe.test_accuracy = 0.11
chance_accuracy = 0.125
hand_written_visual_stat_best = 0.15
```

极简 renderer 后的 200-sample sanity audit：

```text
metadata_oracle_accuracy = 1.0
score_argmax_accuracy = 1.0
learned_candidate_only_probe.test_accuracy = 0.13
chance_accuracy = 0.125
```

当前视觉表面约束：

```text
layout = 1x3 context + 8 full candidates
visual surface = typed boundary + grayscale object attribute
quantity primitive = circle only
boundary shapes = circle / square / diamond / hexagon
no color / no visible role label / no right-side output node
semantic boundary lines only; no decorative line art
attribute surfaces = gray_level, size_level, count, position_set
```

## 9. Paper Claim Boundary

可以写：

```text
VQ-Expr studies whether models can bind fuzzy visual quantity attributes
to executable arithmetic expression trees under controlled counterfactual choices.
```

## 9.1 Release Gate Checklist

| Gate | Status | Evidence |
| --- | --- | --- |
| Visual grammar readable as AVR-style puzzle, not mechanism diagram | typed-boundary prototype, pending human review | v7 sample at `outputs/vqexpr_typed_boundary_1x3_v7_refined/vqexpr_000000_overview.png`; five-family sheet at `outputs/vqexpr_typed_boundary_1x3_v7_refined/typed_boundary_five_family_contact_sheet.png` |
| No visible digits/operators | implemented | presentation constraints + tests over quantity primitives |
| Full-panel answer candidates | implemented | `presentation_constraints.full_panel_candidates` |
| A-SIG-lite scene graph exists for panels/candidates | implemented | `tests.test_vqexpr.TestVQExprAoTAndCandidates.test_visual_scene_graph_is_structured_and_self_contained` |
| Typed boundary and in/out structure are explicit | implemented | `structure_groups`, `boundary_shape`, and `structure.in_out_regions` checked in tests |
| Correct answer position balanced | implemented | `correct_index_counts` rotation; 40-sample probe gives 5 per slot |
| Visual family balanced | implemented | `visual_family_counts` rotation; 40-sample probe gives 8 per family |
| Metadata oracle reproducibility | implemented | post-hoc audit gives `metadata_oracle_accuracy = 1.0` |
| Learned candidate-only sanity probe | implemented | v7 200-sample probe: test accuracy `0.11` vs chance `0.125`; hand-written visual-stat best heuristic `0.15` |
| Strong candidate-only CNN/ViT baseline | pending | needed before publication claims |
| Rule-OOD / Attribute-OOD / Calibration-OOD splits | designed, not fully generated | split definitions in evaluation plan; generation hooks exist |
| Human readability sanity check | pending | needed because visual clarity is partially subjective |

不要写：

```text
VQ-Expr proves VLMs cannot do fuzzy symbolic reasoning.
VQ-Expr is the first visual arithmetic benchmark.
VQ-Expr is harder than RAVEN/MNR/MathVista in general.
VQ-Expr matches human numerical cognition.
```

Reviewer 风险与修复：

| 风险 | 修复 |
| --- | --- |
| “这只是把数字藏起来。” | sample-local calibration + token remapping + calibration-shuffled split |
| “这不就是 RAVEN？” | RAVEN 是离散 abstract rule；VQ-Expr 是 fuzzy quantity + executable arithmetic AoT |
| “这不就是 MNR 换皮？” | 无 visible digits/operators；候选是 full-panel visual-semantic counterfactual |
| “1-9 太简单。” | 复杂性来自 fuzzy decoding、binding、scope、AoT depth 和 counterfactual，而不是大数 |
| “候选偏置。” | candidate-only audit + mutation balancing |
| “fuzzy 主观。” | deterministic decoder、threshold、margin、oracle 复算 |

## 10. 下一步

短期：

```text
1. 人工审查极简 renderer 是否过稀疏、是否还保留足够 role binding。
2. 人工审查 v7 typed-boundary renderer 是否已经像 AVR/RAVEN-style composition，而不是只是固定点阵。
3. 把 `visual_rule_graph` 从旧 gate graph 改名或重构成 `visual_role_binding_graph`。
4. 生成 1k+ probe，跑 oracle/candidate-only/hard-decoder baseline。
5. 对每类 mutation 平衡视觉统计，避免正确候选在面积/密度上泄漏。
```

中期：

```text
1. Rule-OOD / Attribute-OOD / Calibration-OOD split。
2. 人类 sanity check：1-9 版本是否可读、是否过于简单。
3. 若 1-9 被证明太弱，再扩到 compositional multi-digit，而不是直接回到 1-99 dense encoding。
```

一句话收束：

```text
VQ-Expr v1 的正确方向是少画东西、画必要属性：
1x3 context row teaches hidden arithmetic relation;
full-panel candidates bind the same query roles;
candidates differ by one controlled semantic mutation.
```

## 参考线索

- RAVEN: A Dataset for Relational and Analogical Visual rEasoNing, CVPR 2019. https://arxiv.org/abs/1903.02741
- PGM / Measuring abstract reasoning in neural networks, ICML 2018. https://arxiv.org/abs/1807.04225
- RAVEN-FAIR / Scale-Localized Abstract Reasoning, CVPR 2021. https://arxiv.org/abs/2009.09405
- MathVerse, ECCV 2024. https://arxiv.org/abs/2403.14624
- DynaMath, 2024. https://arxiv.org/abs/2411.00836
- VisioMath, 2025. https://arxiv.org/html/2506.06727
- Zadeh fuzzy sets; Logic Tensor Networks / Real Logic for fuzzy first-order semantics.
