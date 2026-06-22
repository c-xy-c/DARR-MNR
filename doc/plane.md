# FVNB-MNR 可行计划：把 Fuzzy Logic 放进 Rule Semantics

> 2026-06-18 更新：本文是 FVNB v0 计划，核心贡献是把 fuzzy logic 放进 rule semantics；但它仍以 `digit anchor` 为主要 ontology。根据后续讨论，顶会对齐版本应升级为 **AFVNB-MNR**：数字是 `value_token` 对象的 `numeric_value` attribute，视觉结构提供 fuzzy predicates，rule 是 executable fuzzy first-order program。完整新蓝图见 [`doc/afvnb_oral_benchmark_blueprint.md`](./afvnb_oral_benchmark_blueprint.md)；后续实现和论文包装应以该文档为主。

日期：2026-06-17  
目标：围绕当前 MNR 数据集的 number-vision coupling 不足，提出一个有 Oral 潜力的新数据集版本。本文不是泛泛地说“做更难的数据集”，而是把一个缺失的研究对象定义清楚：**fuzzy visual-number rule binding**。

## 0. 一句话结论

我们建议新数据集叫 **FVNB-MNR: Fuzzy Visual-Number Binding Machine Number Reasoning**。它的核心不是让图像更复杂，而是让视觉结构以连续隶属度的形式进入规则真值：

```text
visual structure G
  -> role membership mu_G(role | digit anchor)
  -> fuzzy rule F over roles and arithmetic predicates
  -> truth score T_F
  -> deterministic label by threshold and margin
```

最关键的一步是第三步：**fuzzy logic 必须放进 rule semantics**。如果 fuzzy 只放在图像渲染里，它只是视觉噪声；如果只放在角色解析里，最后仍然 hard threshold 成离散表达式，论文贡献会退化成“更软的 parser”。我们要测的是：模型是否能把 graded visual evidence 传播到 symbolic/arithmetic rule truth。

## 1. 研究定位

### 1.1 从现有 MNR 看到的缺口

DARR/MNR 已经把 visual arithmetic reasoning 从 missing-number generation 推到 rule matching：每题有 3 个 context panels 和 8 个 candidates，模型要选出与 context 共享同一算术表达式的候选。这是合理且有价值的设计。

但是从仓库生成器看，当前题内视觉布局通常先绘制一次，再复制到 context、correct candidate 和 wrong candidates；候选差异主要来自数字、操作符或括号变异。因此，在同一道多选题里，视觉更像数字读取地址和分组模板，而不是会改变候选语义的变量。

更准确的批评不是“MNR 没有视觉”，而是：

```text
MNR/DARR 是有效的 arithmetic visual reasoning benchmark；
但 released generator 和 saved format 没有把 candidate-level visual role binding
作为独立可诊断变量暴露出来。
```

FVNB-MNR 要补的就是这根轴。

### 1.2 RAVEN 确实有一点这个意思

RAVEN 的重要性在于它不是纯图像分类，而是把视觉结构、对象属性和规则推理联系起来。RAVEN-FAIR 又说明多选候选如果构造不干净，会被 context-blind/candidate-only shortcut 利用。

因此我们的定位必须谦虚：

```text
RAVEN: visual structure -> discrete structure representation -> discrete rule matching
Hard VNB: visual structure -> hard visual roles -> crisp arithmetic expression
FVNB-MNR: visual structure -> role memberships -> fuzzy rule truth -> deterministic answer
```

也就是说，RAVEN 给了“结构进入推理”的方向；FVNB-MNR 的新变量是 **graded structure strength enters rule truth**。这比“再做一个 RAVEN 风格数据集”更有 paper object。

### 1.3 近两年视觉数学 benchmark 给出的约束

近年视觉数学和视觉推理工作共同指向三个趋势：

| 方向 | 代表工作 | 对 FVNB-MNR 的约束 |
| --- | --- | --- |
| 显式视觉依赖 | MathVerse, VC-Bench, VisuLogic | 题目不能只“带图”，图必须改变标签或 rule truth。 |
| 程序化与动态变体 | DynaMath, TACIT | 必须保存 seed、program/oracle、变体关系，支持可复算标签。 |
| 图像候选与 near-miss | VisioMath, TACIT, RAVEN-FAIR | 负例不能随机凑，要有结构相似且只违反一个机制约束的 near-miss。 |
| 结构化 metadata | CLEVR, GQA | 必须保存 scene graph、program/rule、role binding、negative type，才能诊断错误。 |
| 广覆盖视觉数学 | MathVista, MATH-Vision | 它们证明方向重要，但 FVNB-MNR 应定位为小而可控的机制 benchmark，不和真实题库比规模。 |

FVNB-MNR 的独特位置应写成：

```text
不是最大的视觉数学 benchmark；
不是普通 abstract visual reasoning benchmark；
而是一个机制诊断型 abstract visual arithmetic benchmark，
专门测 visual structure 如何把 numbers 绑定到 fuzzy arithmetic rule semantics。
```

## 2. Yangshi 式 Proxy Split

| 元素 | FVNB-MNR 中的定义 |
| --- | --- |
| Field momentum | 视觉数学、抽象视觉推理、多模态推理 benchmark 正在强调 explicit visual dependency、programmatic generation 和 counterfactual evaluation。 |
| Proxy A | 模型能在固定视觉布局中读出数字、解析 hard roles，并匹配一个离散算式。 |
| Construct B | 模型能理解视觉结构以连续强度决定数字角色，并把这些 role memberships 传入 fuzzy rule truth。 |
| Regime R | 数字集合不变，坐标甚至不变，hard role 甚至不变，但视觉边界/距离/包含强度改变，导致 fuzzy truth ranking 改变。 |
| Mechanism H | **Fuzzy rule binding collapse**：模型把 graded visual evidence 压缩成 hard address/hard role，导致不能跟随 fuzzy rule truth。 |
| Artifact O | 带 deterministic fuzzy oracle、role membership metadata、counterfactual negatives 和 mechanism metrics 的新数据集 FVNB-MNR。 |

诊断 thesis：

```text
If fuzzy rule binding collapse is real, then models or hard-role parsers that succeed on
crisp visual arithmetic should fail when the number set, coordinates, or hard roles are
held fixed but fuzzy role memberships change the rule truth ranking.
```

这句话是本文的研究门槛。若不能做出这个 regime，项目就只是“更复杂的 MNR”，达不到 Oral 的研究对象标准。

## 3. Formalization：Fuzzy Logic 放在 Rule 里面

### 3.1 Panel 定义

每个 panel 定义为：

```text
p = (G, A, N, R, mu_G, F)
```

其中：

```text
G: scene graph，包括外框、内框、边界、区域、包含/相交/邻接关系；
A = {a_i}: digit anchors，每个 anchor 有像素坐标；
N = {n_i}: 每个 anchor 上的数字值；
R: visual roles，例如 outer, inner, boundary, outside；
mu_G(r | a_i) in [0, 1]: 由 G 诱导的角色隶属度；
F: fuzzy rule，作用在 role predicates 和 arithmetic predicates 上。
```

### 3.2 Rule truth

对一个 rule schema，例如：

```text
outer - inner = boundary
```

不要先 hard assign `outer=9, inner=4, boundary=5` 再判断真假；而是定义 fuzzy formula：

```text
F_diff(a, b, c) =
  role(a, outer)
  AND role(b, inner)
  AND role(c, boundary)
  AND near_eq(value(a) - value(b), value(c))
```

候选 panel 的 truth score 为：

```text
T_F(p) =
  max over injective role-anchor assignments pi:
    Tnorm(
      {mu_G(r | pi(r)) for r in R_F}
      union
      {tau_arith(E(N, pi))}
    )
```

v0 默认：

```text
Tnorm_product(x_1, ..., x_k) = product_i x_i
tau_eq(x, y) = 1 if x = y else 0
```

也就是说，第一版先让算术部分保持 crisp，fuzzy 性主要来自视觉角色隶属度。这能避免任务难度混入“近似算术”，使核心变量干净地落在 visual-to-rule binding 上。

### 3.3 T-norm variants

为避免 reviewer 认为结论依赖 product t-norm，v0 就应保留三种 rule semantics：

```text
Product:      AND(x, y) = x * y
Godel/min:    AND(x, y) = min(x, y)
Lukasiewicz:  AND(x, y) = max(0, x + y - 1)
```

主集用 product；ablation 用 min 和 Lukasiewicz。论文中不需要声称某个 t-norm 是“人类真实逻辑”，只需要说明：不同 many-valued semantics 下，核心 counterfactual 仍然要求模型传播 role membership，而不是 harden 掉它。

### 3.4 Deterministic label

Fuzzy rule 不等于模糊答案。生成器对 8 个候选计算 truth score，然后用阈值和 margin 产生唯一标签：

```text
score(c) = T_F(c)
correct = argmax_c score(c)

valid sample iff:
  score(correct) >= theta_pos
  score(correct) - max_{c != correct} score(c) >= delta
  max_{c != correct} score(c) <= theta_neg
```

建议 v0 默认：

```text
theta_pos = 0.75
delta = 0.20
theta_neg = 0.45
```

这些不是理论常数，而是 dataset construction hyperparameters。论文应报告 sensitivity sweep。

## 4. Membership 的可行实现

v0 从 containment family 开始，因为它最容易保证 construct validity。每个 panel 画一个 outer rectangle 和 inner rectangle，数字 anchor 分布在 outer-only、inner、inner-boundary 或 outside 区域。

### 4.1 Signed-distance membership

对 anchor `a`，定义：

```text
d_outer(a): a 到 outer 边界的 signed distance，区域内为正；
d_inner(a): a 到 inner 边界的 signed distance，区域内为正；
bd_inner(a): a 到 inner 边界线的距离。
```

raw membership：

```text
mu_inner_raw(a) =
  sigmoid((d_inner(a) - m) / tau)

mu_boundary_raw(a) =
  w_b * exp(-bd_inner(a)^2 / (2 sigma^2))

mu_outer_raw(a) =
  sigmoid((d_outer(a) - m) / tau) * (1 - sigmoid((d_inner(a) - m) / tau))

mu_outside_raw(a) =
  1 - sigmoid((d_outer(a) - m) / tau)
```

再归一化：

```text
mu_G(r | a) = mu_r_raw(a) / sum_{r'} mu_{r'}_raw(a)
```

保存时同时保留 raw 和 normalized：raw 反映视觉证据强弱，normalized 适合作 rule assignment。

### 4.2 为什么 containment 足够作为 v0

containment 看起来简单，但这正是 v0 的优点。第一版不是追求视觉复杂度，而是证明 fuzzy rule binding 这个变量能被干净测到。set relation、partition、topology 都可放到 v1/v2；如果 v0 的简单 containment 都不能打掉 number-only、fixed-coordinate、hard-role shortcut，直接做复杂图形只会让归因变混。

## 5. “同样数字，不同视觉结构 -> 算式语义改变”如何合理成立

最强版本不是简单 role swap，而是固定数字、固定坐标，甚至固定 hard role，只改变 membership 强度。

### 5.1 Hard VNB 基础例子

```text
Rule: outer - inner = boundary

Candidate A:
  values: {9, 4, 5}
  binding: outer=9, inner=4, boundary=5
  truth: 9 - 4 = 5, true

Candidate B:
  same values: {9, 4, 5}
  visual boundary moves
  binding: outer=4, inner=9, boundary=5
  truth: 4 - 9 = 5, false
```

这里语义改变来自 `G -> role binding -> expression input`，不是人为规定。

### 5.2 FVNB 的核心例子

更有新意的例子是 hard role 不变：

```text
Rule:
  role(a, outer) AND role(b, inner) AND role(c, boundary) AND near_eq(a - b, c)

Candidate A:
  values: a=9, b=4, c=5
  hard roles: outer(a), inner(b), boundary(c)
  memberships: 0.95, 0.95, 0.95
  product truth: 0.857
  valid

Candidate B:
  values and coordinates unchanged
  hard roles still: outer(a), inner(b), boundary(c)
  boundary shifts so memberships become: 0.55, 0.55, 0.55
  product truth: 0.166
  invalid
```

这就是 FVNB-MNR 的 Oral-level hook：

```text
discrete structure is unchanged,
but graded visual evidence changes rule truth.
```

这使 FVNB-MNR 与 RAVEN/hard-VNB 拉开差异。RAVEN-style hard structure parser 可以给出相同离散角色，但它不能解释为什么 truth score 从 0.857 掉到 0.166。

## 6. v0 数据集规格

### 6.1 Task format

沿用 MNR/RAVEN 风格：

```text
Input: 3 context panels + 8 candidate panels
Output: choose 1 candidate
```

context panels 共享同一个 fuzzy rule schema，但数字值和几何 instance 可变化。candidate set 中只有一个满足 fuzzy rule truth threshold 和 margin。

### 6.2 Visual family

v0 只做 containment：

```text
outer rectangle
inner rectangle
digit anchors
roles = outer, inner, boundary, outside
```

推荐图像保持低视觉噪声、黑白线框、清晰数字。先不要加自然语言题干。

### 6.3 Rule schemas

v0 只做三类：

| Schema id | Fuzzy rule | 设计原因 | 过滤条件 |
| --- | --- | --- | --- |
| `diff_outer_inner_boundary` | `outer - inner = boundary` | 非交换，role swap 后通常改变真假 | 避免 `outer=inner`、`boundary=0` |
| `sum_outer_inner_boundary` | `outer = inner + boundary` | directed composition，output role 固定 | 避免另一个排列也成立 |
| `ratio_outer_inner_boundary` | `outer / inner = boundary` | 对 role order 最敏感 | `inner != 1`，必须整除 |

第一版不要加入长表达式。算术越长，模型失败越难归因到 binding。

### 6.4 Candidate set

每题 8 个候选固定包含：

| 候选类型 | 作用 |
| --- | --- |
| `correct` | fuzzy rule truth 最高且过阈值 |
| `same_number_fuzzy_role_shift` | 数字集合相同，membership/role truth 变，打击 number-only |
| `same_coordinate_fuzzy_boundary_shift` | 数字坐标相同，边界移动，打击 fixed-coordinate |
| `hard_role_invariant_fuzzy_flip` | hard role 不变，membership 变，打击 crisp parser |
| `fuzzy_rule_false_positive` | hard expression 成立但 membership 低，打击 expression-only |
| `shortcut_consistent_false_positive` | 按坐标/数字 shortcut 看像正确，按 fuzzy rule 错 |
| `arithmetic_only_distractor` | visual binding 对但算术错 |
| `near_miss_distractor` | truth score 接近但未过 margin，用于边界难度 |

这些 negative 不是“凑 7 个错项”，而是机制负例。每个负例都要有 `negative_type` 和 `shortcut_consistency` metadata。

### 6.5 Validity checks

每个 sample 必须通过：

| Check | 条件 | 失败处理 |
| --- | --- | --- |
| Unique answer | 只有一个 candidate 过 `theta_pos` 且满足 margin | 丢弃整题 |
| Membership resolvability | correct 的核心 role membership 足够高，非核心候选不产生多解释 | 重采样 geometry |
| Non-degenerate arithmetic | 避免交换后仍成立、重复数导致等价、除 0、`inner=1` 等 | 重采样数字 |
| Counterfactual isolation | 每个机制负例主要只破坏一个因素 | 重采样该负例 |
| Candidate bias control | correct/negative 的数字范围、位置、线宽、视觉复杂度近似平衡 | 重采样或重排 |
| Oracle reproducibility | 从 metadata 复算 truth score 与保存值一致 | 失败即 bug |

## 7. Metadata Schema

每个 `.npz` 不能只保存 image 和 answer index。至少保存：

```json
{
  "sample_id": "fvnb_000001",
  "family": "containment",
  "rule": {
    "id": "diff_outer_inner_boundary",
    "t_norm": "product",
    "expression": ["=", ["-", "outer", "inner"], "boundary"]
  },
  "thresholds": {
    "theta_pos": 0.75,
    "theta_neg": 0.45,
    "delta": 0.20
  },
  "scene_graph": {
    "outer_rect": {"left": 8, "top": 8, "right": 72, "bottom": 72},
    "inner_rect": {"left": 30, "top": 30, "right": 50, "bottom": 50}
  },
  "candidates": [
    {
      "candidate_id": "c0",
      "is_correct": true,
      "negative_type": null,
      "truth_score": 0.857,
      "hard_truth_score": 1.0,
      "anchors": [
        {
          "id": "a_outer",
          "value": 9,
          "xy": [20, 40],
          "hard_role": "outer",
          "membership": {
            "raw": {"outer": 0.91, "inner": 0.01, "boundary": 0.04, "outside": 0.04},
            "normalized": {"outer": 0.91, "inner": 0.01, "boundary": 0.04, "outside": 0.04}
          }
        }
      ],
      "shortcut_consistency": {
        "number_only": false,
        "fixed_coordinate": false,
        "hard_role": true,
        "fuzzy_rule": true
      }
    }
  ]
}
```

metadata 是这个数据集的科学价值之一。没有 metadata，就无法做 Hardening Gap、Membership Sensitivity、Shortcut Reliance 等机制指标。

## 8. Splits

v0 先做小规模 probe；v1 再做论文主数据集。

### 8.1 v0 probe

```text
family: containment
schemas: 3
samples: 1k-5k
t-norm: product main, min/Lukasiewicz ablation
goal: prove task variable is clean and shortcuts fail where expected
```

### 8.2 v1 paper dataset

| Split | 设计 | 测什么 |
| --- | --- | --- |
| IID | family/schema/fuzzy width 同分布 | 基础可学性 |
| Membership-OOD | 测试未见过的 `tau/sigma/margin` | 是否学 membership-sensitive rule |
| Rule-OOD | 测试新 rule combination | 是否把 role membership 和 arithmetic schema 解耦 |
| T-norm-OOD | train product, test min/Lukasiewicz generated labels | 是否过拟合单个 fuzzy connective |
| Counterfactual stress | same-number、same-coordinate、hard-role invariant fuzzy flip | 是否真正依赖 fuzzy rule binding |
| Candidate-bias audit | candidate-only evaluation | 是否有 RAVEN 式候选偏置 |

## 9. Metrics

不要只报告 8-way accuracy。主指标应该围绕机制：

| Metric | 定义 | 解释 |
| --- | --- | --- |
| FRA | Fuzzy Rule Accuracy，按 fuzzy oracle label 的 8-way accuracy | 总体能力 |
| FCC | Fuzzy Counterfactual Consistency，成对 counterfactual 中预测是否随 truth ranking 改变 | 是否跟随 fuzzy rule |
| HG | Hardening Gap，fuzzy oracle 与 hard-role oracle/模型在 hard-role invariant split 上的差距 | 是否 harden 掉 membership |
| MSA | Membership Sensitivity Accuracy，hard role 不变但 membership 变的子集准确率 | 是否感知角色强度 |
| MBA | Margin-Binned Accuracy，按 top1-top2 truth margin 分桶 | 是否在低 margin 样本退化 |
| FSRR | Fuzzy Shortcut Reliance Rate，错误是否与 number/coordinate/hard-role shortcut 一致 | 错误归因 |
| ARR | Ambiguity Rejection Rate，生成器因 margin 不足拒绝样本比例 | 数据清洁度 |

预期最有说服力的表格：

| Split | Number-only | Fixed-coordinate | Hard-role oracle | Fuzzy oracle | Strong model |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy crisp | low/mid | mid | high | ~100 | high |
| Same-number fuzzy shift | low | mid | high or mid | ~100 | ? |
| Same-coordinate fuzzy shift | low | low | high or mid | ~100 | ? |
| Hard-role invariant fuzzy flip | low | low | low | ~100 | ? |

如果 hard-role oracle 在 hard-role invariant fuzzy flip 上仍然很高，说明反事实没有真正把 fuzzy logic 放进 rule，应回到生成器重做。

## 10. Baselines

### 10.1 必须先跑的低成本 baseline

| Baseline | 输入 | 作用 |
| --- | --- | --- |
| Random | 无 | sanity |
| Candidate-only | 只看 8 candidates | 检查候选偏置 |
| Number-only solver | 数字集合 | 检查 bag-of-numbers shortcut |
| Fixed-coordinate solver | 固定 anchor order | 检查地址模板 shortcut |
| Hard-role oracle | `argmax_r mu(r|a)` 后做 crisp expression | 检查 RAVEN/hard-VNB 是否足够 |
| Fuzzy symbolic oracle | 使用完整 `mu` 和 `F` | 上界和标签验证 |
| Membership-shuffled oracle | 打乱 anchor 和 membership | 负控制 |

### 10.2 深度模型 baseline

v0 可先只做 DARR 和一个简单 CNN/ViT 多选模型。v1 再加入：

```text
DARR, WReN, MRNet/SCAR/PredRNet 类 AVR 模型,
以及若做多模态扩展再加入 GPT-4o/GPT-4.1, Gemini, Claude, Qwen-VL, InternVL 等。
```

第一篇不要让 VLM 榜单抢走主线。主线是 fuzzy rule binding 机制，不是“谁在我们的榜上第一”。

## 11. 一周可执行版本

### Day 1：形式化和 oracle

交付：

```text
Rect/signed distance
containment_membership(point, outer, inner)
FuzzyRule(rule_id, t_norm)
evaluate(panel) -> truth_score, best_assignment
harden_panel_memberships(panel)
```

验收：

```text
同样 hard role 下，高 membership panel score > 0.75；
同样 hard role 下，低 membership panel score < 0.45；
hard-role oracle 对二者都给 true。
```

### Day 2：渲染器和 context/candidate pipeline

交付：

```text
render_panel(scene_graph, anchors)
generate_context(rule_id)
generate_correct_candidate(rule_id)
```

验收：

```text
context_images shape = (3, H, W)
answer_set_images shape = (8, H, W)
metadata 可从 seed 重放
```

### Day 3：机制负例

交付 7 类 negatives：

```text
same_number_fuzzy_role_shift
same_coordinate_fuzzy_boundary_shift
hard_role_invariant_fuzzy_flip
fuzzy_rule_false_positive
shortcut_consistent_false_positive
arithmetic_only_distractor
near_miss_distractor
```

验收：

```text
每类 negative 有 negative_type；
每类 negative 的 shortcut_consistency 与预期一致；
correct truth margin >= delta。
```

### Day 4：生成 1k probe

交付：

```text
1k samples
metadata.jsonl
generation report
ARR / rejected reasons
```

验收：

```text
unique-answer rate 稳定；
oracle reproducibility = 100%;
candidate-only baseline 接近随机。
```

### Day 5：shortcut baselines

交付：

```text
number-only
fixed-coordinate
hard-role oracle
fuzzy symbolic oracle
membership-shuffled oracle
```

验收：

```text
fuzzy oracle 接近 100%;
hard-role oracle 在 hard-role invariant fuzzy flip 上显著低于 fuzzy oracle；
number-only / fixed-coordinate 在对应 split 上下降。
```

### Day 6：人工抽检和 teaser

交付：

```text
50 samples manual audit
2x2 teaser:
  same digits
  same coordinates
  same hard roles
  different memberships
  different fuzzy truth
```

验收：

```text
人工 ambiguity rate <= 5%;
teaser 一眼能看出问题不是 OCR 或普通算术。
```

### Day 7：paper package

交付：

```text
one-sentence thesis
claim-evidence ledger
main table prototype
failure taxonomy
v1 scaling plan
```

验收：

```text
每个 contribution 都能指向一个 evidence object；
没有 unsupported claim 写进 abstract/introduction。
```

## 12. Oral-level 论证

### 12.1 这篇工作不能怎么写

不要写：

```text
We propose a harder visual math dataset.
We add fuzzy logic to images.
We outperform existing benchmarks.
```

这些都会显得像增量数据集。

### 12.2 应该怎么写

更强的写法是：

```text
We identify fuzzy rule binding as a missing evaluation object in visual arithmetic reasoning:
the ability to propagate graded visual evidence into symbolic/arithmetic rule truth.
```

Oral 标准里最重要的是“改变一个研究对象或评价标准”。FVNB-MNR 的标准转移是：

```text
From: can models parse a hard visual structure and match a crisp expression?
To: can models preserve graded visual evidence until the rule-truth level?
```

### 12.3 Contribution bullets

计划阶段可以这样写：

```text
1. We identify fuzzy rule binding collapse, a failure mode where visual arithmetic solvers harden graded visual evidence into addresses or discrete roles.
2. We formalize fuzzy visual-number rule binding using scene graphs, role memberships, t-norm fuzzy formulas, and deterministic margin-based labels.
3. We instantiate the formalization as FVNB-MNR, a programmatic counterfactual benchmark with fuzzy-rule metadata and shortcut-diagnostic candidates.
4. We evaluate the construct with hard-role oracles, fuzzy symbolic oracles, shortcut baselines, t-norm ablations, and membership-sensitive stress splits.
```

真正投稿时，第 4 条必须换成实验事实，而不是计划。

### 12.4 最强 teaser

第一页 teaser 应该展示四列：

```text
same numbers -> same coordinates -> same hard roles -> different fuzzy memberships -> different rule truth
```

这张图如果画不出来，说明机制还不够锋利。

## 13. Claim-Evidence Ledger

| Claim | 当前状态 | 允许写法 | 需要证据 |
| --- | --- | --- | --- |
| 当前 MNR 题内布局复制会削弱候选级视觉绑定诊断 | 代码支持 | “may under-isolate candidate-level binding” | 引用代码和论文描述 |
| RAVEN 已将视觉结构接入离散规则推理 | 文献支持 | “RAVEN-style benchmarks already connect structure and rules” | 引用 RAVEN |
| 现有视觉数学 benchmark 强调显式视觉依赖但未主测 fuzzy rule truth | 文献支持但需谨慎 | “leave room for a controlled fuzzy-rule diagnostic” | related work table |
| fuzzy rule binding collapse 是模型真实失败 | hypothesis | 不能强写 | v0 baselines |
| hard-role invariant fuzzy flip 是核心新 regime | design claim | “designed to isolate crisp-parser shortcuts” | hard-role vs fuzzy oracle gap |
| FVNB-MNR 有 Oral 潜力 | conditional | “Oral-aspiring if v0 evidence holds” | FRA/HG/MSA/FSRR + human audit |
| FVNB-MNR 更接近人类推理 | unsupported | 不写 | human study |
| FVNB-MNR 比 MathVista/MNR 更难 | unsupported | 不写 | matched evaluation |

当前 maturity：

```text
L3 approaching L4:
proxy split and mechanism are clear;
minimal artifact is specified;
the next gate is a v0 diagnostic probe proving hard-role/number/coordinate shortcuts fail
where fuzzy symbolic oracle remains clean.
```

## 14. Reviewer 风险和反驳

| 风险 | 反驳/修复 |
| --- | --- |
| “这只是 synthetic toy dataset” | 强调不是 toy complexity，而是 new evaluation object；用 hard-role invariant fuzzy flip 展示 hard structure 不够。 |
| “RAVEN 已经做结构推理” | 承认 RAVEN 是前身；区别是 discrete structure vs graded rule truth。 |
| “fuzzy label 主观” | label 由 deterministic fuzzy formula、threshold 和 margin 生成；低 margin 样本拒绝。 |
| “containment 太简单” | v0 追求 construct validity；v1 加 set/partition，v2 加 topology/symmetry。 |
| “模型失败可能只是看不清图” | 使用清晰图像、metadata oracle、人类抽检、easy crisp split 控制 OCR/视觉难度。 |
| “candidate bias” | 必跑 candidate-only baseline；负例位置和视觉复杂度平衡。 |
| “product t-norm 任意” | 报 min/Lukasiewicz ablation；主张不是某个 t-norm，而是 membership enters rule semantics。 |
| “没有真实应用” | 定位为 diagnostic benchmark，类似 CLEVR/RAVEN；价值是隔离机制，不是替代真实数学题库。 |

## 15. 最终可行版本

最合理的 v0 就是：

```text
Name:
  FVNB-MNR

Family:
  containment

Roles:
  outer, inner, boundary, outside

Membership:
  signed-distance inner/outer
  Gaussian boundary membership
  raw + normalized saved

Rules:
  diff_outer_inner_boundary
  sum_outer_inner_boundary
  ratio_outer_inner_boundary

Logic:
  fuzzy formulas over role predicates and arithmetic predicates
  product t-norm main
  min/Lukasiewicz ablation

Labels:
  deterministic argmax with theta_pos, theta_neg, delta

Candidates:
  correct
  same_number_fuzzy_role_shift
  same_coordinate_fuzzy_boundary_shift
  hard_role_invariant_fuzzy_flip
  fuzzy_rule_false_positive
  shortcut_consistent_false_positive
  arithmetic_only_distractor
  near_miss_distractor

Metadata:
  scene graph
  anchors
  values
  hard roles
  raw/normalized memberships
  fuzzy rule
  truth score
  hard truth score
  negative type
  shortcut consistency

Metrics:
  FRA, FCC, HG, MSA, MBA, FSRR, ARR

Baselines:
  random
  candidate-only
  number-only
  fixed-coordinate
  hard-role oracle
  fuzzy symbolic oracle
  membership-shuffled oracle
  DARR
```

这版足够小，能一周内完成 probe；也足够锋利，能支撑一个 Oral-aspiring benchmark story。关键验收句是：

```text
same numbers, same coordinates, same hard roles,
but different visual memberships produce different fuzzy rule truth.
```

如果我们能把这句话变成 teaser、metadata oracle、hardening gap 和 baseline failure table，这个数据集就不只是“新题库”，而是一个新的评价对象。

## 16. 参考来源

- [DARR: A Dual-Branch Arithmetic Regression Reasoning Framework for Solving Machine Number Reasoning, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32127)
- [Machine Number Sense: A Dataset of Visual Arithmetic Problems for Abstract and Relational Reasoning, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5489)
- [RAVEN: A Dataset for Relational and Analogical Visual rEasoNing, CVPR 2019](https://arxiv.org/abs/1903.02741)
- [RAVEN-FAIR / Scale-Localized Abstract Reasoning, CVPR 2021](https://arxiv.org/abs/2009.09405)
- [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning, CVPR 2017](https://arxiv.org/abs/1612.06890)
- [GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering, CVPR 2019](https://arxiv.org/abs/1902.09506)
- [MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts, ICLR 2024](https://arxiv.org/abs/2310.02255)
- [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?, ECCV 2024](https://arxiv.org/abs/2403.14624)
- [MATH-Vision: Measuring Multimodal Mathematical Reasoning, NeurIPS 2024 Datasets and Benchmarks](https://arxiv.org/abs/2402.14804)
- [DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of VLMs, 2024](https://arxiv.org/abs/2411.00836)
- [VC-Bench: Benchmarking Multimodal Mathematical Reasoning with Explicit Visual Dependency, 2025](https://arxiv.org/abs/2504.18589)
- [VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models, 2025](https://visulogic-benchmark.github.io/VisuLogic/)
- [VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs, 2025](https://arxiv.org/abs/2506.06727)
- [TACIT Benchmark: A Programmatic Visual Reasoning Benchmark for Generative and Discriminative Models, 2026](https://arxiv.org/abs/2603.00206)
- [Zadeh, Fuzzy Sets, Information and Control, 1965](https://doi.org/10.1016/S0019-9958(65)90241-X)
- [Stanford Encyclopedia of Philosophy: Fuzzy Logic](https://plato.stanford.edu/entries/logic-fuzzy/)
- [Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge, 2016](https://arxiv.org/abs/1606.04422)
