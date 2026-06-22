# RAVEN 视觉语法对照与 VQ-Expr 大改方向

日期：2026-06-21

本文只解决一个问题：为什么旧版 VQ-Expr 虽然已经是 `1x3 context row + 8 choices`，视觉上仍然不像正常 AVR/RAVEN 数据集，以及当前极简 renderer 应该怎样约束。

## 1. Yangshi 式问题拆解

当前错误的 proxy split 是：

| 项 | 内容 |
| --- | --- |
| Proxy A | 图像满足 `1x3 + 8 candidates`，没有数字和运算符。 |
| Construct B | 图像具有 RAVEN/AVR 的 object-attribute scene grammar。 |
| Regime R | 人眼只看单个 panel 的视觉形态，不读 metadata，也不看代码。 |
| Divergence | A 成立，但 B 不成立：旧版 panel 像彩色节点/编码图，不像 RPM scene。 |
| Hidden mechanism H | visual primitive 没有被 RAVEN 化；数值被画成 decoder token，而不是 object attribute。 |
| Minimal artifact O | 极简灰度 renderer grammar，而不是继续调排版。 |

一句话诊断：

```text
虽然旧版 VQ-Expr 已经采用 1x3 context 与 8 个 full-panel candidates，
但它把“布局格式”误当成“AVR 视觉语法”；
当只看 panel 表面时，彩色 token、右侧输出点和 role cluster 暴露出机制图感，
因此必须把数量语义压入 RAVEN-like object attributes。
```

## 2. RAVEN 的视觉特点

RAVEN 的正常感来自一套很稳定的视觉语法，而不是来自 3x3 本身。

### 2.1 黑白/灰度几何对象

RAVEN 的 panel 基本是：

```text
white background
thin black/gray outlines
gray-scale filled geometric shapes
no color semantics
no text
no Arabic digits
no operator symbols
```

它的 `Color` attribute 在代码中是灰度值序列，不是彩色类别：

```text
COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
```

所以旧版 VQ-Expr 的彩色圆点是第一处不正常：颜色看起来像人工 codebook token，而不是 RAVEN object attribute。

### 2.2 固定 figure configurations

RAVEN 不让对象自由漂浮。它有固定配置：

```text
Center
2x2Grid
3x3Grid
Left-Right
Up-Down
Out-InCenter
Out-InGrid
```

这些配置给 panel 一个“数据集正常感”：对象在 slots、grid、inside/outside、left/right、up/down 中变化，而不是被摆成解释图。

### 2.3 层级 AoT：Scene -> Structure -> Component -> Layout -> Entity

RAVEN 的结构不是视觉装饰，而是生成语法：

```text
Scene
  Structure
    Component
      Layout
        Entity
```

`Layout` 上有：

```text
Number
Position
Uniformity
```

`Entity` 上有：

```text
Type
Size
Color
Orientation
```

这说明 RAVEN 的“数”和“视觉”结合方式不是把数字画出来，而是让 Number/Position/Type/Size/Color 成为同一个 scene grammar 的属性。

### 2.4 规则不显式画出来

RAVEN 规则是：

```text
Constant
Progression
Arithmetic
Distribute Three
```

但图像里没有 rule label、operator gate、wire、arrow。规则只通过多个 panel 中 attribute 的变化体现。

这对 VQ-Expr 很关键：我们不能在单个 panel 里画出“输入对象 + 输出对象”的计算方向。输出 role 可以存在于 metadata 里，但视觉上应该只是 scene 中的一个 component/slot，不应该像右侧答案节点。

### 2.5 Answer candidates 是完整 panel

RAVEN `.npz` 组织方式是：

```text
image: (16, 160, 160)
first 8 figures = problem matrix/context
last 8 figures = choices
```

代码中也是：

```text
image = imgs[0:8] + answers
```

所以 VQ-Expr 的 `1x3 + 8 choices` 方向是对的。但这只是 presentation shell；shell 对了，不代表视觉 primitive 对了。

### 2.6 RAVEN-FAIR 的候选教训

RAVEN-FAIR 指出：如果 negative candidates 由 correct answer 随机改一个属性生成，correct choice 可能在 choice graph 中处于特殊位置，模型只看 candidates 就能猜答案。

对 VQ-Expr 的约束：

```text
full-panel candidates 必须保留；
negative 不能随机凑；
correct slot 必须均衡；
candidate-only audit 必须保留；
correct vs negative 的视觉统计不能泄漏答案。
```

## 3. 旧版 VQ-Expr 为什么不像 RAVEN

旧版图像的主要问题不是 `1x3`，而是 panel 内的 visual primitive。

| 当前设计 | 问题 | RAVEN 对照 |
| --- | --- | --- |
| 彩色圆点表示 local token | 像 codebook/legend，不像灰度几何 attribute | RAVEN color 是灰度，shape/size/count/position 是主属性 |
| 多个圆点围成 role cluster | 像节点图或机制图 | RAVEN 是 object scene，不是 computation graph |
| 右侧 output 位置固定 | 暗示输入到输出的函数卡片 | RAVEN 缺失的是 panel，不是 panel 内的右侧答案节点 |
| quantity family 包括 bar/path/dial | 像独立读数器，不像统一 scene grammar | RAVEN 的属性都附着在 Entity/Layout 上 |
| role layout 直接表达 operand binding | 太接近机制说明 | RAVEN 的 role 来自 Structure/Component/Layout 层级 |
| 彩色 family 跨候选变化明显 | 容易形成候选视觉捷径 | RAVEN-FAIR 要求 candidate-only bias audit |

结论：

```text
旧版 renderer 是“无数字机制图”，不是“RAVEN-like visual reasoning panel”。
```

## 4. VQ-Expr 新视觉语法：Fuzzy Attribute RPM

推荐把 VQ-Expr 从“visual quantity glyph”改为“RAVEN-like object attribute”。

### 4.1 Panel 结构

每个 panel 是一个 scene，不是一个公式卡片。

```text
panel
  structure template: implicit role slots in 1x3 panels
  components: operand roles and target role
  layout attributes: number, position, occupancy
  entity attributes: size or gray
```

视觉上只出现：

```text
几何形状
灰度填充
大小差异
数量差异
位置/slot 差异
```

不出现：

```text
彩色 token
bar chart
dial
path beads
right-side output node
operator-like grouping
debug labels
role-specific shape
internal structure lines
```

### 4.2 1-9 数值如何编码

保留 1-9，但把它们做成 RAVEN 属性：

| Attribute family | 视觉表面 | 1-9 表达 | fuzzy 来源 |
| --- | --- | --- | --- |
| Count | 1-9 个同类几何对象 | Layout.Number | 遮挡/小尺寸/边界 entity 造成相邻数 membership |
| Position set | 3x3 slots 中占据若干位置 | Layout.Position pattern rank | slot 偏移、边界位置、近邻 slot membership |
| Size level | 一个或多个对象的离散大小 | Entity.Size level 1-9 | 连续尺寸落在相邻 bin 边界 |
| Gray level | 几何对象灰度深浅 | Entity.Color/gray level 1-9 | 灰度接近相邻 level |

这些属性都是 RAVEN-like 的；当前主 renderer 只保留这四类。Nested ratio 可以作为后续 fuzzy 连续性实验分支，但不进入极简主视觉。

### 4.3 Role binding 怎么做

不要用颜色、右侧输出点、role-specific shape 表示角色。角色来自固定 slot identity：

| Slot pattern | 可绑定角色 |
| --- | --- |
| upper slot | q1 or high-scope role |
| middle-left slot | q2 |
| middle-right slot | q3 |
| lower slot | target or q4, depending on AoT family |
| extra lower/side slot | additional operand for 4-leaf families |

同一个数值 attribute 在不同 structure 中绑定到不同 role，表达式语义改变：

```text
same visible value v
  in outer component -> q_outer
  in inner component -> q_inner
  in grid corner -> q_corner
  in center slot -> q_center

AoT uses role identity, not raw visual value alone.
```

这就是我们和 RAVEN 的真正区别：

```text
RAVEN:
  role/attribute -> discrete relation rule

VQ-Expr:
  role/attribute -> fuzzy quantity -> arithmetic AoT
```

### 4.4 Candidate panel 怎么画

候选仍然是 8 个 full panels，但它们必须像 RAVEN answer choices：

```text
same structure template
same query role layout
same non-target attributes controlled
only target role's semantic attribute changes
```

候选不能像“答案 token”。每个 candidate 是完整 scene，只是 target component 的 attribute 取值不同或违反一个 controlled semantic mutation。

## 5. 三种改造路线

### A. RAVEN-Strict Fuzzy Attribute Renderer

完全放弃当前彩色 token/bar/path/dial 表面。只用白底、外框、同一种灰度圆形对象、固定 role slots、entity/layout attributes。

优点：

```text
最像正常 AVR 数据集；
最容易向审稿人解释；
能直接对照 RAVEN/PGM/RAVEN-FAIR；
视觉捷径更容易审计。
```

风险：

```text
fuzzy quantity 需要通过灰度/大小/position 边界设计体现；
实现需要重写 renderer。
```

建议：采用。

### B. Hybrid Quantity-as-RAVEN Renderer

保留当前五类 quantity family 的 decoder，但把视觉表面重画成 RAVEN 属性。例如 `metric_scale` 不再画 bar，而是画 size level；`local_place_value_codebook` 不再画彩色 token，而是画 gray/shape code。

优点：

```text
保留部分 metadata 和 decoder；
实现成本低一点。
```

风险：

```text
仍可能像编码表；
论文里解释会显得不够干净。
```

### C. Full RPM Matrix Renderer

改成真正 3x3 RAVEN matrix。

优点：

```text
视觉上最容易被识别为 RPM。
```

风险：

```text
偏离用户指定的 1x3；
会把研究问题拉回 RAVEN 本身；
VQ-Expr 的 arithmetic AoT 贡献被稀释。
```

不建议。

## 6. 推荐新版数据样式

推荐最终样式：

```text
Top:
  [context scene 1] [context scene 2] [context scene 3]

Bottom:
  8 full candidate scenes in 4x2 grid

All panels:
  same RAVEN-like structure family
  grayscale geometric objects
  no explicit output marker
  no visible digits/operators/text
  target role is only known through metadata
```

一个 nested 表达式样例可以这样设计：

```text
structure = Out-InGrid

outer polygon gray/size/count -> q_outer
inner 2x2 or 3x3 group -> q_inner_a, q_inner_b
one target component/slot -> q_target

hidden AoT:
  q_target = clamp_1_9(q_outer - fuzzy_add(q_inner_a, q_inner_b))

context panels:
  three scenes satisfying this relation

candidates:
  same query scene with target component varied
```

视觉上，用户只看到几何对象的数量、大小、灰度、位置变化；不会看到“这是输入，这是输出”的机制说明。

## 7. 下一步实现验收标准

新版 renderer 生成的第一张图必须通过以下人工 checklist：

```text
1. 第一眼像 RAVEN/PGM/RPM panel，而不是节点图。
2. 全图只有黑白/灰度几何对象。
3. 1x3 上方三个都是 context scene，没有 missing/query slot。
4. 8 个候选都是完整 scene，和 context 同构。
5. 单个 panel 内没有明显“输入 -> 输出”的方向。
6. 数值信息来自 count/position/size/gray/nested ratio。
7. role binding 来自 structure/component/slot，不来自颜色标签。
8. metadata oracle 仍能复算唯一正确答案。
9. candidate-only audit 仍接近 chance。
```

## 8. Claim-Evidence Ledger

| Claim | Evidence now | Status | Allowed wording |
| --- | --- | --- | --- |
| 当前 VQ-Expr 的 layout shell 已接近 RAVEN 的 context/choice split。 | generator 已输出 3 context + 8 candidates；RAVEN code 使用 context + choices。 | supported | 可以说“presentation organization is aligned”。 |
| 当前 VQ-Expr 视觉表面像 RAVEN。 | 极简灰度原型已生成，但仍需人工确认。 | needs weaker wording | 可以说“closer to RAVEN-style minimal object panels”。 |
| RAVEN 的视觉正常感来自 object-attribute scene grammar。 | RAVEN paper/code: A-SIG, 7 configurations, layout/entity attributes。 | supported | 可以说“RAVEN-style grammar suggests...”。 |
| VQ-Expr 新贡献不是 RAVEN 换皮。 | 需要新版 renderer + fuzzy AoT audit。 | needs probe | 只能说“intended distinction is...”。 |
| 新版数据集没有 candidate-only shortcut。 | 旧版 200-sample linear probe 低于 chance，但新版未生成。 | needs probe | 新版生成后再说。 |

## 9. 一句话研究定位

```text
VQ-Expr should not be a colored visual-code arithmetic task;
it should be a RAVEN-like object-attribute reasoning task where fuzzy visual quantities,
bound by scene structure, execute a hidden arithmetic AoT.
```

## 10. 主要参考

- RAVEN: A Dataset for Relational and Analogical Visual rEasoNing, CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_RAVEN_A_Dataset_for_Relational_and_Analogical_Visual_REasoNing_CVPR_2019_paper.html
- RAVEN code: https://github.com/WellyZhang/RAVEN
- RAVEN-FAIR / Scale-Localized Abstract Reasoning, CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Benny_Scale-Localized_Abstract_Reasoning_CVPR_2021_paper.html
- PGM / Measuring abstract reasoning in neural networks, ICML 2018. https://proceedings.mlr.press/v80/barrett18a.html
