# VG-MNR 实验协议与基线验证方案

日期：2026-07-30

本文档定义 VG-MNR 当前阶段的正式验证方案，用于判断现有数据集是否真正满足 benchmark 目标，而不是仅仅“能生成、能保存、能跑通”。

它回答三个问题：

1. 数据集代码是否已经具备做第一轮基线验证的条件？
2. 应该选哪些具体算法进入实验？
3. 如何根据实验结果判断数据集是否合理、是否需要调整？

---

## 1. 当前版本是否适合做第一步验证？

结论：**适合做第一步验证，但仍应视为 prototype validation set，而不是最终 benchmark release。**

当前 `mnr_dataset.vgmnr_main` 已经满足第一步验证所需的最低条件：

- 四种模式都可稳定生成：`full`、`no_visual`、`no_context`、`collision`
- `accepted / rejected` 统计已可追踪
- `candidate_diagnostics` 可用于检查候选集质量
- `split_buckets` 已能把样本路由到对应 split
- `generation_report.json` 可给出基础诊断
- `collision` 已有稳定的 `operand_swap` 基线

这意味着：

- 可以冻结一个小规模验证版数据集
- 可以开始跑基线算法
- 可以比较不同算法在不同 split 上的表现是否符合设计预期

当前仍不成熟的地方：

- 更复杂的 collision family 仍在扩展中
- 还没有真实模型的系统性基线结果
- 还没有证明该 benchmark 对不同模型具有稳定区分度

---

## 2. 实验目标

本实验不是为了争论“哪个模型最好”，而是为了验证以下期望：

- `full` 下，模型应该能利用完整视觉与上下文信息完成任务
- `no_visual` 下，模型应明显受损，因为视觉关系被削弱
- `no_context` 下，模型应明显受损，因为上下文信息被削弱
- `collision` 下，模型应比 `full` 更容易混淆，因为这是专门构造的冲突对
- 简单 heuristic 不应轻易接近真实模型
- 随机 baseline 应保持最低水平

如果这些预期不成立，说明数据集还需要调整，而不是直接进入论文叙事。

---

## 3. 进入实验的 5 个具体算法

下面这 5 个算法是本轮建议进入实验的具体 baseline。它们不是抽象类别，而是可以实际实现、实际跑的算法。

### 3.1 Random baseline

**输入**：仅接收候选数量信息，不看上下文、不看 query。

**输出**：随机选择 8 选 1 的答案。

**作用**：

- 作为理论下限
- 检查数据集是否存在明显标签泄露

**预期**：

- 准确率接近 12.5%（8-way 随机）

---

### 3.2 Candidate-only heuristic

**输入**：只看 `answer_set_images`，不看上下文和 query。

**实现建议**：

- 对候选图做简单图像统计或固定位置特征提取
- 使用最简单的打分规则或轻量分类器

**作用**：

- 检查候选布局、颜色、边框、位置等是否泄露答案
- 判断当前候选构造是否真的没有 shortcut

**预期**：

- 应明显高于随机，但不应接近真实模型
- 如果它过高，说明候选设计存在偏差

---

### 3.3 Number-only heuristic / numeric shortcut baseline

**输入**：只使用可从样本中抽取出的数值或弱结构信息，不利用完整视觉关系。

**实现建议**：

- 从 metadata 中提取显式数值特征
- 或从图像中用简单 OCR / 数值提取近似恢复数值，再做规则猜测

**作用**：

- 检查任务是否能被“只看数字、不看结构”解决
- 验证视觉绑定是否真的必要

**预期**：

- 在 `full` 可能有一定效果，但不应接近完整模型
- 在 `no_visual` / `collision` 上应明显下降

---

### 3.4 Small CNN baseline

**输入**：整张图像，端到端做 8-way 分类。

**实现建议**：

- 一个轻量级卷积网络
- 将 context/query/candidate 图像编码后做简单融合

**作用**：

- 提供一个真正的图像学习 baseline
- 检查任务是否能被局部纹理或低级特征解决

**预期**：

- 应高于 random 和简单 heuristic
- 但通常低于更强的视觉 transformer / 多模态模型

---

### 3.5 ViT-style or multimodal baseline

**输入**：整套样本，使用更强的视觉编码器或多模态推理模型。

**实现建议**：

- ViT image encoder
- 或现成 VLM / multimodal transformer

**作用**：

- 作为较强 baseline，验证 benchmark 是否能区分更高层次能力
- 检查 `full` / `no_visual` / `no_context` / `collision` 是否呈现合理难度梯度

**预期**：

- 在 `full` 上最好
- 在 `no_visual`、`no_context`、`collision` 上下降
- 如果下降不明显，说明 split 设计还不够有效

---

## 4. 为什么只选这 5 个

这 5 个 baseline 覆盖了三类关键对照：

1. **下限对照**：Random baseline
2. **shortcut 对照**：Candidate-only / Number-only
3. **学习能力对照**：Small CNN / ViT-style or multimodal baseline

这已经足够判断当前 benchmark 是否满足“有区分度、能反映能力层次”的基本要求。

如果一开始就把算法数量扩得太大，反而会让调试成本飙升，不利于判断数据集问题。

---

## 5. 实验设置

### 5.1 数据版本

建议固定一个验证版数据集，不再边跑边改规则。

最低要求：

- `full`
- `no_visual`
- `no_context`
- `collision`

都使用同一套 schema 和报告格式。

### 5.2 评测方式

每个算法都要在每个 split 上跑准确率：

- `full_supervision`
- `visual_ablation`
- `context_ablation`
- `pair_collision`

并额外记录：

- overall accuracy
- per-split accuracy
- candidate-only accuracy
- error breakdown

### 5.3 统计口径

优先使用：

- sample-level accuracy
- pair-level consistency for collision
- rejection statistics for data generation

如果后续做更深分析，可再增加 calibration、confidence gap、oracles 等指标。

---

## 6. 什么样的结果是理想的？

理想结果不是“所有模型都高分”，而是“结果符合你对功能的预期”。

### 6.1 预期排序

一般建议观察下面的相对关系：

```text
Random < Candidate-only / Number-only < Small CNN < ViT-style / multimodal
```

如果这个梯度不成立，说明数据集的诊断力还不足。

### 6.2 分 split 的理想趋势

#### `full_supervision`
- 应是最容易的 split 之一
- 强模型明显优于弱模型

#### `visual_ablation`
- 相比 `full_supervision` 应下降
- 如果没下降，说明视觉关系削弱得不够

#### `context_ablation`
- 相比 `full_supervision` 也应下降
- 如果没下降，说明上下文信息还不够关键

#### `pair_collision`
- 应该比 `full_supervision` 更难，至少不能完全无差别
- 若与 `full_supervision` 持平，说明 collision 不够有挑战

### 6.3 heuristic 的理想表现

- Candidate-only / Number-only 应该明显好于随机
- 但不应接近真实模型
- 如果 heuristic 过强，说明存在 shortcut

---

## 7. 什么样的结果是不合理的？

### 7.1 随机与 heuristic 都很高

说明：

- 任务可能有明显偏差
- 候选构造可能泄露标签
- 数据集存在 shortcut

调整方向：

- 改候选构造
- 检查候选位置/颜色/尺寸泄露
- 降低固定模式偏差

### 7.2 四个 split 表现几乎一样

说明：

- ablation 没有真正改变信息结构
- split 只是名字不同，实质没区分

调整方向：

- 强化视觉必要性
- 强化上下文必要性
- 让 collision 真正引入对照难度

### 7.3 所有模型都接近随机

说明：

- 任务太难或信息不足
- 数据生成过于苛刻

调整方向：

- 降低任务复杂度
- 增强结构可解性
- 提高样本清晰度

### 7.4 强模型和弱模型差距太小

说明：

- benchmark 区分度不足
- 任务太容易或 shortcut 过强

调整方向：

- 增强 collision 复杂度
- 引入更细 split
- 调整候选与负样本构造

---

## 8. 结论

当前 VG-MNR 数据集代码**已经可以支持第一轮正式验证**。建议采用以下实验顺序：

1. 固定当前稳定数据版本
2. 跑 Random baseline
3. 跑 Candidate-only / Number-only heuristic
4. 跑 Small CNN
5. 跑 ViT-style / multimodal baseline
6. 对照四个 split 看能力是否按预期下降

如果结果符合预期，则可以把这版作为“稳定原型基线”。
如果结果不符合预期，再针对 shortcut、ablation、collision 三个方向回改数据集。
