# DARR-MNR 项目推进进度报告

日期：2026-08-04

## 1. 项目当前状态概览

DARR-MNR 已经从“论文实现仓库”推进到“标准化 benchmark 仓库骨架”。

目前最核心的 benchmark 基础设施已经完成，包括：

- 任务协议
- 数据格式验证
- 提交格式规范
- 统一评测入口
- 结果报告格式
- 随机基线与简单预测入口
- 自动化 contract tests
- README 级别的使用说明
- 学长接手文档

因此，当前仓库已经具备 benchmark 的基本工程闭环。

---

## 2. 已完成工作

### 2.1 Benchmark protocol

已新增并固定：

- `benchmark_protocol.md`

该文档定义了 DARR-MNR 的正式评测协议，主要约束为：

- 输入为 3 个 context panels
- 输出为 8 选 1 的预测
- 标签范围固定为 `0..7`
- 官方预测文件必须为 CSV
- 评测指标为 top-1 accuracy
- chance accuracy 为 `0.125`

该协议还规定了：

- 不能随意更改样本语义
- 不能混用不同版本的数据定义
- 数据生成、split 或标签语义变化时必须升级 protocol version

---

### 2.2 数据验证与评测

已新增：

- `benchmark.py`

作用：

- 验证 `.npz` 样本是否包含必要字段
- 检查 context/candidate 数量是否符合协议
- 检查标签是否在合法范围
- 检查图像是否存在非有限值
- 读取并校验预测 CSV
- 计算 accuracy、coverage、chance accuracy 等指标

这意味着现在任何正式结果都必须经过统一脚本验证，而不是手工统计。

---

### 2.3 提交格式

已新增：

- `benchmark_submission.py`
- `docs_submission_format.md`

该部分定义了模型提交必须满足的格式：

```text
sample_id,prediction
prob_000001.npz,3
```

要求包括：

- 每个样本必须恰好出现一次
- 不允许重复或遗漏
- `prediction` 必须是 `0..7`
- 样本 ID 必须与官方 split 完全一致

此外，`benchmark_submission.py` 可以自动生成模板，便于模型负责人接入。

---

### 2.4 结果报告

已新增：

- `benchmark_report.py`
- `docs_results_report.md`

该部分定义了标准化结果报告格式，报告中至少包含：

- `protocol_version`
- `split_dir`
- `num_samples`
- `correct`
- `accuracy`
- `chance_accuracy`
- `coverage`
- `prediction_file`
- `runner`

这一步的意义是：

- 结果可以机器读取
- 结果可以归档
- 结果可以复查
- 不再依赖截图或手工表述

---

### 2.5 预测与运行入口

已新增：

- `benchmark_predict.py`
- `benchmark_baseline.py`
- `benchmark_run.py`

功能分别是：

- `benchmark_predict.py`：统一预测入口，目前支持 `random` 和 `first` 两种基础策略
- `benchmark_baseline.py`：生成随机 baseline 预测文件
- `benchmark_run.py`：端到端运行，生成预测并输出 JSON 报告

这样 benchmark 的链路已经完整闭环：

```text
split -> validation -> prediction -> scoring -> report
```

---

### 2.6 自动化测试

已新增并通过的测试包括：

- `tests/test_benchmark.py`
- `tests/test_benchmark_run.py`
- `tests/test_submission_format.py`
- `tests/test_benchmark_report.py`
- `tests/test_benchmark_predict.py`

这些测试覆盖了：

- 数据格式验证
- 预测覆盖检查
- 提交模板生成
- 报告字段检查
- 预测文件生成
- 端到端 benchmark 运行

目前相关测试已全部通过。

---

### 2.7 接手说明

已新增：

- `docs_handoff_to_darr_owner.md`

该文档明确了分工：

- 当前负责人：benchmark 基础设施
- 学长：DARR 主模型训练、推理、checkpoint、复现结果

同时明确了学长接手时需要提供的内容，包括：

- 训练入口
- 推理入口
- checkpoint
- 环境说明
- 随机种子
- 外部数据使用政策
- 标准预测文件

---

## 3. 当前仓库已经具备的 benchmark 能力

现在仓库至少具备以下能力：

1. 验证数据是否符合官方规范
2. 统一生成提交模板
3. 接收模型预测并评分
4. 生成标准化结果报告
5. 通过自动化测试保证协议稳定
6. 给模型负责人提供明确接入方式

这意味着 DARR-MNR 不再只是一个“论文附属仓库”，而是一个已经有 benchmark 工程结构的项目。

---

## 4. 当前仍待学长完成的部分

当前还未由 benchmark 基础设施侧完成的内容，主要在 DARR 主模型这边：

- DARR 训练入口
- DARR 推理入口
- checkpoint 管理
- 论文主结果复现
- 正式 baseline / ablation 结果
- 最终 benchmark 成绩表

这些部分将由学长继续推进。

---

## 5. 交接时建议的最小工作流

建议学长后续采用如下流程：

### 第一步：验证官方 split

```bash
python benchmark.py ProbSet/test_set
```

### 第二步：由模型产生预测文件

模型输出必须整理成：

```text
sample_id,prediction
prob_000001.npz,3
```

### 第三步：生成 benchmark report

```bash
python benchmark_report.py ProbSet/test_set --predictions predictions.csv --output benchmark_report.json
```

### 第四步：归档实验结果

建议目录结构：

```text
runs/<run_name>/
├─ config.json
├─ checkpoint.json
├─ predictions.csv
├─ benchmark_report.json
└─ stdout.log
```

---

## 6. 目前的阶段性判断

### 对你负责的部分

已完成，且可以交付给学长。

### 对整个项目

DARR-MNR 已具备 benchmark 的标准化骨架，但最终 benchmark 成果仍需要学长提供主模型训练与正式复现实验结果。

换句话说：

- 你负责的 benchmark 工程部分：已经完成
- 学长负责的模型实验部分：仍待推进

---

## 7. 一句话总结

DARR-MNR 当前已经建立了完整的 benchmark 协议、评测、提交、报告和测试框架，接下来只需要学长把 DARR 主模型的训练与推理接入进来，就可以形成正式的 benchmark 实验闭环。
