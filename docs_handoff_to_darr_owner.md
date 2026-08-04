# DARR-MNR 学长接手说明

## 1. 分工边界

本仓库的 benchmark 基础设施已经由当前负责人完成，包括任务协议、数据校验、提交格式、统一评测和结果报告。

DARR 主模型负责人需要负责模型训练、推理和论文结果复现，不需要重写 benchmark 评测逻辑。

## 2. 学长需要提供的内容

请提供以下文件或信息：

- 训练入口和配置文件。
- 推理入口，能够对官方 split 生成预测。
- checkpoint 文件或 checkpoint 下载/生成说明。
- 训练环境、依赖、GPU 和随机种子。
- 是否使用外部预训练模型或额外数据。
- 论文主结果与当前复现结果。
- 每个 split 的标准预测 CSV。

预测 CSV 必须严格符合：

```text
sample_id,prediction
prob_000001.npz,3
```

其中 `sample_id` 必须是测试目录中的精确 `.npz` 文件名，`prediction` 必须是 `0..7` 的整数。

## 3. 接入流程

### 第一步：验证数据

```bash
python benchmark.py ProbSet/test_set
```

### 第二步：由 DARR 推理脚本生成预测

模型输出需要转换为 `predictions.csv`。每个测试样本必须恰好出现一次。

### 第三步：生成标准报告

```bash
python benchmark_report.py ProbSet/test_set \
  --predictions predictions.csv \
  --runner darr_official \
  --output runs/darr_official/benchmark_report.json
```

### 第四步：检查报告

报告至少应包含：

- `protocol_version`
- `split_dir`
- `prediction_file`
- `num_samples`
- `correct`
- `accuracy`
- `chance_accuracy`
- `coverage`
- `runner`

## 4. 不得修改的 benchmark 约束

除非同步升级 protocol version，否则不要改变：

- 3 个 context panels 的输入定义。
- 8 个 answer candidates 的定义。
- 标签范围 `0..7`。
- prediction CSV 的列名和样本覆盖规则。
- 官方测试集标签的使用方式。

如果数据生成、split 或标签语义发生变化，必须更新 `benchmark_protocol.md` 并递增协议版本。

## 5. 结果记录要求

每次正式实验建议保存：

```text
runs/<run_name>/
├─ config.json
├─ checkpoint.json
├─ predictions.csv
├─ benchmark_report.json
└─ stdout.log
```

正式结果不得只提交截图或手工填写的准确率；必须提供机器可读预测文件和由官方脚本生成的报告。

## 6. 当前已完成与待办

已完成：

- `benchmark_protocol.md`
- `benchmark.py`
- `benchmark_predict.py`
- `benchmark_baseline.py`
- `benchmark_submission.py`
- `benchmark_run.py`
- `benchmark_report.py`
- benchmark contract tests

待学长完成：

- DARR 训练入口。
- DARR 推理入口。
- checkpoint 和环境说明。
- 论文主结果复现。
- DARR 预测接入标准 CSV。
- 正式结果报告。
