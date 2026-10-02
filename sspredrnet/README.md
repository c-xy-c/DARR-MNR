# SSPredRNet：RAVEN 自监督预测

基于 [Neural Prediction Errors as a Unified Cue for Abstract Visual Reasoning](https://github.com/ZjjConan/AVR-PredRNet-and-SSPredRNet) 的原版三层预测网络。保留原始神经预测误差和单向训练任务，使用两个组件视图改善复合布局的候选排序。

## 方法

1. 用RAVEN公开布局和连通墨迹分成两个保持位置、尺度的视图。两视图取像素最小值能重建原图。上下文无法可靠分图时使用两份完整图；单个候选无法分图只使该候选回退完整图。
2. 每个视图沿用原版任务：第一行三格及第二行前两格预测已知第六格。第六格为正例，八个答案图像为负例；训练不读取答案索引或XML。
3. 排除与已知第六格逐像素相同的负例。原版margin损失在两个视图及受监督的两个预测块之间平均。
4. 测试用两条支持行分别预测第三行末格，再平均两个视图的候选误差，选择最小者。

分图依赖公开布局先验，不是学到对象发现。center/four/nine使用两份完整图。每题两视图增加计算量；共享BatchNorm保留原版行为。分图、假负例处理、统计变化和额外计算的作用尚未分别消融。

## 环境和数据

在仓库根目录运行。要求Python 3.10+和CUDA GPU。

```bash
python -m pip install -r sspredrnet/requirements.txt
```

验证环境：单RTX4090，PyTorch 2.11.0+cu128、NumPy 2.5.2、OpenCV 5.0.0。数据目录包含七个RAVEN构型文件夹，各有6000个`*_train.npz`、2000个`*_val.npz`、2000个`*_test.npz`。图像为16×160×160，使用最近邻缩放至80像素。

## 训练与评估

```bash
python -m sspredrnet.train \
  --dataset-root /path/to/RAVEN --run-dir sspredrnet/runs/raven \
  --epochs 20 --batch-size 128 --seed 12345 --fp16

python -m sspredrnet.evaluate \
  --dataset-root /path/to/RAVEN --run-dir sspredrnet/runs/raven
```

训练默认Adam lr0.001、weight decay1e-5、margin0.7、dropout0.1。验证集选择`best.pt`，最后一轮保存`final.pt`，完成所有训练轮次后才评估测试集。`last.pt`包含优化器、scaler、加载器及CPU/CUDA随机状态；加`--resume`恢复同一运行，配置必须一致。

仓库中的发布检查点只包含权重及轮次，用于评估。它们经过一次性精简，不能作为`--resume`的训练状态。

```bash
python -m sspredrnet.evaluate \
  --dataset-root /path/to/RAVEN --run-dir sspredrnet/results/raven-20epoch

python -m unittest discover -s sspredrnet/tests -v
```

## 已验证结果

2026-10-03，seed12345，完整20轮，每轮42000题、329批。验证集和测试集均14000题。20轮按原始题计数，网络每批实际处理256个视图题。

| 测试构型 | 原版验证选模 | 分图，验证选中的第17轮 | 分图，第20轮 |
| --- | ---: | ---: | ---: |
| center_single | 77.85% | 95.30% | 86.85% |
| distribute_four | 35.45% | 45.60% | 51.40% |
| distribute_nine | 30.00% | 33.40% | 32.70% |
| in_center_single_out_center_single | 23.60% | 90.10% | 85.90% |
| in_distribute_four_out_center_single | 26.45% | 38.45% | 36.25% |
| left_center_single_right_center_single | 24.45% | 94.70% | 90.50% |
| up_center_single_down_center_single | 23.20% | 95.10% | 90.00% |
| 平均 | 34.43% | **70.38%** | **67.66%** |

最佳测试9853/14000，末轮9472/14000；验证分别70.17%、67.14%。最佳模型通过验证集选择。训练及逐轮验证38.15分钟，原版匹配对照14.06分钟；这不是相同计算量比较。四格、九格和内四外单仍较弱，训练曲线也有明显波动。

历史原版44.90%来自不同GPU/加载协议并按测试集选模；论文57.1%为原版100轮结果。这两项不作为当前20轮匹配对照。

训练记录来自整理前的同一实验，保存于[`results/raven-20epoch/`](results/raven-20epoch/)。整理后已检查初始化、七类输入、推理、FP16损失、梯度及CUDA随机状态与原实现逐位一致，并用精简代码重新评估best/final的完整验证和测试集。记录不代表另跑了一次20轮训练。

## 代码结构

- `data.py`：只读取必要字段的RAVEN加载、规范排序和归一化。
- `views.py`：组件分图和不确定时的回退。
- `layers.py`、`model.py`：原版编码器、三个预测块和唯一自监督损失。
- `train.py`：训练、恢复、验证选模和末尾测试。
- `evaluate.py`：直接调用原生reasoner，重算并核对两个检查点。
- `tests/`：训练不需要答案字段、假负例梯度、视图独立性及恢复状态等检查。

没有历史版本入口、辅助对象头、多数据集注册、监督分类器、目标切换或硬编码服务器启动脚本。公开权重严格匹配当前模型，源权重与发布权重的哈希及整理验证记录见结果目录。

## 来源

原版代码：Lingxiao Yang等，[AVR-PredRNet-and-SSPredRNet](https://github.com/ZjjConan/AVR-PredRNet-and-SSPredRNet)，commit `347ec81cee92e857f51bb6933b1addeee21e6724`。本目录是针对RAVEN分图自监督实验的精简实现；原版参数名称、初始化顺序、预测计算和距离定义得到保留。
