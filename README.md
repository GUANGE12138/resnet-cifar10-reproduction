# ResNet 论文复现项目

本项目用于课程自选题中的 **Deep Residual Learning for Image Recognition / ResNet** 论文复现。

本项目目标不是单纯刷高准确率，而是围绕 ResNet 论文的核心问题，完成模型实现、对照实验、消融实验、失败分析和合理改进尝试。

---

## 1. 复现主线

ResNet 论文关注的核心问题是：**深层 plain CNN 在继续加深后，训练误差和验证误差可能反而变差**。这种现象不是典型过拟合，而是深层 plain network 的优化退化问题，即 **degradation problem**。

ResNet 通过 **residual learning** 和 **shortcut connection**，将目标映射从直接学习：

```text
H(x)
```

改写为学习：

```text
H(x) = F(x) + x
```

也就是说，网络不再强迫若干层直接拟合完整映射，而是学习相对于输入的残差修正量 `F(x)`。当最优映射接近 identity mapping 时，残差分支只需要学习接近 0 的映射，从而降低深层网络的优化难度。

本项目主要复现以下结论：

1. **PlainNet-34 相比 PlainNet-18 出现性能下降**，说明 plain network 直接加深可能导致退化。
2. **ResNet-34 相比 ResNet-18 没有明显退化**，说明 residual connection 能帮助深层网络保持可训练性。
3. **ResNet-34 明显优于 PlainNet-34**，说明性能提升主要来自 residual shortcut。
4. **小数据场景下 ResNet 仍可能过拟合**，说明 residual connection 主要解决优化问题，不能自动解决数据不足导致的泛化问题。

---

## 2. 项目结构

```text
resnet-cifar10-reproduction/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── eval.py
├── configs/
│   ├── cifar10_plain18.yaml
│   ├── cifar10_plain34.yaml
│   ├── cifar10_resnet18.yaml
│   ├── cifar10_resnet34.yaml
│   ├── ablation_resnet34_no_shortcut.yaml
│   ├── ablation_resnet34_adamw_cosine.yaml
│   └── failure_resnet34_small_data.yaml
├── datasets/
│   ├── __init__.py
│   └── dataloaders.py
├── models/
│   ├── __init__.py
│   └── resnet.py
├── utils/
│   ├── checkpoint.py
│   ├── config.py
│   ├── logger.py
│   ├── metrics.py
│   └── seed.py
├── analysis/
│   ├── plot_curves.py
│   └── summarize_results.py
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── tables/
└── report/
    ├── report_resnet_reproduction.md
    └── project_checklist_completed.md
```

各目录说明：

| 路径 | 作用 |
|---|---|
| `configs/` | 保存所有实验配置文件 |
| `datasets/` | CIFAR-10 数据读取与训练/验证划分 |
| `models/` | PlainNet 和 ResNet 模型实现 |
| `utils/` | 配置读取、日志、指标、随机种子、checkpoint 工具 |
| `analysis/` | 训练曲线绘制和结果表格汇总 |
| `outputs/figures/` | 保存训练曲线图 |
| `outputs/logs/` | 保存训练日志 CSV |
| `outputs/tables/` | 保存实验结果汇总表 |
| `report/` | 保存复现报告和自检清单 |
| `train.py` | 训练入口 |
| `eval.py` | 模型评估入口 |

本项目不上传 CIFAR-10 原始数据和模型 checkpoint。CIFAR-10 可由 `torchvision.datasets.CIFAR10(download=True)` 自动下载。

---

## 3. 环境安装

推荐使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，请确认 PyTorch CUDA 可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

若输出为：

```text
True
```

说明 GPU 可用于训练。

---

## 4. 推荐运行顺序

### 4.1 主实验：PlainNet vs ResNet

```bash
python train.py --config configs/cifar10_plain18.yaml
python train.py --config configs/cifar10_plain34.yaml
python train.py --config configs/cifar10_resnet18.yaml
python train.py --config configs/cifar10_resnet34.yaml
```

四个主实验分别对应：

| 实验编号 | 模型 | 目的 |
|---|---|---|
| E1 | PlainNet-18 | 浅层 plain baseline |
| E2 | PlainNet-34 | 验证 plain 网络加深后的退化 |
| E3 | ResNet-18 | 浅层 residual baseline |
| E4 | ResNet-34 | 验证 residual 网络加深后是否稳定 |

### 4.2 消融实验

```bash
python train.py --config configs/ablation_resnet34_no_shortcut.yaml
python train.py --config configs/ablation_resnet34_adamw_cosine.yaml
```

消融实验包括：

| 实验编号 | 模型 | 目的 |
|---|---|---|
| A1 | ResNet-34 no shortcut | 验证 shortcut connection 是否关键 |
| A2 | ResNet-34 + AdamW/Cosine | 验证优化器和学习率策略是否带来提升 |

### 4.3 Failure Analysis：小数据实验

```bash
python train.py --config configs/failure_resnet34_small_data.yaml
```

该实验只使用较少训练数据，用于分析 ResNet 在数据不足时是否会发生过拟合。

### 4.4 生成结果表和训练曲线

生成主实验结果表：

```bash
python analysis/summarize_results.py \
  --logs outputs/logs/cifar10_plain18.csv outputs/logs/cifar10_plain34.csv outputs/logs/cifar10_resnet18.csv outputs/logs/cifar10_resnet34.csv \
  --labels PlainNet-18 PlainNet-34 ResNet-18 ResNet-34 \
  --out outputs/tables/main_results.csv
```

绘制验证准确率曲线：

```bash
python analysis/plot_curves.py \
  --logs outputs/logs/cifar10_plain18.csv outputs/logs/cifar10_plain34.csv outputs/logs/cifar10_resnet18.csv outputs/logs/cifar10_resnet34.csv \
  --labels PlainNet-18 PlainNet-34 ResNet-18 ResNet-34 \
  --metric val_acc \
  --out outputs/figures/main_val_acc.png
```

绘制训练 loss 曲线：

```bash
python analysis/plot_curves.py \
  --logs outputs/logs/cifar10_plain18.csv outputs/logs/cifar10_plain34.csv outputs/logs/cifar10_resnet18.csv outputs/logs/cifar10_resnet34.csv \
  --labels PlainNet-18 PlainNet-34 ResNet-18 ResNet-34 \
  --metric train_loss \
  --out outputs/figures/main_train_loss.png
```

绘制验证 loss 曲线：

```bash
python analysis/plot_curves.py \
  --logs outputs/logs/cifar10_plain18.csv outputs/logs/cifar10_plain34.csv outputs/logs/cifar10_resnet18.csv outputs/logs/cifar10_resnet34.csv \
  --labels PlainNet-18 PlainNet-34 ResNet-18 ResNet-34 \
  --metric val_loss \
  --out outputs/figures/main_val_loss.png
```

---

## 5. 实验结果

### 5.1 主实验结果

| Model | Best Epoch | Best Val Acc (%) | Final Train Acc (%) | Final Val Acc (%) |
|---|---:|---:|---:|---:|
| PlainNet-18 | 93 | 94.12 | 99.96 | 93.88 |
| PlainNet-34 | 100 | 92.06 | 99.83 | 91.72 |
| ResNet-18 | 111 | 94.30 | 99.98 | 94.02 |
| ResNet-34 | 90 | 94.40 | 99.98 | 94.40 |

主实验结果显示，PlainNet-34 相比 PlainNet-18 下降 **2.06%**，说明 plain network 加深后出现性能退化。相比之下，ResNet-34 相比 ResNet-18 没有退化，并且 ResNet-34 比 PlainNet-34 高 **2.34%**，支持 ResNet 论文中 residual connection 缓解深层网络优化退化的核心结论。

### 5.2 Shortcut 消融结果

| Model | Best Val Acc (%) | Final Val Acc (%) |
|---|---:|---:|
| ResNet-34 | 94.40 | 94.40 |
| ResNet-34 no shortcut | 91.84 | 91.36 |

移除 shortcut 后，ResNet-34 的最佳验证准确率从 **94.40%** 降至 **91.84%**，说明 shortcut connection 是性能提升的关键因素。

### 5.3 Optimizer / Scheduler 消融结果

| Model | Best Val Acc (%) | Final Val Acc (%) |
|---|---:|---:|
| ResNet-34 baseline | 94.40 | 94.40 |
| ResNet-34 AdamW + Cosine | 93.92 | 93.48 |

AdamW + Cosine 没有超过原始 SGD + MultiStep 设置，说明当前实验中的主要性能收益来自 residual architecture，而不是优化器替换。

### 5.4 Failure Analysis 结果

| Model | Best Val Acc (%) | Final Train Acc (%) | Final Val Acc (%) |
|---|---:|---:|---:|
| ResNet-34 full data | 94.40 | 99.98 | 94.40 |
| ResNet-34 small data | 66.38 | 98.69 | 65.34 |

小数据实验中，ResNet-34-small-data 的最终训练准确率达到 **98.69%**，但最终验证准确率只有 **65.34%**。这说明模型可以拟合训练集，但泛化能力明显不足，主要问题是 **overfitting / data scarcity**，而不是 optimization issue。

---

## 6. 报告

完整复现报告位于：

```text
report/report_resnet_reproduction.md
```

报告包含：

- ResNet 论文核心思想理解；
- PlainNet 与 ResNet 的主实验对比；
- shortcut、深度、optimizer/scheduler 消融；
- small-data failure analysis；
- 最终复现结论。

---

## 7. 项目范围说明
本项目已完成：

- ResNet 论文核心思想梳理；
- PlainNet-18 / PlainNet-34 / ResNet-18 / ResNet-34 实现；
- CIFAR-10 主实验；
- shortcut 消融；
- optimizer/scheduler 消融；
- small-data failure analysis；
- 结果图表与复现报告。
