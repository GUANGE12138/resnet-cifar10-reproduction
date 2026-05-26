# ResNet 复现实验记录表（已填写）

| Exp ID | Date | Config | Model | Dataset | Key Hyperparams | Best Val Acc | Problem / Observation | Owner |
|---|---|---|---|---|---|---:|---|---|
| E1 | 2026-05-26 | configs/cifar10_plain18.yaml | PlainNet-18 | CIFAR-10 | SGD lr=0.1, 120 epochs | 94.12 | 浅层 plain baseline，性能较好 | 待填写 |
| E2 | 2026-05-26 | configs/cifar10_plain34.yaml | PlainNet-34 | CIFAR-10 | SGD lr=0.1, 120 epochs | 92.06 | 加深后低于 PlainNet-18，出现退化 | 待填写 |
| E3 | 2026-05-26 | configs/cifar10_resnet18.yaml | ResNet-18 | CIFAR-10 | SGD lr=0.1, 120 epochs | 94.30 | residual baseline，略高于 PlainNet-18 | 待填写 |
| E4 | 2026-05-26 | configs/cifar10_resnet34.yaml | ResNet-34 | CIFAR-10 | SGD lr=0.1, 120 epochs | 94.40 | 深层 residual 网络未退化，优于 PlainNet-34 | 待填写 |
| A1 | 2026-05-26 | configs/ablation_resnet34_no_shortcut.yaml | ResNet-34 no shortcut | CIFAR-10 | SGD lr=0.1, 120 epochs | 91.84 | 移除 shortcut 后下降 2.56%，说明 shortcut 关键 | 待填写 |
| A2 | 2026-05-26 | configs/ablation_resnet34_adamw_cosine.yaml | ResNet-34 | CIFAR-10 | AdamW + cosine | 93.92 | 未超过 SGD baseline，是负结果 | 待填写 |
| F1 | 2026-05-26 | configs/failure_resnet34_small_data.yaml | ResNet-34 | CIFAR-10 small-data | SGD lr=0.1, 120 epochs | 66.38 | train acc 高但 val acc 低，归因为 overfitting / data scarcity | 待填写 |

## 单次实验记录示例：E4

- 日期：2026-05-26
- 负责人：待填写
- Git commit：待填写
- 配置文件：configs/cifar10_resnet34.yaml
- 运行命令：`python train.py --config configs/cifar10_resnet34.yaml`
- 主要目的：验证 residual network 加深后是否保持稳定并优于 PlainNet-34
- 实验结果：Best Val Acc = 94.40%，Final Train Acc = 99.98%，Final Val Acc = 94.40%
- 观察现象：ResNet-34 未出现 PlainNet-34 的退化，明显优于 PlainNet-34
- 下一步：基于 shortcut 消融和 small-data failure analysis 完成报告
