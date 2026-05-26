# ResNet 复现实验协议

为了保证实验公平，所有主实验必须共享同一套训练设置。

## 1. 公平对照原则

除被研究因素外，其余设置保持一致：

- 同一数据划分
- 同一 batch size
- 同一 optimizer
- 同一 learning rate schedule
- 同一数据增强
- 同一训练 epoch
- 同一 random seed

## 2. 主实验

| 实验 | 控制变量 | 改变变量 |
|---|---|---|
| PlainNet-18 vs PlainNet-34 | 训练设置一致 | 深度 |
| ResNet-18 vs ResNet-34 | 训练设置一致 | 深度 |
| PlainNet-34 vs ResNet-34 | 深度接近，训练设置一致 | shortcut |

## 3. 判断标准

### Optimization Issue

如果更深模型的 train loss 更高、train accuracy 更低，说明它不是泛化问题，而是训练优化本身更难。

### Overfitting

如果 train accuracy 很高但 val accuracy 很低，说明模型记住训练集但泛化差。

### Distribution Shift

如果 clean validation 表现好，但 corrupted / shifted validation 表现明显下降，说明存在分布偏移敏感性。

## 4. 结果记录

每个实验必须记录：

- config 文件
- Git commit hash
- 最好验证准确率
- 最终训练准确率
- 最终验证准确率
- 训练曲线
- 失败或异常现象
