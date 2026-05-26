# Failure Analysis：小数据训练下 ResNet-34 验证准确率下降

## Failure Case 名称

小数据训练下 ResNet-34 出现严重过拟合，验证准确率明显下降。

## 1. 现象

- 模型：ResNet-34
- 数据条件：CIFAR-10 small-data setting，只使用部分训练数据进行训练。
- 训练设置：与主实验保持相同的训练流程，使用 SGD optimizer，初始学习率 0.1，训练 120 epochs。
- 观察结果：

| Model | Best Epoch | Best Val Acc | Final Train Acc | Final Val Acc |
|---|---:|---:|---:|---:|
| ResNet-34-full-data | 90 | 94.40% | 99.98% | 94.40% |
| ResNet-34-small-data | 119 | 66.38% | 98.69% | 65.34% |

在完整数据集上，ResNet-34 的最佳验证准确率为 94.40%。但在 small-data setting 下，最佳验证准确率下降到 66.38%，相比 full-data baseline 下降了 28.02%。同时，small-data setting 的最终训练准确率达到 98.69%，但最终验证准确率只有 65.34%。

这说明模型可以很好地拟合训练集，但无法泛化到验证集。

## 2. 初始假设

本 failure case 的主要假设是：

- Overfitting：训练准确率很高，但验证准确率明显偏低。
- Data scarcity：训练样本过少，导致模型学习到训练集中的局部模式，而不是稳定、可泛化的视觉特征。

不支持的假设：

- Optimization issue：如果是优化问题，训练准确率本身应该也上不去。但 small-data setting 下最终训练准确率达到 98.69%，说明模型并不是无法优化训练集。
- Distribution shift：本实验没有改变验证集分布，因此不能归因为训练/验证分布不一致。
- Label noise：本实验没有人为加入标签噪声，因此不是主要原因。
- Class imbalance：本实验没有专门构造类别不平衡，因此暂不作为主要归因。

## 3. 验证实验

| 验证实验 | 预期结果 | 实际结果 | 是否支持假设 |
|---|---|---|---|
| 对比 full-data 和 small-data 的验证准确率 | small-data 验证准确率显著低于 full-data | full-data best val acc = 94.40%，small-data best val acc = 66.38%，下降 28.02% | 支持 data scarcity |
| 查看 train acc 和 val acc 差距 | 如果过拟合，train acc 高但 val acc 低 | small-data final train acc = 98.69%，final val acc = 65.34%，差距约 33.35% | 强烈支持 overfitting |
| 判断是否为 optimization issue | 如果是 optimization issue，训练准确率应较低 | small-data train acc 接近 99%，说明训练集可被拟合 | 不支持 optimization issue |
| 对比 ResNet-34 主实验 | 如果问题来自小数据，完整数据下应表现正常 | 完整数据下 ResNet-34 best val acc = 94.40%，表现稳定 | 支持小数据导致泛化失败 |
| 画 confusion matrix | 预期部分类别会被错误集中混淆 | 本次未执行，可作为后续补充 | 暂无结论 |
| 使用更强数据增强 | 预期可以缓解过拟合，提高验证准确率 | 本次未执行，可作为后续改进实验 | 后续验证 |
| 使用 class-balanced loss | 如果类别不平衡明显，可能改善少数类表现 | 本次未构造类别不平衡，不作为当前主要实验 | 暂不适用 |

## 4. 结论

本 failure case 的最终归因是：small-data setting 下的性能下降主要来自 overfitting 和训练数据不足，而不是 optimization issue。

证据是：ResNet-34-small-data 的最终训练准确率达到 98.69%，说明模型可以很好地拟合训练集；但最终验证准确率只有 65.34%，说明模型泛化能力很差。相比完整数据下 94.40% 的最佳验证准确率，小数据设置下降了 28.02%。

因此，residual connection 可以缓解深层网络的优化困难，但不能自动解决小数据场景下的泛化问题。换句话说，ResNet 解决的是“深层网络是否容易训练”的问题，而不是“数据不足时是否一定不会过拟合”的问题。

## 5. 改进方向

针对该 failure case，后续可以尝试以下改进：

1. 更强数据增强  
   例如 RandomCrop、RandomHorizontalFlip、ColorJitter、RandomErasing、CutMix 或 MixUp。其目标是增加训练样本的多样性，缓解模型记忆训练集。

2. 正则化方法  
   可以尝试 weight decay 调整、label smoothing、dropout 或 stochastic depth，降低模型过拟合风险。

3. 使用预训练模型  
   在数据较少时，从 ImageNet 或其他大规模数据集预训练的模型进行 fine-tuning，可能比从零训练更稳定。

4. 降低模型容量  
   在 small-data setting 下，ResNet-34 可能容量过大。可以尝试 ResNet-18 或更小模型，观察是否能减少过拟合。

5. 增加验证分析  
   后续可以补充 confusion matrix 和 per-class accuracy，观察错误是否集中在某些类别上。