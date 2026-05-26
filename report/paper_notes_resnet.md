# ResNet 论文阅读笔记

论文：Deep Residual Learning for Image Recognition  
作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

## 1. 一句话总结

ResNet 通过 residual learning 和 shortcut connection 缓解深层网络中的 degradation problem，使网络加深后仍然容易优化并获得更好性能。

## 2. 论文背景

传统 CNN 加深后理论上表达能力更强，但实际训练中会出现：

- 训练误差不降反升；
- 验证误差也变差；
- 这个现象不是典型过拟合，因为训练误差本身更高；
- 也不完全是 vanishing gradient，因为 BN 和初始化已经缓解了梯度消失。

论文称该现象为 degradation problem。

## 3. 核心方法

原始目标是学习映射：

```text
H(x)
```

ResNet 改为学习残差：

```text
F(x) = H(x) - x
H(x) = F(x) + x
```

如果最优映射接近 identity mapping，那么让残差分支学到接近 0，比让多层非线性网络直接学 identity 更容易。

## 4. BasicBlock

结构：

```text
x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> + x -> ReLU
```

当尺寸或通道数变化时，用：

```text
shortcut = Conv1x1(x)
```

或使用 zero padding 的 identity shortcut。

## 5. 需要复现的关键现象

| 对照 | 预期现象 | 解释 |
|---|---|---|
| PlainNet-18 vs PlainNet-34 | PlainNet-34 不一定更好 | 深层 plain 网络优化困难 |
| ResNet-18 vs ResNet-34 | ResNet-34 更好 | residual learning 支持加深 |
| PlainNet-34 vs ResNet-34 | ResNet-34 明显更好 | shortcut 缓解 degradation |

## 6. 报告中可以强调的理解

- ResNet 不是简单“加深网络”。
- shortcut 的关键价值是优化路径，而不是显著增加参数。
- residual learning 提供了一种接近 identity mapping 的良好参数化方式。
- 如果训练集很小，ResNet 仍可能过拟合；这说明 residual connection 主要解决优化，不自动解决泛化。
