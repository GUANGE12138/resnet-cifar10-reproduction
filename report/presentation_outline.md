# ResNet 复现 Presentation 大纲（15 分钟）

## Slide 1: Title

- Deep Residual Learning for Image Recognition 复现
- 小组成员和分工

## Slide 2: Motivation

- 为什么深层网络重要？
- 为什么 plain 网络不是越深越好？
- 引出 degradation problem。

## Slide 3: Paper Key Idea

- `H(x)` vs `F(x)+x`
- residual block 图示
- identity shortcut 不增加参数。

## Slide 4: Experimental Setup

- 数据集
- 模型：PlainNet-18/34, ResNet-18/34
- 训练超参数

## Slide 5: Main Reproduction Results

- 主结果表格
- train loss / val acc 曲线
- 是否复现论文核心现象

## Slide 6: Why It Works

- PlainNet-34 训练困难说明 optimization issue
- ResNet-34 更容易优化
- shortcut 的作用不是简单增加参数

## Slide 7: Ablation Study

- shortcut 消融
- depth 消融
- optimizer / augmentation 消融

## Slide 8: Failure Analysis

- 小数据
- 类别不平衡
- 分布偏移
- 判断是 overfitting、distribution shift 还是 optimization issue

## Slide 9: Improvement

- 改进动机
- 方法
- 结果
- 是否有效

## Slide 10: Conclusion

- 复现结论
- 失败边界
- 后续工作

## Slide 11: Team Contribution

- 每个人负责的代码、实验、报告
- GitHub contribution 截图或统计
