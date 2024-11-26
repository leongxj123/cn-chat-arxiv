# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression](https://arxiv.org/abs/2403.13565) | 提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。 |

# 详细

[^1]: AdaTrans：针对高维回归的特征自适应与样本自适应迁移学习

    AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression

    [https://arxiv.org/abs/2403.13565](https://arxiv.org/abs/2403.13565)

    提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。

    

    我们考虑高维背景下的迁移学习问题，在该问题中，特征维度大于样本大小。为了学习可迁移的信息，该信息可能在特征或源样本之间变化，我们提出一种自适应迁移学习方法，可以检测和聚合特征-wise (F-AdaTrans)或样本-wise (S-AdaTrans)可迁移结构。我们通过采用一种新颖的融合惩罚方法，结合权重，可以根据可迁移结构进行调整。为了选择权重，我们提出了一个在理论上建立，数据驱动的过程，使得 F-AdaTrans 能够选择性地将可迁移的信号与目标融合在一起，同时滤除非可迁移的信号，S-AdaTrans则可以获得每个源样本传递的信息的最佳组合。我们建立了非渐近速率，可以在特殊情况下恢复现有的近最小似乎最优速率。效果证明...

    arXiv:2403.13565v1 Announce Type: cross  Abstract: We consider the transfer learning problem in the high dimensional setting, where the feature dimension is larger than the sample size. To learn transferable information, which may vary across features or the source samples, we propose an adaptive transfer learning method that can detect and aggregate the feature-wise (F-AdaTrans) or sample-wise (S-AdaTrans) transferable structures. We achieve this by employing a novel fused-penalty, coupled with weights that can adapt according to the transferable structure. To choose the weight, we propose a theoretically informed, data-driven procedure, enabling F-AdaTrans to selectively fuse the transferable signals with the target while filtering out non-transferable signals, and S-AdaTrans to obtain the optimal combination of information transferred from each source sample. The non-asymptotic rates are established, which recover existing near-minimax optimal rates in special cases. The effectivene
    

