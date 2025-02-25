# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Expressive Power of Spectral Graph Neural Networks with Eigenvalue Correction](https://arxiv.org/abs/2401.15603) | 该论文提出了一种特征值修正策略，可以提升谱图神经网络的表达能力，使多项式滤波器摆脱重复特征值输入的限制，并增强了特征值的均匀分布。 |

# 详细

[^1]: 用特征值修正提升谱图神经网络的表达能力

    Improving Expressive Power of Spectral Graph Neural Networks with Eigenvalue Correction

    [https://arxiv.org/abs/2401.15603](https://arxiv.org/abs/2401.15603)

    该论文提出了一种特征值修正策略，可以提升谱图神经网络的表达能力，使多项式滤波器摆脱重复特征值输入的限制，并增强了特征值的均匀分布。

    

    在最近几年中，特征为多项式滤波器的谱图神经网络越来越受到关注，在节点分类等任务中取得了显著的表现。这些模型通常假设规范化拉普拉斯矩阵的特征值彼此不同，因此期望多项式滤波器具有很高的拟合能力。然而，本文在实证上观察到规范化拉普拉斯矩阵经常具有重复的特征值。此外，我们从理论上建立了可辨认特征值的数量在确定谱图神经网络的表达能力方面起着关键作用。鉴于这一观察结果，我们提出了一种特征值修正策略，可以使多项式滤波器摆脱重复特征值输入的限制。具体而言，所提出的特征值修正策略增强了特征值的均匀分布，从而减轻了谱图神经网络的表达能力受限的问题。

    arXiv:2401.15603v2 Announce Type: replace  Abstract: In recent years, spectral graph neural networks, characterized by polynomial filters, have garnered increasing attention and have achieved remarkable performance in tasks such as node classification. These models typically assume that eigenvalues for the normalized Laplacian matrix are distinct from each other, thus expecting a polynomial filter to have a high fitting ability. However, this paper empirically observes that normalized Laplacian matrices frequently possess repeated eigenvalues. Moreover, we theoretically establish that the number of distinguishable eigenvalues plays a pivotal role in determining the expressive power of spectral graph neural networks. In light of this observation, we propose an eigenvalue correction strategy that can free polynomial filters from the constraints of repeated eigenvalue inputs. Concretely, the proposed eigenvalue correction strategy enhances the uniform distribution of eigenvalues, thus mit
    

