# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards understanding neural collapse in supervised contrastive learning with the information bottleneck method.](http://arxiv.org/abs/2305.11957) | 本文将神经网络崩溃建模为信息瓶颈问题，证明神经网络崩溃导致良好的泛化，特别是当它接近分类问题的最优信息瓶颈解时。 |

# 详细

[^1]: 通过信息瓶颈方法探索监督对比学习中神经网络崩溃的理解

    Towards understanding neural collapse in supervised contrastive learning with the information bottleneck method. (arXiv:2305.11957v1 [cs.LG])

    [http://arxiv.org/abs/2305.11957](http://arxiv.org/abs/2305.11957)

    本文将神经网络崩溃建模为信息瓶颈问题，证明神经网络崩溃导致良好的泛化，特别是当它接近分类问题的最优信息瓶颈解时。

    

    神经网络崩溃是指在超出性能平台训练时，深度神经网络最后一层激活的几何学表现。目前存在的问题包括神经网络崩溃是否会导致更好的泛化，如果是，超出性能平台的训练如何帮助神经网络崩溃。本文将神经网络崩溃建模为信息瓶颈问题，以探究是否存在这样一种紧凑的表示，并发现其与泛化性的关联。我们证明神经网络崩溃导致良好的泛化，特别是当它接近分类问题的最优信息瓶颈解时。最近的研究表明，使用相同的对比损失目标独立训练的两个深度神经网络是线性可识别的，这意味着得到的表示等效于矩阵变换。我们利用线性可识别性来近似信息瓶颈问题的解析解。这个近似表明，当类平均值相等时，最优解非常接近端到端模型，并提供了进一步的理论分析。

    Neural collapse describes the geometry of activation in the final layer of a deep neural network when it is trained beyond performance plateaus. Open questions include whether neural collapse leads to better generalization and, if so, why and how training beyond the plateau helps. We model neural collapse as an information bottleneck (IB) problem in order to investigate whether such a compact representation exists and discover its connection to generalization. We demonstrate that neural collapse leads to good generalization specifically when it approaches an optimal IB solution of the classification problem. Recent research has shown that two deep neural networks independently trained with the same contrastive loss objective are linearly identifiable, meaning that the resulting representations are equivalent up to a matrix transformation. We leverage linear identifiability to approximate an analytical solution of the IB problem. This approximation demonstrates that when class means exh
    

