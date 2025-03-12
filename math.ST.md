# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Random Geometric Graph Alignment with Graph Neural Networks](https://arxiv.org/abs/2402.07340) | 本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。 |
| [^2] | [Ledoit-Wolf linear shrinkage with unknown mean.](http://arxiv.org/abs/2304.07045) | 本文研究了在未知均值下的大维协方差矩阵估计问题，并提出了一种新的估计器，证明了其二次收敛性，在实验中表现优于其他标准估计器。 |

# 详细

[^1]: 用图神经网络对随机几何图进行对齐

    Random Geometric Graph Alignment with Graph Neural Networks

    [https://arxiv.org/abs/2402.07340](https://arxiv.org/abs/2402.07340)

    本文研究了在图对齐问题中，通过图神经网络可以高概率恢复正确的顶点对齐。通过特定的特征稀疏性和噪声水平条件，我们证明了图神经网络的有效性，并与直接匹配方法进行了比较。

    

    我们研究了在顶点特征信息存在的情况下，图神经网络在图对齐问题中的性能。具体而言，给定两个独立扰动的单个随机几何图以及噪声稀疏特征的情况下，任务是恢复两个图的顶点之间的未知一对一映射关系。我们证明在特征向量的稀疏性和噪声水平满足一定条件的情况下，经过精心设计的单层图神经网络可以在很高的概率下通过图结构来恢复正确的顶点对齐。我们还证明了噪声水平的条件上界，仅存在对数因子差距。最后，我们将图神经网络的性能与直接在噪声顶点特征上求解分配问题进行了比较。我们证明了当噪声水平至少为常数时，这种直接匹配会导致恢复不完全，而图神经网络可以容忍n

    We characterize the performance of graph neural networks for graph alignment problems in the presence of vertex feature information. More specifically, given two graphs that are independent perturbations of a single random geometric graph with noisy sparse features, the task is to recover an unknown one-to-one mapping between the vertices of the two graphs. We show under certain conditions on the sparsity and noise level of the feature vectors, a carefully designed one-layer graph neural network can with high probability recover the correct alignment between the vertices with the help of the graph structure. We also prove that our conditions on the noise level are tight up to logarithmic factors. Finally we compare the performance of the graph neural network to directly solving an assignment problem on the noisy vertex features. We demonstrate that when the noise level is at least constant this direct matching fails to have perfect recovery while the graph neural network can tolerate n
    
[^2]: Ledoit-Wolf线性收缩方法在未知均值的情况下的应用(arXiv:2304.07045v1 [math.ST])

    Ledoit-Wolf linear shrinkage with unknown mean. (arXiv:2304.07045v1 [math.ST])

    [http://arxiv.org/abs/2304.07045](http://arxiv.org/abs/2304.07045)

    本文研究了在未知均值下的大维协方差矩阵估计问题，并提出了一种新的估计器，证明了其二次收敛性，在实验中表现优于其他标准估计器。

    

    本研究探讨了在未知均值下的大维协方差矩阵估计问题。当维数和样本数成比例并趋向于无穷大时，经验协方差估计器失效，此时称为Kolmogorov渐进性。当均值已知时，Ledoit和Wolf（2004）提出了一个线性收缩估计器，并证明了在这些演进下的收敛性。据我们所知，当均值未知时，尚未提出正式证明。为了解决这个问题，我们提出了一个新的估计器，并在Ledoit和Wolf的假设下证明了它的二次收敛性。最后，我们通过实验证明它胜过了其他标准估计器。

    This work addresses large dimensional covariance matrix estimation with unknown mean. The empirical covariance estimator fails when dimension and number of samples are proportional and tend to infinity, settings known as Kolmogorov asymptotics. When the mean is known, Ledoit and Wolf (2004) proposed a linear shrinkage estimator and proved its convergence under those asymptotics. To the best of our knowledge, no formal proof has been proposed when the mean is unknown. To address this issue, we propose a new estimator and prove its quadratic convergence under the Ledoit and Wolf assumptions. Finally, we show empirically that it outperforms other standard estimators.
    

