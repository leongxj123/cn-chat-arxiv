# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inferring Dynamic Networks from Marginals with Iterative Proportional Fitting](https://arxiv.org/abs/2402.18697) | 通过识别一个生成网络模型，我们建立了一个设置，IPF可以恢复最大似然估计，揭示了关于在这种设置中使用IPF的隐含假设，并可以为IPF的参数估计提供结构相关的误差界。 |

# 详细

[^1]: 从边际推断动态网络的方法：迭代比例拟合

    Inferring Dynamic Networks from Marginals with Iterative Proportional Fitting

    [https://arxiv.org/abs/2402.18697](https://arxiv.org/abs/2402.18697)

    通过识别一个生成网络模型，我们建立了一个设置，IPF可以恢复最大似然估计，揭示了关于在这种设置中使用IPF的隐含假设，并可以为IPF的参数估计提供结构相关的误差界。

    

    来自现实数据约束的常见网络推断问题是如何从时间聚合的邻接矩阵和时间变化边际（即行向量和列向量之和）推断动态网络。先前的方法为了解决这个问题重新利用了经典的迭代比例拟合（IPF）过程，也称为Sinkhorn算法，并取得了令人满意的经验结果。然而，使用IPF的统计基础尚未得到很好的理解：在什么情况下，IPF提供了从边际准确估计动态网络的原则性，以及它在多大程度上估计了网络？在这项工作中，我们确定了这样一个设置，通过识别一个生成网络模型，IPF可以恢复其最大似然估计。我们的模型揭示了关于在这种设置中使用IPF的隐含假设，并使得可以进行新的分析，如有关IPF参数估计的结构相关误差界。当IPF失败时

    arXiv:2402.18697v1 Announce Type: cross  Abstract: A common network inference problem, arising from real-world data constraints, is how to infer a dynamic network from its time-aggregated adjacency matrix and time-varying marginals (i.e., row and column sums). Prior approaches to this problem have repurposed the classic iterative proportional fitting (IPF) procedure, also known as Sinkhorn's algorithm, with promising empirical results. However, the statistical foundation for using IPF has not been well understood: under what settings does IPF provide principled estimation of a dynamic network from its marginals, and how well does it estimate the network? In this work, we establish such a setting, by identifying a generative network model whose maximum likelihood estimates are recovered by IPF. Our model both reveals implicit assumptions on the use of IPF in such settings and enables new analyses, such as structure-dependent error bounds on IPF's parameter estimates. When IPF fails to c
    

