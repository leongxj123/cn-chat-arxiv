# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fisher-Rao Gradient Flows of Linear Programs and State-Action Natural Policy Gradients](https://arxiv.org/abs/2403.19448) | 研究了基于Fisher信息矩阵的自然梯度方法在线性规划中的应用，展示了线性收敛性，提出了改进现有结果的熵正则化误差估计，并对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性进行了研究。 |
| [^2] | [Label Differential Privacy via Aggregation.](http://arxiv.org/abs/2310.10092) | 以前研究表明朴素的LBA和LLP不能提供标签差分隐私。但本研究显示，使用具有随机抽样的加权LBA可以提供标签差分隐私。 |

# 详细

[^1]: Fisher-Rao线性规划和状态-动作自然策略梯度的梯度流

    Fisher-Rao Gradient Flows of Linear Programs and State-Action Natural Policy Gradients

    [https://arxiv.org/abs/2403.19448](https://arxiv.org/abs/2403.19448)

    研究了基于Fisher信息矩阵的自然梯度方法在线性规划中的应用，展示了线性收敛性，提出了改进现有结果的熵正则化误差估计，并对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性进行了研究。

    

    Kakade的自然策略梯度方法近年来得到广泛研究，表明在有或无正则化的情况下具有线性收敛性。我们研究了另一种基于状态-动作分布的Fisher信息矩阵的自然梯度方法，但在理论方面接受度较低。在这里，状态-动作分布在状态-动作多面体内遵循Fisher-Rao梯度流，相对于线性势。因此，我们更全面地研究线性规划的Fisher-Rao梯度流，并显示了线性收敛性，其速率取决于线性规划的几何特性。换句话说，这提供了线性规划的熵正则化引起的误差估计，这改进了现有结果。我们拓展了这些结果，并展示了对扰动的Fisher-Rao梯度流和自然梯度流的次线性收敛性，直到逼近误差。

    arXiv:2403.19448v1 Announce Type: cross  Abstract: Kakade's natural policy gradient method has been studied extensively in the last years showing linear convergence with and without regularization. We study another natural gradient method which is based on the Fisher information matrix of the state-action distributions and has received little attention from the theoretical side. Here, the state-action distributions follow the Fisher-Rao gradient flow inside the state-action polytope with respect to a linear potential. Therefore, we study Fisher-Rao gradient flows of linear programs more generally and show linear convergence with a rate that depends on the geometry of the linear program. Equivalently, this yields an estimate on the error induced by entropic regularization of the linear program which improves existing results. We extend these results and show sublinear convergence for perturbed Fisher-Rao gradient flows and natural gradient flows up to an approximation error. In particul
    
[^2]: 通过聚合实现标签差分隐私

    Label Differential Privacy via Aggregation. (arXiv:2310.10092v1 [cs.LG])

    [http://arxiv.org/abs/2310.10092](http://arxiv.org/abs/2310.10092)

    以前研究表明朴素的LBA和LLP不能提供标签差分隐私。但本研究显示，使用具有随机抽样的加权LBA可以提供标签差分隐私。

    

    在许多现实应用中，特别是由于隐私领域的最新发展，训练数据可以进行聚合，以保护敏感训练标签的隐私。在标签比例学习(LLP)框架中，数据集被划分为特征向量的包，只能获得每个包中标签的总和。进一步限制的限制学习(LBA)是只能获得包的特征向量的总和（可能是加权的）。我们研究这种聚合技术是否能够在标签差分隐私(label-DP)的概念下提供隐私保证，该概念之前在[Chaudhuri-Hsu'11, Ghazi et al.'21, Esfandiari et al.'22]中进行了研究。很容易看出，朴素的LBA和LLP不能提供标签差分隐私。然而，我们的主要结果表明，使用具有$m$个随机抽样的不相交$k$-大小包的加权LBA实际上是$(\varepsilon,

    In many real-world applications, in particular due to recent developments in the privacy landscape, training data may be aggregated to preserve the privacy of sensitive training labels. In the learning from label proportions (LLP) framework, the dataset is partitioned into bags of feature-vectors which are available only with the sum of the labels per bag. A further restriction, which we call learning from bag aggregates (LBA) is where instead of individual feature-vectors, only the (possibly weighted) sum of the feature-vectors per bag is available. We study whether such aggregation techniques can provide privacy guarantees under the notion of label differential privacy (label-DP) previously studied in for e.g. [Chaudhuri-Hsu'11, Ghazi et al.'21, Esfandiari et al.'22].  It is easily seen that naive LBA and LLP do not provide label-DP. Our main result however, shows that weighted LBA using iid Gaussian weights with $m$ randomly sampled disjoint $k$-sized bags is in fact $(\varepsilon, 
    

