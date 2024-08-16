# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Learning of Item Response Theory Models](https://arxiv.org/abs/2403.00680) | 该论文提出了一种从大数据中学习项目反应理论模型中的潜在变量的方法，利用这些模型与逻辑回归之间的相似性来提高计算的效率和可伸缩性。 |
| [^2] | [Optimal Scalarizations for Sublinear Hypervolume Regret.](http://arxiv.org/abs/2307.03288) | 研究了用于亚线性超体积遗憾度量的最优标量化方法，证明了具有均匀随机权重的超体积标量化方法在最小化超体积遗憾方面是最优的，并在多目标随机线性赌博机问题上进行了案例研究。 |

# 详细

[^1]: 可扩展的项目反应理论模型学习

    Scalable Learning of Item Response Theory Models

    [https://arxiv.org/abs/2403.00680](https://arxiv.org/abs/2403.00680)

    该论文提出了一种从大数据中学习项目反应理论模型中的潜在变量的方法，利用这些模型与逻辑回归之间的相似性来提高计算的效率和可伸缩性。

    

    项目反应理论（IRT）模型旨在评估 $n$ 名考生的潜在能力以及 $m$ 个测验项目的隐含难度特征，这些项目是从表明其对应答案质量的分类数据中得出的。传统的心理测量评估基于相对较少的考生和项目，例如一个由 $200$ 名学生解决包含 $10$ 道题目的考试的班级。而近年来的全球大规模评估，如PISA，或互联网研究，可能导致参与者数量显著增加。此外，在机器学习领域，算法扮演考生角色，数据分析问题扮演项目角色，$n$ 和 $m$ 都可能变得非常大，挑战计算的效率和可伸缩性。为了从大数据中学习IRT模型中的潜在变量，我们利用这些模型与逻辑回归之间的相似性，后者可以使用s准确地近似。

    arXiv:2403.00680v1 Announce Type: new  Abstract: Item Response Theory (IRT) models aim to assess latent abilities of $n$ examinees along with latent difficulty characteristics of $m$ test items from categorical data that indicates the quality of their corresponding answers. Classical psychometric assessments are based on a relatively small number of examinees and items, say a class of $200$ students solving an exam comprising $10$ problems. More recent global large scale assessments such as PISA, or internet studies, may lead to significantly increased numbers of participants. Additionally, in the context of Machine Learning where algorithms take the role of examinees and data analysis problems take the role of items, both $n$ and $m$ may become very large, challenging the efficiency and scalability of computations. To learn the latent variables in IRT models from large data, we leverage the similarity of these models to logistic regression, which can be approximated accurately using s
    
[^2]: 用于亚线性超体积遗憾度量的最优标量化方法

    Optimal Scalarizations for Sublinear Hypervolume Regret. (arXiv:2307.03288v1 [cs.LG])

    [http://arxiv.org/abs/2307.03288](http://arxiv.org/abs/2307.03288)

    研究了用于亚线性超体积遗憾度量的最优标量化方法，证明了具有均匀随机权重的超体积标量化方法在最小化超体积遗憾方面是最优的，并在多目标随机线性赌博机问题上进行了案例研究。

    

    标量化是一种通用的技术，可以应用于任何多目标设置中，将多个目标减少为一个，例如最近在RLHF中用于训练校准人类偏好的奖励模型。然而，一些人对这种经典方法持否定态度，因为已知线性标量化会忽略帕累托前沿的凹区域。为此，我们旨在找到简单的非线性标量化方法，以通过被支配的超体积来探索帕累托前沿上的多样化目标集。我们证明，具有均匀随机权重的超体积标量化令人惊讶地是为了证明最小化超体积遗憾而最优的，实现了 $O(T^{-1/k})$ 的最优亚线性遗憾界，同时匹配的下界表明在渐近情况下没有任何算法能做得更好。作为一个理论案例研究，我们考虑了多目标随机线性赌博机问题，并展示了通过利用超线性遗憾界的超体积标量化方法，

    Scalarization is a general technique that can be deployed in any multiobjective setting to reduce multiple objectives into one, such as recently in RLHF for training reward models that align human preferences. Yet some have dismissed this classical approach because linear scalarizations are known to miss concave regions of the Pareto frontier. To that end, we aim to find simple non-linear scalarizations that can explore a diverse set of $k$ objectives on the Pareto frontier, as measured by the dominated hypervolume. We show that hypervolume scalarizations with uniformly random weights are surprisingly optimal for provably minimizing the hypervolume regret, achieving an optimal sublinear regret bound of $O(T^{-1/k})$, with matching lower bounds that preclude any algorithm from doing better asymptotically. As a theoretical case study, we consider the multiobjective stochastic linear bandits problem and demonstrate that by exploiting the sublinear regret bounds of the hypervolume scalariz
    

