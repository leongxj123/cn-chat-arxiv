# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Optimization with Hidden Constraints via Latent Decision Models.](http://arxiv.org/abs/2310.18449) | 本文介绍了一种基于潜在决策模型的贝叶斯优化方法，通过利用变分自编码器学习可行决策的分布，在原始空间和潜在空间之间实现了双向映射，从而解决了公共决策制定中的隐藏约束问题。 |

# 详细

[^1]: 基于潜在决策模型的具有隐藏约束的贝叶斯优化方法

    Bayesian Optimization with Hidden Constraints via Latent Decision Models. (arXiv:2310.18449v1 [stat.ML])

    [http://arxiv.org/abs/2310.18449](http://arxiv.org/abs/2310.18449)

    本文介绍了一种基于潜在决策模型的贝叶斯优化方法，通过利用变分自编码器学习可行决策的分布，在原始空间和潜在空间之间实现了双向映射，从而解决了公共决策制定中的隐藏约束问题。

    

    贝叶斯优化（BO）已经成为解决复杂决策问题的强大工具，尤其在公共政策领域如警察划区方面。然而，由于定义可行区域的复杂性和决策的高维度，其在公共决策制定中的广泛应用受到了阻碍。本文介绍了一种新的贝叶斯优化方法——隐藏约束潜在空间贝叶斯优化（HC-LSBO），该方法集成了潜在决策模型。该方法利用变分自编码器来学习可行决策的分布，实现了原始决策空间与较低维度的潜在空间之间的双向映射。通过这种方式，HC-LSBO捕捉了公共决策制定中固有的隐藏约束的细微差别，在潜在空间中进行优化的同时，在原始空间中评估目标。我们通过对合成数据集和真实数据集进行数值实验来验证我们的方法，特别关注大规模问题。

    Bayesian optimization (BO) has emerged as a potent tool for addressing intricate decision-making challenges, especially in public policy domains such as police districting. However, its broader application in public policymaking is hindered by the complexity of defining feasible regions and the high-dimensionality of decisions. This paper introduces the Hidden-Constrained Latent Space Bayesian Optimization (HC-LSBO), a novel BO method integrated with a latent decision model. This approach leverages a variational autoencoder to learn the distribution of feasible decisions, enabling a two-way mapping between the original decision space and a lower-dimensional latent space. By doing so, HC-LSBO captures the nuances of hidden constraints inherent in public policymaking, allowing for optimization in the latent space while evaluating objectives in the original space. We validate our method through numerical experiments on both synthetic and real data sets, with a specific focus on large-scal
    

