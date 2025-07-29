# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ShaRP: Explaining Rankings with Shapley Values.](http://arxiv.org/abs/2401.16744) | ShaRP是一个基于Shapley值的框架，用于解释排名结果中各个特征的贡献。即使使用线性评分函数，特征的权重也不一定对应其Shapley值的贡献，而是取决于特征分布和评分特征之间的局部相互作用。 |

# 详细

[^1]: ShaRP：用Shapley值解释排名

    ShaRP: Explaining Rankings with Shapley Values. (arXiv:2401.16744v1 [cs.AI])

    [http://arxiv.org/abs/2401.16744](http://arxiv.org/abs/2401.16744)

    ShaRP是一个基于Shapley值的框架，用于解释排名结果中各个特征的贡献。即使使用线性评分函数，特征的权重也不一定对应其Shapley值的贡献，而是取决于特征分布和评分特征之间的局部相互作用。

    

    在招聘、大学招生和贷款等重要领域的算法决策常常是基于排名的。由于这些决策对个人、组织和人群的影响，有必要了解它们：了解决策是否遵守法律，帮助个人提高他们的排名，并设计更好的排名程序。本文提出了ShaRP（Shapley for Rankings and Preferences），这是一个基于Shapley值的框架，用于解释特征对排名结果不同方面的贡献。使用ShaRP，我们展示了即使算法排名器使用的评分函数是已知的且是线性的，每个特征的权重也不一定对应其Shapley值的贡献。贡献取决于特征的分布以及评分特征之间微妙的局部相互作用。ShaRP基于量化输入影响框架，并可以计算贡献。

    Algorithmic decisions in critical domains such as hiring, college admissions, and lending are often based on rankings. Because of the impact these decisions have on individuals, organizations, and population groups, there is a need to understand them: to know whether the decisions are abiding by the law, to help individuals improve their rankings, and to design better ranking procedures.  In this paper, we present ShaRP (Shapley for Rankings and Preferences), a framework that explains the contributions of features to different aspects of a ranked outcome, and is based on Shapley values. Using ShaRP, we show that even when the scoring function used by an algorithmic ranker is known and linear, the weight of each feature does not correspond to its Shapley value contribution. The contributions instead depend on the feature distributions, and on the subtle local interactions between the scoring features. ShaRP builds on the Quantitative Input Influence framework, and can compute the contri
    

