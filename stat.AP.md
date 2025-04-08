# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions.](http://arxiv.org/abs/2108.11328) | 本文提出了一种可解释的非参数加性模型，使用少量主要和成对交互效应预测调查反应率。该模型可以生成易于可视化和解释的预测面，并取得了 ROAM 数据集上的最先进性能，可以提供改进美国人口普查局和其他调查的反应率议论。 |

# 详细

[^1]: 用简洁可解释的加性模型和结构交互预测人口普查调查反应率

    Predicting Census Survey Response Rates With Parsimonious Additive Models and Structured Interactions. (arXiv:2108.11328v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2108.11328](http://arxiv.org/abs/2108.11328)

    本文提出了一种可解释的非参数加性模型，使用少量主要和成对交互效应预测调查反应率。该模型可以生成易于可视化和解释的预测面，并取得了 ROAM 数据集上的最先进性能，可以提供改进美国人口普查局和其他调查的反应率议论。

    

    本文考虑使用一系列灵活且可解释的非参数模型预测调查反应率。本研究受到美国人口普查局著名的 ROAM 应用的启发，该应用使用在美国人口普查规划数据库数据上训练的线性回归模型来识别难以调查的区域。十年前组织的一场众包竞赛表明，基于回归树集成的机器学习方法在预测调查反应率方面表现最佳；然而，由于它们的黑盒特性，相应的模型不能用于拟定的应用。我们考虑使用 $\ell_0$-based 惩罚的非参数加性模型，它具有少数主要和成对交互效应。从方法论的角度来看，我们研究了我们估计器的计算和统计方面，并讨论了将强层次交互合并的变体。我们的算法（在Github 上开源）允许我们生成易于可视化和解释的预测面，从而获得有关调查反应率的可行见解。我们提出的模型在 ROAM 数据集上实现了最先进的性能，并可以提供有关美国人口普查局和其他调查的改进调查反应率的见解。

    In this paper we consider the problem of predicting survey response rates using a family of flexible and interpretable nonparametric models. The study is motivated by the US Census Bureau's well-known ROAM application which uses a linear regression model trained on the US Census Planning Database data to identify hard-to-survey areas. A crowdsourcing competition organized around ten years ago revealed that machine learning methods based on ensembles of regression trees led to the best performance in predicting survey response rates; however, the corresponding models could not be adopted for the intended application due to their black-box nature. We consider nonparametric additive models with small number of main and pairwise interaction effects using $\ell_0$-based penalization. From a methodological viewpoint, we study both computational and statistical aspects of our estimator; and discuss variants that incorporate strong hierarchical interactions. Our algorithms (opensourced on gith
    

