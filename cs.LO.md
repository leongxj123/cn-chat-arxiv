# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Distributional Uncertainty of the SHAP score in Explainable Machine Learning.](http://arxiv.org/abs/2401.12731) | 本研究提出了一个原则性框架，用于处理在未知实体群体分布下的SHAP评分问题。通过考虑一个不确定性区域，我们可以确定所有特征的SHAP评分的紧束范围。 |

# 详细

[^1]: SHAP评分在可解释机器学习中的分布不确定性

    The Distributional Uncertainty of the SHAP score in Explainable Machine Learning. (arXiv:2401.12731v1 [cs.AI])

    [http://arxiv.org/abs/2401.12731](http://arxiv.org/abs/2401.12731)

    本研究提出了一个原则性框架，用于处理在未知实体群体分布下的SHAP评分问题。通过考虑一个不确定性区域，我们可以确定所有特征的SHAP评分的紧束范围。

    

    归属分数反映了输入实体中的特征值对机器学习模型输出的重要性。其中最受欢迎的评分之一是SHAP评分，它是合作博弈理论中Shapley值的具体实例。该评分的定义依赖于实体群体的概率分布。由于通常不知道精确的分布，因此需要主观地进行分配或从数据中进行估计，这可能会导致误导性的特征评分。在本文中，我们提出了一个基于不知道实体群体分布的SHAP评分推理的原则性框架。在我们的框架中，我们考虑一个包含潜在分布的不确定性区域，而特征的SHAP评分成为在该区域上定义的一个函数。我们研究了找到该函数的最大值和最小值的基本问题，这使我们能够确定所有特征的SHAP评分的紧束范围。

    Attribution scores reflect how important the feature values in an input entity are for the output of a machine learning model. One of the most popular attribution scores is the SHAP score, which is an instantiation of the general Shapley value used in coalition game theory. The definition of this score relies on a probability distribution on the entity population. Since the exact distribution is generally unknown, it needs to be assigned subjectively or be estimated from data, which may lead to misleading feature scores. In this paper, we propose a principled framework for reasoning on SHAP scores under unknown entity population distributions. In our framework, we consider an uncertainty region that contains the potential distributions, and the SHAP score of a feature becomes a function defined over this region. We study the basic problems of finding maxima and minima of this function, which allows us to determine tight ranges for the SHAP scores of all features. In particular, we pinp
    

