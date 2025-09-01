# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semisupervised score based matching algorithm to evaluate the effect of public health interventions](https://arxiv.org/abs/2403.12367) | 提出了一种基于二次评分函数的一对一匹配算法，通过设计权重最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异 |

# 详细

[^1]: 半监督计分匹配算法评估公共卫生干预效果

    Semisupervised score based matching algorithm to evaluate the effect of public health interventions

    [https://arxiv.org/abs/2403.12367](https://arxiv.org/abs/2403.12367)

    提出了一种基于二次评分函数的一对一匹配算法，通过设计权重最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异

    

    多元匹配算法在观察性研究中“配对”相似的研究单元，以消除由于缺乏随机性而引起的潜在偏倚和混杂效应。我们提出了一种基于二次评分函数的新型一对一匹配算法，权重$\beta$被设计为最小化配对训练单元之间的得分差异，同时最大化未配对训练单元之间的得分差异。

    arXiv:2403.12367v1 Announce Type: cross  Abstract: Multivariate matching algorithms "pair" similar study units in an observational study to remove potential bias and confounding effects caused by the absence of randomizations. In one-to-one multivariate matching algorithms, a large number of "pairs" to be matched could mean both the information from a large sample and a large number of tasks, and therefore, to best match the pairs, such a matching algorithm with efficiency and comparatively limited auxiliary matching knowledge provided through a "training" set of paired units by domain experts, is practically intriguing.   We proposed a novel one-to-one matching algorithm based on a quadratic score function $S_{\beta}(x_i,x_j)= \beta^T (x_i-x_j)(x_i-x_j)^T \beta$. The weights $\beta$, which can be interpreted as a variable importance measure, are designed to minimize the score difference between paired training units while maximizing the score difference between unpaired training units
    

