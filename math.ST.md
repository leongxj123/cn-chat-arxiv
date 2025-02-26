# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric logistic regression with deep learning.](http://arxiv.org/abs/2401.12482) | 本文提出了一种简单的方法来分析非参数 logistic 回归问题，通过在温和的假设下，在 Hellinger 距离下推导出了最大似然估计器的收敛速率。 |

# 详细

[^1]: 非参数 logistic 回归与深度学习

    Nonparametric logistic regression with deep learning. (arXiv:2401.12482v1 [math.ST])

    [http://arxiv.org/abs/2401.12482](http://arxiv.org/abs/2401.12482)

    本文提出了一种简单的方法来分析非参数 logistic 回归问题，通过在温和的假设下，在 Hellinger 距离下推导出了最大似然估计器的收敛速率。

    

    考虑非参数 logistic 回归问题。在 logistic 回归中，我们通常考虑最大似然估计器，而过度风险是真实条件类概率和估计条件类概率之间 Kullback-Leibler (KL) 散度的期望。然而，在非参数 logistic 回归中，KL 散度很容易发散，因此，过度风险的收敛很难证明或不成立。若干现有研究表明，在强假设下 KL 散度的收敛性。在大多数情况下，我们的目标是估计真实的条件类概率。因此，不需要分析过度风险本身，只需在某些合适的度量下证明最大似然估计器的一致性即可。在本文中，我们使用简单统一的方法分析非参数最大似然估计器 (NPMLE)，直接推导出 NPMLE 在 Hellinger 距离下的收敛速率，在温和的假设下成立。

    Consider the nonparametric logistic regression problem. In the logistic regression, we usually consider the maximum likelihood estimator, and the excess risk is the expectation of the Kullback-Leibler (KL) divergence between the true and estimated conditional class probabilities. However, in the nonparametric logistic regression, the KL divergence could diverge easily, and thus, the convergence of the excess risk is difficult to prove or does not hold. Several existing studies show the convergence of the KL divergence under strong assumptions. In most cases, our goal is to estimate the true conditional class probabilities. Thus, instead of analyzing the excess risk itself, it suffices to show the consistency of the maximum likelihood estimator in some suitable metric. In this paper, using a simple unified approach for analyzing the nonparametric maximum likelihood estimator (NPMLE), we directly derive the convergence rates of the NPMLE in the Hellinger distance under mild assumptions. 
    

