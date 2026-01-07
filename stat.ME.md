# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Bayesian inference for the generalized linear mixed model](https://arxiv.org/abs/2403.03007) | 该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。 |
| [^2] | [Bayesian score calibration for approximate models.](http://arxiv.org/abs/2211.05357) | 本文提出了一种用于减小偏差和产生更准确不确定性量化的近似后验调整方法，通过优化近似后验的变换来最大化得分规则。这种方法只需要进行少量复杂模型模拟，且具有数值稳定性。 |

# 详细

[^1]: 通用线性混合模型的可扩展贝叶斯推断

    Scalable Bayesian inference for the generalized linear mixed model

    [https://arxiv.org/abs/2403.03007](https://arxiv.org/abs/2403.03007)

    该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。

    

    通用线性混合模型（GLMM）是处理相关数据的一种流行统计方法，在包括生物医学数据等大数据常见的应用领域被广泛使用。本文的重点是针对GLMM的可扩展统计推断，我们将统计推断定义为：（i）对总体参数的估计以及（ii）在存在不确定性的情况下评估科学假设。人工智能（AI）学习算法擅长可扩展的统计估计，但很少包括不确定性量化。相比之下，贝叶斯推断提供完整的统计推断，因为不确定性量化自动来自后验分布。不幸的是，包括马尔可夫链蒙特卡洛（MCMC）在内的贝叶斯推断算法在大数据环境中变得难以计算。在本文中，我们介绍了一个统计推断算法

    arXiv:2403.03007v1 Announce Type: cross  Abstract: The generalized linear mixed model (GLMM) is a popular statistical approach for handling correlated data, and is used extensively in applications areas where big data is common, including biomedical data settings. The focus of this paper is scalable statistical inference for the GLMM, where we define statistical inference as: (i) estimation of population parameters, and (ii) evaluation of scientific hypotheses in the presence of uncertainty. Artificial intelligence (AI) learning algorithms excel at scalable statistical estimation, but rarely include uncertainty quantification. In contrast, Bayesian inference provides full statistical inference, since uncertainty quantification results automatically from the posterior distribution. Unfortunately, Bayesian inference algorithms, including Markov Chain Monte Carlo (MCMC), become computationally intractable in big data settings. In this paper, we introduce a statistical inference algorithm 
    
[^2]: 适用于近似模型的贝叶斯得分校准

    Bayesian score calibration for approximate models. (arXiv:2211.05357v4 [stat.CO] UPDATED)

    [http://arxiv.org/abs/2211.05357](http://arxiv.org/abs/2211.05357)

    本文提出了一种用于减小偏差和产生更准确不确定性量化的近似后验调整方法，通过优化近似后验的变换来最大化得分规则。这种方法只需要进行少量复杂模型模拟，且具有数值稳定性。

    

    科学家们不断发展越来越复杂的机械模型，以更真实地反映他们的知识。使用这些模型进行统计推断可能具有挑战性，因为相应的似然函数通常难以处理，并且模型模拟可能带来计算负担。幸运的是，在许多情况下，可以采用替代模型或近似似然函数。直接使用替代似然函数进行贝叶斯推断可能很方便，但可能导致偏差和不准确的不确定性量化。在本文中，我们提出了一种新的方法，通过优化近似后验的变换来最大化得分规则，从而减小偏差并产生更准确的不确定性量化。我们的方法只需要进行（固定的）少量复杂模型模拟，且具有数值稳定性。我们在几个不断增加的示例上展示了新方法的良好性能。

    Scientists continue to develop increasingly complex mechanistic models to reflect their knowledge more realistically. Statistical inference using these models can be challenging since the corresponding likelihood function is often intractable and model simulation may be computationally burdensome. Fortunately, in many of these situations, it is possible to adopt a surrogate model or approximate likelihood function. It may be convenient to conduct Bayesian inference directly with the surrogate, but this can result in bias and poor uncertainty quantification. In this paper we propose a new method for adjusting approximate posterior samples to reduce bias and produce more accurate uncertainty quantification. We do this by optimizing a transform of the approximate posterior that maximizes a scoring rule. Our approach requires only a (fixed) small number of complex model simulations and is numerically stable. We demonstrate good performance of the new method on several examples of increasin
    

