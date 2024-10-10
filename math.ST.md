# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An analysis of the noise schedule for score-based generative models](https://arxiv.org/abs/2402.04650) | 本研究针对基于得分的生成模型噪声调度进行了分析，提出了目标分布和估计分布之间KL散度的上界以及Wasserstein距离的改进误差界限，同时提出了自动调节噪声调度的算法，并通过实验证明了算法的性能。 |
| [^2] | [The extended Ville's inequality for nonintegrable nonnegative supermartingales.](http://arxiv.org/abs/2304.01163) | 本文提出了一种新的理论来描述非负超马氏过程，并推导出一个新的极大不等式，适用于非可积情况，并说明了混合方法的扩展以及该理论在顺序统计中的应用。 |

# 详细

[^1]: 基于得分的生成模型噪声调度分析

    An analysis of the noise schedule for score-based generative models

    [https://arxiv.org/abs/2402.04650](https://arxiv.org/abs/2402.04650)

    本研究针对基于得分的生成模型噪声调度进行了分析，提出了目标分布和估计分布之间KL散度的上界以及Wasserstein距离的改进误差界限，同时提出了自动调节噪声调度的算法，并通过实验证明了算法的性能。

    

    基于得分的生成模型（SGMs）旨在通过仅使用目标数据的噪声扰动样本来学习得分函数，从而估计目标数据分布。最近的文献主要关注评估目标分布和估计分布之间的误差，通过KL散度和Wasserstein距离来衡量生成质量。至今为止，所有现有结果都是针对时间均匀变化的噪声调度得到的。在对数据分布进行温和假设的前提下，我们建立了目标分布和估计分布之间KL散度的上界，明确依赖于任何时间相关的噪声调度。假设得分是利普希茨连续的情况下，我们提供了更好的Wasserstein距离误差界限，利用了有利的收缩机制。我们还提出了一种使用所提出的上界自动调节噪声调度的算法。我们通过实验证明了算法的性能。

    Score-based generative models (SGMs) aim at estimating a target data distribution by learning score functions using only noise-perturbed samples from the target. Recent literature has focused extensively on assessing the error between the target and estimated distributions, gauging the generative quality through the Kullback-Leibler (KL) divergence and Wasserstein distances.  All existing results  have been obtained so far for time-homogeneous speed of the noise schedule.  Under mild assumptions on the data distribution, we establish an upper bound for the KL divergence between the target and the estimated distributions, explicitly depending on any time-dependent noise schedule. Assuming that the score is Lipschitz continuous, we provide an improved error bound in Wasserstein distance, taking advantage of favourable underlying contraction mechanisms. We also propose an algorithm to automatically tune the noise schedule using the proposed upper bound. We illustrate empirically the perfo
    
[^2]: 非可积非负超马氏过程的扩展维尔不等式

    The extended Ville's inequality for nonintegrable nonnegative supermartingales. (arXiv:2304.01163v1 [math.PR])

    [http://arxiv.org/abs/2304.01163](http://arxiv.org/abs/2304.01163)

    本文提出了一种新的理论来描述非负超马氏过程，并推导出一个新的极大不等式，适用于非可积情况，并说明了混合方法的扩展以及该理论在顺序统计中的应用。

    

    本文在 Robbins 的初始工作基础上，严密地提出了一种非负超马氏过程的扩展理论，不需要可积性或有限性。特别地，我们推导了 Robbins 预示的一个关键极大不等式，称为扩展维尔不等式，它加强了经典的维尔不等式（适用于可积非负超马氏过程），并适用于我们的非可积设置。我们推导了混合方法的扩展，适用于我们扩展的非负超马氏过程的 $\sigma$- 有限混合。我们介绍了我们理论在顺序统计中的一些应用，如在推导非参数置信序列和（扩展）e-过程中使用不适当混合（先验）。

    Following initial work by Robbins, we rigorously present an extended theory of nonnegative supermartingales, requiring neither integrability nor finiteness. In particular, we derive a key maximal inequality foreshadowed by Robbins, which we call the extended Ville's inequality, that strengthens the classical Ville's inequality (for integrable nonnegative supermartingales), and also applies to our nonintegrable setting. We derive an extension of the method of mixtures, which applies to $\sigma$-finite mixtures of our extended nonnegative supermartingales. We present some implications of our theory for sequential statistics, such as the use of improper mixtures (priors) in deriving nonparametric confidence sequences and (extended) e-processes.
    

