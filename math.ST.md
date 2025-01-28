# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An analysis of the noise schedule for score-based generative models](https://arxiv.org/abs/2402.04650) | 本研究针对基于得分的生成模型噪声调度进行了分析，提出了目标分布和估计分布之间KL散度的上界以及Wasserstein距离的改进误差界限，同时提出了自动调节噪声调度的算法，并通过实验证明了算法的性能。 |
| [^2] | [Improved learning theory for kernel distribution regression with two-stage sampling.](http://arxiv.org/abs/2308.14335) | 本文改进了核分布回归的学习理论，引入了新的近无偏条件，并提供了关于两阶段采样效果的新误差界。 |

# 详细

[^1]: 基于得分的生成模型噪声调度分析

    An analysis of the noise schedule for score-based generative models

    [https://arxiv.org/abs/2402.04650](https://arxiv.org/abs/2402.04650)

    本研究针对基于得分的生成模型噪声调度进行了分析，提出了目标分布和估计分布之间KL散度的上界以及Wasserstein距离的改进误差界限，同时提出了自动调节噪声调度的算法，并通过实验证明了算法的性能。

    

    基于得分的生成模型（SGMs）旨在通过仅使用目标数据的噪声扰动样本来学习得分函数，从而估计目标数据分布。最近的文献主要关注评估目标分布和估计分布之间的误差，通过KL散度和Wasserstein距离来衡量生成质量。至今为止，所有现有结果都是针对时间均匀变化的噪声调度得到的。在对数据分布进行温和假设的前提下，我们建立了目标分布和估计分布之间KL散度的上界，明确依赖于任何时间相关的噪声调度。假设得分是利普希茨连续的情况下，我们提供了更好的Wasserstein距离误差界限，利用了有利的收缩机制。我们还提出了一种使用所提出的上界自动调节噪声调度的算法。我们通过实验证明了算法的性能。

    Score-based generative models (SGMs) aim at estimating a target data distribution by learning score functions using only noise-perturbed samples from the target. Recent literature has focused extensively on assessing the error between the target and estimated distributions, gauging the generative quality through the Kullback-Leibler (KL) divergence and Wasserstein distances.  All existing results  have been obtained so far for time-homogeneous speed of the noise schedule.  Under mild assumptions on the data distribution, we establish an upper bound for the KL divergence between the target and the estimated distributions, explicitly depending on any time-dependent noise schedule. Assuming that the score is Lipschitz continuous, we provide an improved error bound in Wasserstein distance, taking advantage of favourable underlying contraction mechanisms. We also propose an algorithm to automatically tune the noise schedule using the proposed upper bound. We illustrate empirically the perfo
    
[^2]: 改进的核分布回归学习理论与两阶段采样

    Improved learning theory for kernel distribution regression with two-stage sampling. (arXiv:2308.14335v1 [math.ST])

    [http://arxiv.org/abs/2308.14335](http://arxiv.org/abs/2308.14335)

    本文改进了核分布回归的学习理论，引入了新的近无偏条件，并提供了关于两阶段采样效果的新误差界。

    

    分布回归问题涵盖了许多重要的统计和机器学习任务，在各种应用中都有出现。在解决这个问题的各种现有方法中，核方法已经成为首选的方法。事实上，核分布回归在计算上是有利的，并且得到了最近的学习理论的支持。该理论还解决了两阶段采样的设置，其中只有输入分布的样本可用。在本文中，我们改进了核分布回归的学习理论。我们研究了基于希尔伯特嵌入的核，这些核包含了大多数（如果不是全部）现有方法。我们引入了嵌入的新近无偏条件，使我们能够通过新的分析提供关于两阶段采样效果的新误差界。我们证明了这种新近无偏条件对三个重要的核类别成立，这些核基于最优输运和平均嵌入。

    The distribution regression problem encompasses many important statistics and machine learning tasks, and arises in a large range of applications. Among various existing approaches to tackle this problem, kernel methods have become a method of choice. Indeed, kernel distribution regression is both computationally favorable, and supported by a recent learning theory. This theory also tackles the two-stage sampling setting, where only samples from the input distributions are available. In this paper, we improve the learning theory of kernel distribution regression. We address kernels based on Hilbertian embeddings, that encompass most, if not all, of the existing approaches. We introduce the novel near-unbiased condition on the Hilbertian embeddings, that enables us to provide new error bounds on the effect of the two-stage sampling, thanks to a new analysis. We show that this near-unbiased condition holds for three important classes of kernels, based on optimal transport and mean embedd
    

