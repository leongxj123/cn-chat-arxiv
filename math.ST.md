# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interference Among First-Price Pacing Equilibria: A Bias and Variance Analysis](https://arxiv.org/abs/2402.07322) | 本文提出了一种并行的预算控制的A/B测试设计，通过市场细分的方式在更大的市场中识别子市场，并在每个子市场上进行并行实验。 |
| [^2] | [Kernel Learning in Ridge Regression "Automatically" Yields Exact Low Rank Solution.](http://arxiv.org/abs/2310.11736) | 该论文研究了核岭回归问题中核学习的低秩解的性质。在只有低维子空间对响应变量有解释能力的情况下，通过该方法可以自动得到精确的低秩解，无需额外的正则化。 |
| [^3] | [Dynamic treatment effects: high-dimensional inference under model misspecification.](http://arxiv.org/abs/2111.06818) | 本文提出了一种新的鲁棒估计方法来解决动态治疗效应估计中的挑战，提高了在模型错误下的高维环境中的估计鲁棒性和可靠性。 |

# 详细

[^1]: 第一价拍卖均衡中的干扰：偏差和方差分析

    Interference Among First-Price Pacing Equilibria: A Bias and Variance Analysis

    [https://arxiv.org/abs/2402.07322](https://arxiv.org/abs/2402.07322)

    本文提出了一种并行的预算控制的A/B测试设计，通过市场细分的方式在更大的市场中识别子市场，并在每个子市场上进行并行实验。

    

    在互联网行业中，在线A/B测试被广泛用于决策新功能的推出。然而对于在线市场（如广告市场），标准的A/B测试方法可能导致结果出现偏差，因为买家在预算约束下运作，试验组的预算消耗会影响对照组的表现。为了解决这种干扰，可以采用“预算分割设计”，即每个实验组都有一个独立的预算约束，并且每个实验组接收相等的预算份额，从而实现“预算控制的A/B测试”。尽管预算控制的A/B测试有明显的优势，但当预算分割得太小时，性能会下降，限制了这种系统的总吞吐量。本文提出了一种并行的预算控制的A/B测试设计，通过市场细分的方式在更大的市场中识别子市场，并在每个子市场上进行并行实验。我们的贡献如下：首先，引入了一种新的方法来分析第一价拍卖的均衡状况，揭示了其中的偏差和方差。

    Online A/B testing is widely used in the internet industry to inform decisions on new feature roll-outs. For online marketplaces (such as advertising markets), standard approaches to A/B testing may lead to biased results when buyers operate under a budget constraint, as budget consumption in one arm of the experiment impacts performance of the other arm. To counteract this interference, one can use a budget-split design where the budget constraint operates on a per-arm basis and each arm receives an equal fraction of the budget, leading to ``budget-controlled A/B testing.'' Despite clear advantages of budget-controlled A/B testing, performance degrades when budget are split too small, limiting the overall throughput of such systems. In this paper, we propose a parallel budget-controlled A/B testing design where we use market segmentation to identify submarkets in the larger market, and we run parallel experiments on each submarket.   Our contributions are as follows: First, we introdu
    
[^2]: 在Ridge回归中，核学习“自动”给出精确的低秩解

    Kernel Learning in Ridge Regression "Automatically" Yields Exact Low Rank Solution. (arXiv:2310.11736v1 [math.ST])

    [http://arxiv.org/abs/2310.11736](http://arxiv.org/abs/2310.11736)

    该论文研究了核岭回归问题中核学习的低秩解的性质。在只有低维子空间对响应变量有解释能力的情况下，通过该方法可以自动得到精确的低秩解，无需额外的正则化。

    

    我们考虑形式为$(x,x') \mapsto \phi(\|x-x'\|^2_\Sigma)$且由参数$\Sigma$参数化的核函数。对于这样的核函数，我们研究了核岭回归问题的变体，它同时优化了预测函数和再现核希尔伯特空间的参数$\Sigma$。从这个核岭回归问题中学到的$\Sigma$的特征空间可以告诉我们协变量空间中哪些方向对预测是重要的。假设协变量只通过低维子空间（中心均值子空间）对响应变量有非零的解释能力，我们发现有很高的概率下有限样本核学习目标的全局最小化者也是低秩的。更具体地说，最小化$\Sigma$的秩有很高的概率被中心均值子空间的维度所限制。这个现象很有趣，因为低秩特性是在没有使用任何对$\Sigma$的显式正则化的情况下实现的，例如核范数正则化等。

    We consider kernels of the form $(x,x') \mapsto \phi(\|x-x'\|^2_\Sigma)$ parametrized by $\Sigma$. For such kernels, we study a variant of the kernel ridge regression problem which simultaneously optimizes the prediction function and the parameter $\Sigma$ of the reproducing kernel Hilbert space. The eigenspace of the $\Sigma$ learned from this kernel ridge regression problem can inform us which directions in covariate space are important for prediction.  Assuming that the covariates have nonzero explanatory power for the response only through a low dimensional subspace (central mean subspace), we find that the global minimizer of the finite sample kernel learning objective is also low rank with high probability. More precisely, the rank of the minimizing $\Sigma$ is with high probability bounded by the dimension of the central mean subspace. This phenomenon is interesting because the low rankness property is achieved without using any explicit regularization of $\Sigma$, e.g., nuclear
    
[^3]: 动态治疗效应：模型错误下的高维推断

    Dynamic treatment effects: high-dimensional inference under model misspecification. (arXiv:2111.06818v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2111.06818](http://arxiv.org/abs/2111.06818)

    本文提出了一种新的鲁棒估计方法来解决动态治疗效应估计中的挑战，提高了在模型错误下的高维环境中的估计鲁棒性和可靠性。

    

    估计动态治疗效应在各个学科中都是至关重要的，可以提供有关干预的时变因果影响的微妙见解。然而，由于“维数灾难”和时变混杂的存在，这种估计存在着挑战，可能导致估计偏误。此外，正确地规定日益增多的治疗分配和多重暴露的结果模型似乎过于复杂。鉴于这些挑战，双重鲁棒性的概念，在允许模型错误的情况下，是非常有价值的，然而在实际应用中并没有实现。本文通过提出新的鲁棒估计方法来解决这个问题，同时对治疗分配和结果模型进行鲁棒估计。我们提出了一种“序列模型双重鲁棒性”的解决方案，证明了当每个时间暴露都是双重鲁棒性的时，可以在多个时间点上实现双重鲁棒性。这种方法提高了高维环境下动态治疗效应估计的鲁棒性和可靠性。

    Estimating dynamic treatment effects is essential across various disciplines, offering nuanced insights into the time-dependent causal impact of interventions. However, this estimation presents challenges due to the "curse of dimensionality" and time-varying confounding, which can lead to biased estimates. Additionally, correctly specifying the growing number of treatment assignments and outcome models with multiple exposures seems overly complex. Given these challenges, the concept of double robustness, where model misspecification is permitted, is extremely valuable, yet unachieved in practical applications. This paper introduces a new approach by proposing novel, robust estimators for both treatment assignments and outcome models. We present a "sequential model double robust" solution, demonstrating that double robustness over multiple time points can be achieved when each time exposure is doubly robust. This approach improves the robustness and reliability of dynamic treatment effe
    

