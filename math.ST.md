# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Physics-informed machine learning as a kernel method](https://arxiv.org/abs/2402.07514) | 物理约束的机器学习结合了数据方法的表达能力与物理模型的可解释性，可以用于正则化经验风险并提高估计器的统计性能。 |
| [^2] | [Top-$K$ ranking with a monotone adversary](https://arxiv.org/abs/2402.07445) | 本文针对具有单调对手的Top-K排名问题，提出了一种加权最大似然估计器(MLE)，在样本复杂度方面接近最优。算法创新包括了对加权MLE的精确且紧密的$\ell_\infty$误差分析，并与加权比较图的谱特性相关联。 |

# 详细

[^1]: 物理约束的机器学习作为一种核方法

    Physics-informed machine learning as a kernel method

    [https://arxiv.org/abs/2402.07514](https://arxiv.org/abs/2402.07514)

    物理约束的机器学习结合了数据方法的表达能力与物理模型的可解释性，可以用于正则化经验风险并提高估计器的统计性能。

    

    物理约束的机器学习将基于数据的方法的表达能力与物理模型的可解释性相结合。在这种背景下，我们考虑一个普通的回归问题，其中经验风险由一个偏微分方程正则化，该方程量化了物理不一致性。我们证明对于线性微分先验，该问题可以被表述为核回归任务。利用核理论，我们推导出正则化风险的最小化器的收敛速度，并表明它至少以Sobolev最小化速度收敛。然而，根据物理误差的不同，可以实现更快的收敛速度。通过一个一维示例来说明这个原理，支持一个论点：使用物理信息来正则化经验风险对估计器的统计性能有益。

    Physics-informed machine learning combines the expressiveness of data-based approaches with the interpretability of physical models. In this context, we consider a general regression problem where the empirical risk is regularized by a partial differential equation that quantifies the physical inconsistency. We prove that for linear differential priors, the problem can be formulated as a kernel regression task. Taking advantage of kernel theory, we derive convergence rates for the minimizer of the regularized risk and show that it converges at least at the Sobolev minimax rate. However, faster rates can be achieved, depending on the physical error. This principle is illustrated with a one-dimensional example, supporting the claim that regularizing the empirical risk with physical information can be beneficial to the statistical performance of estimators.
    
[^2]: 具有单调对手的Top-K排名问题

    Top-$K$ ranking with a monotone adversary

    [https://arxiv.org/abs/2402.07445](https://arxiv.org/abs/2402.07445)

    本文针对具有单调对手的Top-K排名问题，提出了一种加权最大似然估计器(MLE)，在样本复杂度方面接近最优。算法创新包括了对加权MLE的精确且紧密的$\ell_\infty$误差分析，并与加权比较图的谱特性相关联。

    

    本文解决了具有单调对手的Top-K排名问题。我们考虑了一个比较图被随机生成且对手可以添加任意边的情况。统计学家的目标是根据从这个半随机比较图导出的两两比较准确地识别出Top-K的首选项。本文的主要贡献是开发出一种加权最大似然估计器(MLE)，它在样本复杂度方面达到了近似最优，最多差一个$log^2(n)$的因子，其中n表示比较项的数量。这得益于分析和算法创新的结合。在分析方面，我们提供了一种更明确、更紧密的加权MLE的$\ell_\infty$误差分析，它与加权比较图的谱特性相关。受此启发，我们的算法创新涉及到了

    In this paper, we address the top-$K$ ranking problem with a monotone adversary. We consider the scenario where a comparison graph is randomly generated and the adversary is allowed to add arbitrary edges. The statistician's goal is then to accurately identify the top-$K$ preferred items based on pairwise comparisons derived from this semi-random comparison graph. The main contribution of this paper is to develop a weighted maximum likelihood estimator (MLE) that achieves near-optimal sample complexity, up to a $\log^2(n)$ factor, where n denotes the number of items under comparison. This is made possible through a combination of analytical and algorithmic innovations. On the analytical front, we provide a refined $\ell_\infty$ error analysis of the weighted MLE that is more explicit and tighter than existing analyses. It relates the $\ell_\infty$ error with the spectral properties of the weighted comparison graph. Motivated by this, our algorithmic innovation involves the development 
    

