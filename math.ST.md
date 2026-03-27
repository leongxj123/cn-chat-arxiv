# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improve Generalization Ability of Deep Wide Residual Network with A Suitable Scaling Factor](https://arxiv.org/abs/2403.04545) | 通过在深广残差网络中使用适当的缩放因子，可以提高泛化能力，即使允许缩放因子随深度减小，也可以实现最小最大速率。 |
| [^2] | [Maximal Inequalities for Empirical Processes under General Mixing Conditions with an Application to Strong Approximations](https://arxiv.org/abs/2402.11394) | 本文提出了针对一般混合随机过程的样本均值上确界的界限，不受混合速度的影响，强调了集中速率和复杂度度量的重要性，并发现了混合速度对集中速率的影响，引入了相变的概念。 |

# 详细

[^1]: 通过合适的缩放因子提高深广残差网络的泛化能力

    Improve Generalization Ability of Deep Wide Residual Network with A Suitable Scaling Factor

    [https://arxiv.org/abs/2403.04545](https://arxiv.org/abs/2403.04545)

    通过在深广残差网络中使用适当的缩放因子，可以提高泛化能力，即使允许缩放因子随深度减小，也可以实现最小最大速率。

    

    深残差神经网络（ResNets）在广泛的实际应用中取得了显著的成功。在本文中，我们确定了深广残差网络中残差分支上的合适缩放因子（用$\alpha$表示），以实现良好的泛化能力。我们展示了如果$\alpha$是一个常数，由残差神经切向核（RNTK）引发的函数类在深度趋近无穷时是渐近不可学习的。我们还强调了一个令人惊讶的现象：即使我们允许$\alpha$随着深度$L$的增加而减小，退化现象仍可能发生。然而，当$\alpha$与$L$快速减小时，使用深层RNTK进行核回归，并且在早停条件下可以实现最小最大速率，前提是目标回归函数落在与无限深度RNTK相关的再生核希尔伯特空间中。我们对合成数据和真实分类任务进行了模拟研究。

    arXiv:2403.04545v1 Announce Type: new  Abstract: Deep Residual Neural Networks (ResNets) have demonstrated remarkable success across a wide range of real-world applications. In this paper, we identify a suitable scaling factor (denoted by $\alpha$) on the residual branch of deep wide ResNets to achieve good generalization ability. We show that if $\alpha$ is a constant, the class of functions induced by Residual Neural Tangent Kernel (RNTK) is asymptotically not learnable, as the depth goes to infinity. We also highlight a surprising phenomenon: even if we allow $\alpha$ to decrease with increasing depth $L$, the degeneration phenomenon may still occur. However, when $\alpha$ decreases rapidly with $L$, the kernel regression with deep RNTK with early stopping can achieve the minimax rate provided that the target regression function falls in the reproducing kernel Hilbert space associated with the infinite-depth RNTK. Our simulation studies on synthetic data and real classification task
    
[^2]: 基于一般混合条件的经验过程的极值不等式及其在强逼近中的应用

    Maximal Inequalities for Empirical Processes under General Mixing Conditions with an Application to Strong Approximations

    [https://arxiv.org/abs/2402.11394](https://arxiv.org/abs/2402.11394)

    本文提出了针对一般混合随机过程的样本均值上确界的界限，不受混合速度的影响，强调了集中速率和复杂度度量的重要性，并发现了混合速度对集中速率的影响，引入了相变的概念。

    

    本文针对具有任意混合率的一般混合随机过程提供了一个样本均值的上确界的界限。无论混合的速度如何，该界限由一个集中速率和一种新颖的复杂度度量组成。然而，混合的速度影响前者的数量，意味着出现了相变。快速混合导致标准的根号n集中速率，而慢速混合导致较慢的集中速率，其速度取决于混合结构。我们的发现应用于推导具有任意混合率的一般混合过程的强逼近结果。

    arXiv:2402.11394v1 Announce Type: cross  Abstract: This paper provides a bound for the supremum of sample averages over a class of functions for a general class of mixing stochastic processes with arbitrary mixing rates. Regardless of the speed of mixing, the bound is comprised of a concentration rate and a novel measure of complexity. The speed of mixing, however, affects the former quantity implying a phase transition. Fast mixing leads to the standard root-n concentration rate, while slow mixing leads to a slower concentration rate, its speed depends on the mixing structure. Our findings are applied to derive strong approximation results for a general class of mixing processes with arbitrary mixing rates.
    

