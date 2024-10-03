# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Budget Aggregation with Single-Peaked Preferences](https://arxiv.org/abs/2402.15904) | 论文研究了具有单峰偏好的最优预算聚合问题，并在多维泛化的星形效用函数类别中探讨不同模型。对于两种备选方案，证明了统一幻影机制是唯一满足比例性的策略证明机制。然后，对于超过两种备选方案的情况，论文表明不存在同时满足效率、策略性和比例性的机制。 |
| [^2] | [Logit-Q Dynamics for Efficient Learning in Stochastic Teams.](http://arxiv.org/abs/2302.09806) | 本文提出了两种Logit-Q学习动力学，通过将经典和独立的对数线性学习更新与在政策上的值迭代更新相结合，实现了在随机博弈中的高效学习。通过对比和量化分析，证明了该动力学在随机团队中可以达到（接近）高效均衡。 |

# 详细

[^1]: 具有单峰偏好的最优预算聚合

    Optimal Budget Aggregation with Single-Peaked Preferences

    [https://arxiv.org/abs/2402.15904](https://arxiv.org/abs/2402.15904)

    论文研究了具有单峰偏好的最优预算聚合问题，并在多维泛化的星形效用函数类别中探讨不同模型。对于两种备选方案，证明了统一幻影机制是唯一满足比例性的策略证明机制。然后，对于超过两种备选方案的情况，论文表明不存在同时满足效率、策略性和比例性的机制。

    

    我们研究了将分布（如预算提案）聚合成集体分布的问题。理想的聚合机制应该是帕累托有效、策略证明和公平的。大多数先前的工作假设代理根据其理想预算与$\ell_1$距离来评估预算。我们研究并比较了来自星形效用函数更大类别的不同模型，这是单峰偏好的多维泛化。对于两种备选方案的情况，我们通过证明在非常一般的假设下，统一幻影机制是唯一满足比例性的策略证明机制，从而扩展了现有结果。对于超过两种备选方案的情况，我们对$\ell_1$和$\ell_\infty$的不满意性建立了全面的不可能性：没有机制能够同时满足效率、策略性和比例性。

    arXiv:2402.15904v1 Announce Type: new  Abstract: We study the problem of aggregating distributions, such as budget proposals, into a collective distribution. An ideal aggregation mechanism would be Pareto efficient, strategyproof, and fair. Most previous work assumes that agents evaluate budgets according to the $\ell_1$ distance to their ideal budget. We investigate and compare different models from the larger class of star-shaped utility functions - a multi-dimensional generalization of single-peaked preferences. For the case of two alternatives, we extend existing results by proving that under very general assumptions, the uniform phantom mechanism is the only strategyproof mechanism that satisfies proportionality - a minimal notion of fairness introduced by Freeman et al. (2021). Moving to the case of more than two alternatives, we establish sweeping impossibilities for $\ell_1$ and $\ell_\infty$ disutilities: no mechanism satisfies efficiency, strategyproofness, and proportionalit
    
[^2]: Logit-Q动力学对于随机团队中的高效学习

    Logit-Q Dynamics for Efficient Learning in Stochastic Teams. (arXiv:2302.09806v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.09806](http://arxiv.org/abs/2302.09806)

    本文提出了两种Logit-Q学习动力学，通过将经典和独立的对数线性学习更新与在政策上的值迭代更新相结合，实现了在随机博弈中的高效学习。通过对比和量化分析，证明了该动力学在随机团队中可以达到（接近）高效均衡。

    

    我们提出了两种Logit-Q学习动力学，将经典和独立的对数线性学习更新与一个在政策上的值迭代更新相结合，以实现在随机博弈中的高效学习。我们证明所提出的Logit-Q动力学在随机团队中达到（接近）高效均衡。我们量化了近似误差的上界。我们还展示了Logit-Q动力学对纯定态策略的合理性，并证明了动力学在奖励函数导致潜在博弈的随机博弈中的收敛性，然而只有一个智能体控制状态转换超出随机团队。关键思路是将动力学与一个虚构的场景近似，其中Q函数估计仅在有限长度的纪元中是定态的，仅用于分析。然后，我们将主要场景和虚构场景中的动力学耦合起来，以展示这两个场景由于逐步减小的步长而越来越相似。

    We present two logit-Q learning dynamics combining the classical and independent log-linear learning updates with an on-policy value iteration update for efficient learning in stochastic games. We show that the logit-Q dynamics presented reach (near) efficient equilibrium in stochastic teams. We quantify a bound on the approximation error. We also show the rationality of the logit-Q dynamics against agents following pure stationary strategies and the convergence of the dynamics in stochastic games where the reward functions induce potential games, yet only a single agent controls the state transitions beyond stochastic teams. The key idea is to approximate the dynamics with a fictional scenario where the Q-function estimates are stationary over finite-length epochs only for analysis. We then couple the dynamics in the main and fictional scenarios to show that these two scenarios become more and more similar across epochs due to the vanishing step size.
    

