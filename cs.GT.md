# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stable matching as transportation](https://arxiv.org/abs/2402.13378) | 通过最优运输理论，研究了具有一致偏好的匹配市场的稳定性、效率和公平等设计目标，揭示了匹配结构特性和不同目标之间的权衡关系。 |
| [^2] | [Logit-Q Dynamics for Efficient Learning in Stochastic Teams.](http://arxiv.org/abs/2302.09806) | 本文提出了两种Logit-Q学习动力学，通过将经典和独立的对数线性学习更新与在政策上的值迭代更新相结合，实现了在随机博弈中的高效学习。通过对比和量化分析，证明了该动力学在随机团队中可以达到（接近）高效均衡。 |

# 详细

[^1]: 稳定匹配作为运输问题

    Stable matching as transportation

    [https://arxiv.org/abs/2402.13378](https://arxiv.org/abs/2402.13378)

    通过最优运输理论，研究了具有一致偏好的匹配市场的稳定性、效率和公平等设计目标，揭示了匹配结构特性和不同目标之间的权衡关系。

    

    我们研究了具有一致偏好的匹配市场，并建立了稳定性、效率和公平等共同设计目标与最优运输理论之间的联系。最优运输为追求这些目标获得的匹配的结构特性提供了新的见解，以及不同目标之间的权衡。具有一致偏好的匹配市场提供了一个易处理的简化模型，捕捉了在各种情境中的供需不平衡，比如伙伴关系形成、学校选择、器官捐赠交换，以及在匹配形成后进行转移谈判的可转让效用市场。

    arXiv:2402.13378v1 Announce Type: new  Abstract: We study matching markets with aligned preferences and establish a connection between common design objectives -- stability, efficiency, and fairness -- and the theory of optimal transport. Optimal transport gives new insights into the structural properties of matchings obtained from pursuing these objectives, and into the trade-offs between different objectives. Matching markets with aligned preferences provide a tractable stylized model capturing supply-demand imbalances in a range of settings such as partnership formation, school choice, organ donor exchange, and markets with transferable utility where bargaining over transfers happens after a match is formed.
    
[^2]: Logit-Q动力学对于随机团队中的高效学习

    Logit-Q Dynamics for Efficient Learning in Stochastic Teams. (arXiv:2302.09806v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.09806](http://arxiv.org/abs/2302.09806)

    本文提出了两种Logit-Q学习动力学，通过将经典和独立的对数线性学习更新与在政策上的值迭代更新相结合，实现了在随机博弈中的高效学习。通过对比和量化分析，证明了该动力学在随机团队中可以达到（接近）高效均衡。

    

    我们提出了两种Logit-Q学习动力学，将经典和独立的对数线性学习更新与一个在政策上的值迭代更新相结合，以实现在随机博弈中的高效学习。我们证明所提出的Logit-Q动力学在随机团队中达到（接近）高效均衡。我们量化了近似误差的上界。我们还展示了Logit-Q动力学对纯定态策略的合理性，并证明了动力学在奖励函数导致潜在博弈的随机博弈中的收敛性，然而只有一个智能体控制状态转换超出随机团队。关键思路是将动力学与一个虚构的场景近似，其中Q函数估计仅在有限长度的纪元中是定态的，仅用于分析。然后，我们将主要场景和虚构场景中的动力学耦合起来，以展示这两个场景由于逐步减小的步长而越来越相似。

    We present two logit-Q learning dynamics combining the classical and independent log-linear learning updates with an on-policy value iteration update for efficient learning in stochastic games. We show that the logit-Q dynamics presented reach (near) efficient equilibrium in stochastic teams. We quantify a bound on the approximation error. We also show the rationality of the logit-Q dynamics against agents following pure stationary strategies and the convergence of the dynamics in stochastic games where the reward functions induce potential games, yet only a single agent controls the state transitions beyond stochastic teams. The key idea is to approximate the dynamics with a fictional scenario where the Q-function estimates are stationary over finite-length epochs only for analysis. We then couple the dynamics in the main and fictional scenarios to show that these two scenarios become more and more similar across epochs due to the vanishing step size.
    

