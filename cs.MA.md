# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Independent RL for Cooperative-Competitive Agents: A Mean-Field Perspective](https://arxiv.org/abs/2403.11345) | 本文从均场视角研究了独立强化学习在合作竞争代理中的应用，提出了一种可实现纳什均衡的线性二次结构RL方法，并通过考虑无限代理数量的情况来解决有限人口环境中的非稳态性问题。 |

# 详细

[^1]: 独立强化学习用于合作竞争Agent：均场视角

    Independent RL for Cooperative-Competitive Agents: A Mean-Field Perspective

    [https://arxiv.org/abs/2403.11345](https://arxiv.org/abs/2403.11345)

    本文从均场视角研究了独立强化学习在合作竞争代理中的应用，提出了一种可实现纳什均衡的线性二次结构RL方法，并通过考虑无限代理数量的情况来解决有限人口环境中的非稳态性问题。

    

    在本文中，我们研究了分成团队的代理之间的强化学习（RL），每个团队内部存在合作，但不同团队之间存在非零和的竞争。为了开发一种可以明确实现纳什均衡的RL方法，我们专注于线性二次结构。此外，为了解决有限人口环境中由多智能体交互引起的非稳态性，我们考虑每个团队内代理数量无限的情况，即均场设置。这导致了一个广义和的LQ均场类型博弈（GS-MFTGs）。我们在标准逆可逆条件下表征了GS-MFTG的纳什均衡（NE）。然后证明了这个MFTG NE在有限人口博弈中为$\mathcal{O}(1/M)$-NE，其中$M$是每个团队中代理数量的下界。这些结构性结果推动了一个名为多玩家递进式自然Pol的算法。

    arXiv:2403.11345v1 Announce Type: cross  Abstract: We address in this paper Reinforcement Learning (RL) among agents that are grouped into teams such that there is cooperation within each team but general-sum (non-zero sum) competition across different teams. To develop an RL method that provably achieves a Nash equilibrium, we focus on a linear-quadratic structure. Moreover, to tackle the non-stationarity induced by multi-agent interactions in the finite population setting, we consider the case where the number of agents within each team is infinite, i.e., the mean-field setting. This results in a General-Sum LQ Mean-Field Type Game (GS-MFTGs). We characterize the Nash equilibrium (NE) of the GS-MFTG, under a standard invertibility condition. This MFTG NE is then shown to be $\mathcal{O}(1/M)$-NE for the finite population game where $M$ is a lower bound on the number of agents in each team. These structural results motivate an algorithm called Multi-player Receding-horizon Natural Pol
    

