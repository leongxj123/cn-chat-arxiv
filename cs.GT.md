# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Dynamic Mechanisms in Unknown Environments: A Reinforcement Learning Approach](https://arxiv.org/abs/2202.12797) | 通过将无奖励在线强化学习引入到在线机制设计问题中，我们提出了能够在未知环境中学习动态VCG机制且具有上界为$\tilde{\mathcal{O}}(T^{2/3})$的遗憾保证的新颖学习算法。 |

# 详细

[^1]: 在未知环境中学习动态机制：一种强化学习方法

    Learning Dynamic Mechanisms in Unknown Environments: A Reinforcement Learning Approach

    [https://arxiv.org/abs/2202.12797](https://arxiv.org/abs/2202.12797)

    通过将无奖励在线强化学习引入到在线机制设计问题中，我们提出了能够在未知环境中学习动态VCG机制且具有上界为$\tilde{\mathcal{O}}(T^{2/3})$的遗憾保证的新颖学习算法。

    

    动态机制设计研究了机制设计者在时变环境中应该如何在代理之间分配资源。我们考虑了一种问题，即代理根据未知的马尔可夫决策过程(MDP)与机制设计者互动，在这个过程中代理的奖励和机制设计者的状态根据一个带有未知奖励函数和转移核的情节MDP演化。我们关注在线设置下的线性函数近似，并提出了新颖的学习算法，在多轮互动中恢复动态Vickrey-Clarke-Grove(VCG)机制。我们方法的一个关键贡献是将无奖励在线强化学习(RL)结合进来，以帮助在丰富的策略空间中进行探索，从而估计动态VCG机制中的价格。我们展示了我们提出的方法的遗憾上界为$\tilde{\mathcal{O}}(T^{2/3})$，并进一步设计了一个下界，以展示我们方法的...

    arXiv:2202.12797v2 Announce Type: replace  Abstract: Dynamic mechanism design studies how mechanism designers should allocate resources among agents in a time-varying environment. We consider the problem where the agents interact with the mechanism designer according to an unknown Markov Decision Process (MDP), where agent rewards and the mechanism designer's state evolve according to an episodic MDP with unknown reward functions and transition kernels. We focus on the online setting with linear function approximation and propose novel learning algorithms to recover the dynamic Vickrey-Clarke-Grove (VCG) mechanism over multiple rounds of interaction. A key contribution of our approach is incorporating reward-free online Reinforcement Learning (RL) to aid exploration over a rich policy space to estimate prices in the dynamic VCG mechanism. We show that the regret of our proposed method is upper bounded by $\tilde{\mathcal{O}}(T^{2/3})$ and further devise a lower bound to show that our a
    

