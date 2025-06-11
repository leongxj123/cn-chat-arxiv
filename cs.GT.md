# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Feint in Multi-Player Games](https://arxiv.org/abs/2403.07932) | 该论文介绍了多人游戏中假动作的首次形式化、实现和定量评估，并证明其能够显著提高奖励收益、增加游戏多样性，且时间消耗方面开销很小。 |
| [^2] | [Stable Menus of Public Goods: A Matching Problem](https://arxiv.org/abs/2402.11370) | 研究匹配问题中的稳定菜单问题，提出了保证存在稳定解决方案的条件，对无策略性稳定匹配给出了积极和消极结果。 |

# 详细

[^1]: 多人游戏中的假动作

    Feint in Multi-Player Games

    [https://arxiv.org/abs/2403.07932](https://arxiv.org/abs/2403.07932)

    该论文介绍了多人游戏中假动作的首次形式化、实现和定量评估，并证明其能够显著提高奖励收益、增加游戏多样性，且时间消耗方面开销很小。

    

    这篇论文介绍了对多人游戏中的假动作进行了首次形式化、实现和定量评估。我们首先从多人游戏的角度对假动作进行了形式化，涉及到时间、空间和它们的集体影响。该形式化建立在非传递性主动马尔可夫游戏模型之上，其中假动作能够产生可观的影响。接下来，我们考虑了在多代理建模的最新进展下（即多智能体强化学习）在多人游戏中实施假动作的实际细节。最后，我们定量检验了我们设计的有效性，结果显示我们的假动作设计可以（1）显著提高游戏中的奖励收益；（2）显著提高多人游戏的多样性；以及（3）仅在时间消耗方面产生可忽略的开销。我们得出结论，我们的假动作设计是有效的。

    arXiv:2403.07932v1 Announce Type: cross  Abstract: This paper introduces the first formalization, implementation and quantitative evaluation of Feint in Multi-Player Games. Our work first formalizes Feint from the perspective of Multi-Player Games, in terms of the temporal, spatial, and their collective impacts. The formalization is built upon Non-transitive Active Markov Game Model, where Feint can have a considerable amount of impacts. Then, our work considers practical implementation details of Feint in Multi-Player Games, under the state-of-the-art progress of multi-agent modeling to date (namely Multi-Agent Reinforcement Learning). Finally, our work quantitatively examines the effectiveness of our design, and the results show that our design of Feint can (1) greatly improve the reward gains from the game; (2) significantly improve the diversity of Multi-Player Games; and (3) only incur negligible overheads in terms of time consumption. We conclude that our design of Feint is effec
    
[^2]: 公共物品的稳定菜单: 一个匹配问题

    Stable Menus of Public Goods: A Matching Problem

    [https://arxiv.org/abs/2402.11370](https://arxiv.org/abs/2402.11370)

    研究匹配问题中的稳定菜单问题，提出了保证存在稳定解决方案的条件，对无策略性稳定匹配给出了积极和消极结果。

    

    我们研究了在没有货币转移的情境下，代理者和公共物品之间的匹配问题。由于物品是公共的，它们没有容量限制。没有外生定义的提供物品的预算。相反，每个提供的物品必须证明其成本，导致在物品的“偏好”中存在很强的互补性。此外，鉴于已提供的其他物品，那些需求量高的物品也必须得到提供。存在一个稳定解决方案（提供的公共物品的菜单）的问题展示了丰富的组合结构。我们揭示了保证存在稳定解决方案的充分条件和必要条件，并为无策略性稳定匹配得出了积极和消极的结果。

    arXiv:2402.11370v1 Announce Type: cross  Abstract: We study a matching problem between agents and public goods, in settings without monetary transfers. Since goods are public, they have no capacity constraints. There is no exogenously defined budget of goods to be provided. Rather, each provided good must justify its cost, leading to strong complementarities in the "preferences" of goods. Furthermore, goods that are in high demand given other already-provided goods must also be provided. The question of the existence of a stable solution (a menu of public goods to be provided) exhibits a rich combinatorial structure. We uncover sufficient conditions and necessary conditions for guaranteeing the existence of a stable solution, and derive both positive and negative results for strategyproof stable matching.
    

