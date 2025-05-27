# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Logarithmic Regret for Matrix Games against an Adversary with Noisy Bandit Feedback.](http://arxiv.org/abs/2306.13233) | 本文提出了一种算法，在带有嘈杂贝叶斯反馈的零和矩阵博弈中，实现了对数遗憾策略。 |

# 详细

[^1]: 基于带有嘈杂贝叶斯反馈的零和矩阵博弈的对数遗憾对策略

    Logarithmic Regret for Matrix Games against an Adversary with Noisy Bandit Feedback. (arXiv:2306.13233v1 [cs.LG])

    [http://arxiv.org/abs/2306.13233](http://arxiv.org/abs/2306.13233)

    本文提出了一种算法，在带有嘈杂贝叶斯反馈的零和矩阵博弈中，实现了对数遗憾策略。

    

    本文研究了零和矩阵博弈的变种，其中每步行选手选择一行$i$，列选手选择一列$j$，行选手收到平均值为$A_{i,j}$的嘈杂奖励。行选手的目标是尽可能地累积奖励，即使对手是一个对手性列选手。该文提出了一种策略，该策略证明在$m \times n$矩阵博弈中，实现了$O(\sqrt{mnT})$对数遗憾，进一步提高了UCB风格算法所获得的$O(m\sqrt{nT})$对数遗憾。

    This paper considers a variant of zero-sum matrix games where at each timestep the row player chooses row $i$, the column player chooses column $j$, and the row player receives a noisy reward with mean $A_{i,j}$. The objective of the row player is to accumulate as much reward as possible, even against an adversarial column player. If the row player uses the EXP3 strategy, an algorithm known for obtaining $\sqrt{T}$ regret against an arbitrary sequence of rewards, it is immediate that the row player also achieves $\sqrt{T}$ regret relative to the Nash equilibrium in this game setting. However, partly motivated by the fact that the EXP3 strategy is myopic to the structure of the game, O'Donoghue et al. (2021) proposed a UCB-style algorithm that leverages the game structure and demonstrated that this algorithm greatly outperforms EXP3 empirically. While they showed that this UCB-style algorithm achieved $\sqrt{T}$ regret, in this paper we ask if there exists an algorithm that provably ach
    

