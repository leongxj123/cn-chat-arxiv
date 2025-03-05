# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Last-Iterate Convergence Properties of Regret-Matching Algorithms in Games.](http://arxiv.org/abs/2311.00676) | 这篇论文研究了基于遗憾匹配的算法在游戏中的最终迭代收敛性质。通过数值实验发现多个实际变体在简单的游戏中缺乏最终迭代收敛保证，而基于平滑技术的最近变体则具有最终迭代收敛性。 |
| [^2] | [Bayesian Analysis of Linear Contracts.](http://arxiv.org/abs/2211.06850) | 本文在贝叶斯框架下为线性合同在实践中普遍存在的原因进行了解释和证明，并表明在线性合同中，当委托-代理环境中存在足够不确定性时，线性合同是近乎最优的。 |

# 详细

[^1]: Regret-Matching算法在游戏中的最终迭代收敛性质

    Last-Iterate Convergence Properties of Regret-Matching Algorithms in Games. (arXiv:2311.00676v1 [cs.GT])

    [http://arxiv.org/abs/2311.00676](http://arxiv.org/abs/2311.00676)

    这篇论文研究了基于遗憾匹配的算法在游戏中的最终迭代收敛性质。通过数值实验发现多个实际变体在简单的游戏中缺乏最终迭代收敛保证，而基于平滑技术的最近变体则具有最终迭代收敛性。

    

    基于遗憾匹配的算法，特别是遗憾匹配+ (RM+)及其变种，是解决大规模双人零和游戏的最流行方法。与具有零和游戏的强最终迭代和遍历收敛性质的算法（如乐观梯度上升）不同，我们对于遗憾匹配算法的最终迭代性质几乎一无所知。鉴于最终迭代收敛性对于数值优化和模拟现实世界中的游戏学习的重要性，本文研究了各种流行的RM+变体的最终迭代收敛性质。首先，我们通过数值实验证明，包括同时RM+、交替RM+和同时预测RM+在内的几个实际变体，甚至在简单的3x3游戏中也缺乏最终迭代收敛保证。然后，我们证明了这些算法的最近变体，基于平滑技术得到了最终迭代收敛性。

    Algorithms based on regret matching, specifically regret matching$^+$ (RM$^+$), and its variants are the most popular approaches for solving large-scale two-player zero-sum games in practice. Unlike algorithms such as optimistic gradient descent ascent, which have strong last-iterate and ergodic convergence properties for zero-sum games, virtually nothing is known about the last-iterate properties of regret-matching algorithms. Given the importance of last-iterate convergence for numerical optimization reasons and relevance as modeling real-word learning in games, in this paper, we study the last-iterate convergence properties of various popular variants of RM$^+$. First, we show numerically that several practical variants such as simultaneous RM$^+$, alternating RM$^+$, and simultaneous predictive RM$^+$, all lack last-iterate convergence guarantees even on a simple $3\times 3$ game. We then prove that recent variants of these algorithms based on a smoothing technique do enjoy last-it
    
[^2]: 线性合同的贝叶斯分析

    Bayesian Analysis of Linear Contracts. (arXiv:2211.06850v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2211.06850](http://arxiv.org/abs/2211.06850)

    本文在贝叶斯框架下为线性合同在实践中普遍存在的原因进行了解释和证明，并表明在线性合同中，当委托-代理环境中存在足够不确定性时，线性合同是近乎最优的。

    

    我们在贝叶斯框架下为实践中普遍存在的线性（佣金制）合同提供了合理性的证明。我们考虑了一个隐藏行动的委托-代理模型，在该模型中，不同行动需要不同的努力量，并且代理人的努力成本是私有的。我们展示了当在委托-代理环境中存在足够的不确定性时，线性合同是近乎最优的。

    We provide a justification for the prevalence of linear (commission-based) contracts in practice under the Bayesian framework. We consider a hidden-action principal-agent model, in which actions require different amounts of effort, and the agent's cost per-unit-of-effort is private. We show that linear contracts are near-optimal whenever there is sufficient uncertainty in the principal-agent setting.
    

