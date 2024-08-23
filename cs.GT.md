# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stake-governed tug-of-war and the biased infinity Laplacian.](http://arxiv.org/abs/2206.08300) | 本文提出一种带有分配预算的拔河游戏数学模型，每个玩家依靠自己的预算进行赌注，在给定的轮次中，根据硬币翻转的结果获得移动的权利，当计数器到达边界时，其中一个玩家向另一个支付由边界定点确定的金额。 |
| [^2] | [Fair Division with Two-Sided Preferences.](http://arxiv.org/abs/2206.05879) | 本文研究公平分配问题中，考虑了玩家对团队的偏好，提出了两侧稳定、EF1和交换稳定的分配方式，并证明了其存在性和多项式时间计算复杂度，同时探讨了EF1和激励无嫉妒的相容性。 |

# 详细

[^1]: 赌注控制的拔河游戏与有偏无限拉普拉斯算子。

    Stake-governed tug-of-war and the biased infinity Laplacian. (arXiv:2206.08300v2 [math.PR] UPDATED)

    [http://arxiv.org/abs/2206.08300](http://arxiv.org/abs/2206.08300)

    本文提出一种带有分配预算的拔河游戏数学模型，每个玩家依靠自己的预算进行赌注，在给定的轮次中，根据硬币翻转的结果获得移动的权利，当计数器到达边界时，其中一个玩家向另一个支付由边界定点确定的金额。

    

    在拔河游戏中，两个玩家通过沿图形的边移动计数器来竞争，每个人根据可能存在偏差的硬币翻转决定在给定轮次中获得移动的权利。当计数器到达边界时，即固定的顶点子集，游戏结束，其中一个玩家向另一个支付由边界定点确定的金额。在本文中，我们提供了一类带有分配预算的拔河游戏的数学处理：每个玩家最初获得一个固定的预算，她在整个游戏中依靠这个预算进行下一轮的赌注，她赢得轮次的概率是

    In tug-of-war, two players compete by moving a counter along edges of a graph, each winning the right to move at a given turn according to the flip of a possibly biased coin. The game ends when the counter reaches the boundary, a fixed subset of the vertices, at which point one player pays the other an amount determined by the boundary vertex. Economists and mathematicians have independently studied tug-of-war for many years, focussing respectively on resource-allocation forms of the game, in which players iteratively spend precious budgets in an effort to influence the bias of the coins that determine the turn victors; and on PDE arising in fine mesh limits of the constant-bias game in a Euclidean setting.  In this article, we offer a mathematical treatment of a class of tug-of-war games with allocated budgets: each player is initially given a fixed budget which she draws on throughout the game to offer a stake at the start of each turn, and her probability of winning the turn is the 
    
[^2]: 两侧偏好的公平分配

    Fair Division with Two-Sided Preferences. (arXiv:2206.05879v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2206.05879](http://arxiv.org/abs/2206.05879)

    本文研究公平分配问题中，考虑了玩家对团队的偏好，提出了两侧稳定、EF1和交换稳定的分配方式，并证明了其存在性和多项式时间计算复杂度，同时探讨了EF1和激励无嫉妒的相容性。

    

    本文研究了一个公平分配的设置，在这个设置中，一些玩家需要在一组团队之间公平地分配。在我们的模型中，不仅团队像经典公平分配一样对玩家有偏好，而且玩家也对团队有偏好。我们的研究重点是保证团队能获得最多一个人的不满（EF1），同时两侧能保持稳定。我们展示出，一种满足EF1、交换稳定和个人稳定的分配总是存在的，并且可以在多项式时间内计算出来，即便团队对玩家有正值或负值的情况下也是如此。同样，一个满足EF1松弛条件并且平衡及交换稳定的分配可以有效地计算。当团队对玩家的价值非负时，我们证明了可以存在EF1和Pareto最优分配，如果估价是二元的，那么可在多项式时间内找到。我们还研究了EF1和激励无嫉妒的相容性。

    We study a fair division setting in which a number of players are to be fairly distributed among a set of teams. In our model, not only do the teams have preferences over the players as in the canonical fair division setting, but the players also have preferences over the teams. We focus on guaranteeing envy-freeness up to one player (EF1) for the teams together with a stability condition for both sides. We show that an allocation satisfying EF1, swap stability, and individual stability always exists and can be computed in polynomial time, even when teams may have positive or negative values for players. Similarly, a balanced and swap stable allocation that satisfies a relaxation of EF1 can be computed efficiently. When teams have nonnegative values for players, we prove that an EF1 and Pareto optimal allocation exists and, if the valuations are binary, can be found in polynomial time. We also examine the compatibility between EF1 and justified envy-freeness.
    

