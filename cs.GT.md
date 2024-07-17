# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Strategizing against Q-learners: A Control-theoretical Approach](https://arxiv.org/abs/2403.08906) | 在这篇论文中，作者探讨了Q-learning算法在游戏中受到策略性对手的操纵的敏感性，并提出了一种控制论方法来解决这个问题。 |
| [^2] | [Learning in Repeated Multi-Unit Pay-As-Bid Auctions.](http://arxiv.org/abs/2307.15193) | 本论文研究了在重复的多单位付费拍卖中学习如何出价的问题。通过在离线设置中优化出价向量，并利用多项式时间动态规划方案，设计了具有多项式时间和空间复杂度的在线学习算法。 |
| [^3] | [Learning Revenue Maximizing Menus of Lotteries and Two-Part Tariffs.](http://arxiv.org/abs/2302.11700) | 该论文研究了经济学中两种机制的可学习性：菜单抽奖和两部分票价。他们提出了第一个针对这两种机制的在线学习算法。 |

# 详细

[^1]: 针对Q-学习者的策略化对抗：一种控制论方法

    Strategizing against Q-learners: A Control-theoretical Approach

    [https://arxiv.org/abs/2403.08906](https://arxiv.org/abs/2403.08906)

    在这篇论文中，作者探讨了Q-learning算法在游戏中受到策略性对手的操纵的敏感性，并提出了一种控制论方法来解决这个问题。

    

    在这篇论文中，我们探讨了Q-learning算法(一种经典且广泛使用的强化学习方法)在游戏中对策略性对手的敏感性。我们量化了如果策略性对手了解对手的Q-learning算法，她可以利用一个天真的Q-学习者多少。为此，我们将策略行为者的问题构建为一个马尔可夫决策过程(具有涵盖所有可能Q值的连续状态空间)，就好像Q-学习算法是底层动态系统一样。我们还提出了一个基于量化的近似方案来处理连续状态空间，并在理论和数值上分析了其性能。

    arXiv:2403.08906v1 Announce Type: cross  Abstract: In this paper, we explore the susceptibility of the Q-learning algorithm (a classical and widely used reinforcement learning method) to strategic manipulation of sophisticated opponents in games. We quantify how much a strategically sophisticated agent can exploit a naive Q-learner if she knows the opponent's Q-learning algorithm. To this end, we formulate the strategic actor's problem as a Markov decision process (with a continuum state space encompassing all possible Q-values) as if the Q-learning algorithm is the underlying dynamical system. We also present a quantization-based approximation scheme to tackle the continuum state space and analyze its performance both analytically and numerically.
    
[^2]: 在重复的多单位付费拍卖中学习

    Learning in Repeated Multi-Unit Pay-As-Bid Auctions. (arXiv:2307.15193v1 [cs.GT])

    [http://arxiv.org/abs/2307.15193](http://arxiv.org/abs/2307.15193)

    本论文研究了在重复的多单位付费拍卖中学习如何出价的问题。通过在离线设置中优化出价向量，并利用多项式时间动态规划方案，设计了具有多项式时间和空间复杂度的在线学习算法。

    

    受碳排放交易方案、国债拍卖和采购拍卖的启发，这些都涉及拍卖同质的多个单位，我们考虑了如何在重复的多单位付费拍卖中学习如何出价的问题。在每个拍卖中，大量（相同的）物品将被分配给最高的出价，每个中标价等于出价本身。由于行动空间的组合性质，学习如何在付费拍卖中出价是具有挑战性的。为了克服这个挑战，我们关注离线设置，其中投标人通过只能访问其他投标人过去提交的出价来优化他们的出价向量。我们证明了离线问题的最优解可以使用多项式时间动态规划（DP）方案来获得。我们利用DP方案的结构，设计了具有多项式时间和空间复杂度的在线学习算法。

    Motivated by Carbon Emissions Trading Schemes, Treasury Auctions, and Procurement Auctions, which all involve the auctioning of homogeneous multiple units, we consider the problem of learning how to bid in repeated multi-unit pay-as-bid auctions. In each of these auctions, a large number of (identical) items are to be allocated to the largest submitted bids, where the price of each of the winning bids is equal to the bid itself. The problem of learning how to bid in pay-as-bid auctions is challenging due to the combinatorial nature of the action space. We overcome this challenge by focusing on the offline setting, where the bidder optimizes their vector of bids while only having access to the past submitted bids by other bidders. We show that the optimal solution to the offline problem can be obtained using a polynomial time dynamic programming (DP) scheme. We leverage the structure of the DP scheme to design online learning algorithms with polynomial time and space complexity under fu
    
[^3]: 学习最大化菜单抽奖和两部分票价的论文

    Learning Revenue Maximizing Menus of Lotteries and Two-Part Tariffs. (arXiv:2302.11700v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.11700](http://arxiv.org/abs/2302.11700)

    该论文研究了经济学中两种机制的可学习性：菜单抽奖和两部分票价。他们提出了第一个针对这两种机制的在线学习算法。

    

    我们通过研究在学习理论和计算经济学交叉领域中近年来蓬勃发展的一系列工作，推进了经济学中两类机制的可学习性研究，分别是菜单抽奖和两部分票价。前者是一类旨在销售多个物品的随机机制，已知能够实现超出确定性机制的收益，而后者则是针对销售单个物品多个单位（副本）的设计，适用于现实世界中的场景，如汽车或自行车共享服务等。我们关注如何从买家估值数据中学习出高收益的这类机制，涵盖多种分布设置，既有直接获得买家估值样本的情况，也有更具挑战性、研究较少的在线设置，其中买家一个接一个到来，并且对他们的估值没有分布假设。我们的主要贡献是提出了第一个针对菜单抽奖和两部分票价的在线学习算法。

    We advance a recently flourishing line of work at the intersection of learning theory and computational economics by studying the learnability of two classes of mechanisms prominent in economics, namely menus of lotteries and two-part tariffs. The former is a family of randomized mechanisms designed for selling multiple items, known to achieve revenue beyond deterministic mechanisms, while the latter is designed for selling multiple units (copies) of a single item with applications in real-world scenarios such as car or bike-sharing services. We focus on learning high-revenue mechanisms of this form from buyer valuation data in both distributional settings, where we have access to buyers' valuation samples up-front, and the more challenging and less-studied online settings, where buyers arrive one-at-a-time and no distributional assumption is made about their values.  Our main contribution is proposing the first online learning algorithms for menus of lotteries and two-part tariffs wit
    

