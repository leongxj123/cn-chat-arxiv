# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous-Time Best-Response and Related Dynamics in Tullock Contests with Convex Costs](https://arxiv.org/abs/2402.08541) | 本研究证明了在具有凸成本的Tullock竞赛中，连续时间最优响应动态收敛到唯一均衡点，并提供了计算近似均衡的算法。同时，我们还证明了相关离散时间动态的收敛性，这表明均衡是这些游戏中代理人行为的可靠预测器。 |
| [^2] | [Learning the Expected Core of Strictly Convex Stochastic Cooperative Games](https://arxiv.org/abs/2402.07067) | 本文研究了随机合作博弈中严格凸情况下，学习预期核心的问题。我们提出了一种名为\texttt{Common-Points-Picking}的算法，在多项式数量的样本给定的情况下，以高概率返回一个稳定分配。 |
| [^3] | [Game Connectivity and Adaptive Dynamics.](http://arxiv.org/abs/2309.10609) | 通过分析最佳响应图的连通特性，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的。这对于游戏中的动态过程有着重要意义，因为许多自适应动态会导致均衡。 |

# 详细

[^1]: Tullock竞赛中基于连续时间最优响应和相关动态的凸成本模型

    Continuous-Time Best-Response and Related Dynamics in Tullock Contests with Convex Costs

    [https://arxiv.org/abs/2402.08541](https://arxiv.org/abs/2402.08541)

    本研究证明了在具有凸成本的Tullock竞赛中，连续时间最优响应动态收敛到唯一均衡点，并提供了计算近似均衡的算法。同时，我们还证明了相关离散时间动态的收敛性，这表明均衡是这些游戏中代理人行为的可靠预测器。

    

    Tullock竞赛模型适用于各种现实情景，包括PoW区块链矿工之间的竞争、寻租和游说活动。我们利用李雅普诺夫方式的论证结果表明，在具有凸成本的Tullock竞赛中，连续时间最优响应动态收敛到唯一均衡点。然后，我们利用这一结果提供了一种计算近似均衡的算法。我们还证明了相关离散时间动态的收敛性，例如，当代理人对其他代理人的经验平均行动做出最优响应时。这些结果表明均衡是这些游戏中代理人行为的可靠预测器。

    Tullock contests model real-life scenarios that range from competition among proof-of-work blockchain miners to rent-seeking and lobbying activities. We show that continuous-time best-response dynamics in Tullock contests with convex costs converges to the unique equilibrium using Lyapunov-style arguments. We then use this result to provide an algorithm for computing an approximate equilibrium. We also establish convergence of related discrete-time dynamics, e.g., when the agents best-respond to the empirical average action of other agents. These results indicate that the equilibrium is a reliable predictor of the agents' behavior in these games.
    
[^2]: 学习严格凸的随机合作博弈的预期核心

    Learning the Expected Core of Strictly Convex Stochastic Cooperative Games

    [https://arxiv.org/abs/2402.07067](https://arxiv.org/abs/2402.07067)

    本文研究了随机合作博弈中严格凸情况下，学习预期核心的问题。我们提出了一种名为\texttt{Common-Points-Picking}的算法，在多项式数量的样本给定的情况下，以高概率返回一个稳定分配。

    

    奖励分配，也称为信用分配问题，是经济学、工程学和机器学习中的重要主题。信用分配中的一个重要概念是核心，它是稳定分配的集合，其中没有代理有动机从大联盟中偏离。在本文中，我们考虑了随机合作博弈的稳定分配学习问题，其中奖励函数被描述为具有未知分布的随机变量。在每一轮中，给定一个返回查询联盟的随机奖励的oracle，我们的目标是学习预期核心，即在期望上稳定的分配集合。在严格凸博弈类中，我们提出了一种名为\texttt{Common-Points-Picking}的算法，它在多项式数量的样本给定的情况下，以高概率返回一个稳定分配。我们的算法分析涉及到凸几何中的几个新结果的发展，包括一个

    Reward allocation, also known as the credit assignment problem, has been an important topic in economics, engineering, and machine learning. An important concept in credit assignment is the core, which is the set of stable allocations where no agent has the motivation to deviate from the grand coalition. In this paper, we consider the stable allocation learning problem of stochastic cooperative games, where the reward function is characterised as a random variable with an unknown distribution. Given an oracle that returns a stochastic reward for an enquired coalition each round, our goal is to learn the expected core, that is, the set of allocations that are stable in expectation. Within the class of strictly convex games, we present an algorithm named \texttt{Common-Points-Picking} that returns a stable allocation given a polynomial number of samples, with high probability. The analysis of our algorithm involves the development of several new results in convex geometry, including an e
    
[^3]: 游戏连通性与自适应动态

    Game Connectivity and Adaptive Dynamics. (arXiv:2309.10609v1 [econ.TH])

    [http://arxiv.org/abs/2309.10609](http://arxiv.org/abs/2309.10609)

    通过分析最佳响应图的连通特性，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的。这对于游戏中的动态过程有着重要意义，因为许多自适应动态会导致均衡。

    

    我们通过分析最佳响应图的连通特性，分析了游戏的典型结构。特别是，我们证明了几乎每个具有纯纳什均衡的“大型”通用游戏都是连通的，这意味着每个非均衡的行动配置都可以通过最佳响应路径到达每个纯纳什均衡。这对于游戏中的动态过程有着重要意义：许多自适应动态，例如带有惯性的最佳响应动态，在连通的游戏中会导致均衡。因此，存在简单的、不耦合的自适应动态，按周期游戏将几乎确定地收敛到具有纯纳什均衡的“大型”通用游戏的情况下。

    We analyse the typical structure of games in terms of the connectivity properties of their best-response graphs. In particular, we show that almost every 'large' generic game that has a pure Nash equilibrium is connected, meaning that every non-equilibrium action profile can reach every pure Nash equilibrium via best-response paths. This has implications for dynamics in games: many adaptive dynamics, such as the best-response dynamic with inertia, lead to equilibrium in connected games. It follows that there are simple, uncoupled, adaptive dynamics for which period-by-period play converges almost surely to a pure Nash equilibrium in almost every 'large' generic game that has one. We build on recent results in probabilistic combinatorics for our characterisation of game connectivity.
    

