# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Paths to Equilibrium in Normal-Form Games](https://arxiv.org/abs/2403.18079) | 本文研究在多智体强化学习中的策略序列，探讨满足特定约束的策略路径，即满足路径，对于构建终止于均衡策略的路径具有重要意义。 |
| [^2] | [On the Redistribution of Maximal Extractable Value: A Dynamic Mechanism](https://arxiv.org/abs/2402.15849) | 该论文提出了一个动态机制，通过更新最大可提取价值（MEV）提取率，以在矿工和用户之间维持健康平衡。 |
| [^3] | [Incentivizing Data Sharing for Energy Forecasting: Analytics Markets with Correlated Data.](http://arxiv.org/abs/2310.06000) | 该论文开发了一个考虑相关性的分析市场，通过采用Shapley值的归因策略来分配收入，促进了数据共享以提高能源预测的准确性。 |

# 详细

[^1]: 正态形式博弈中的均衡路径

    Paths to Equilibrium in Normal-Form Games

    [https://arxiv.org/abs/2403.18079](https://arxiv.org/abs/2403.18079)

    本文研究在多智体强化学习中的策略序列，探讨满足特定约束的策略路径，即满足路径，对于构建终止于均衡策略的路径具有重要意义。

    

    在多智体强化学习（MARL）中，智体会反复在时间上交互，并随着新数据的到来修订他们的策略，从而产生一系列策略概况。本文研究满足一种由强化学习中政策更新启发的成对约束的策略序列，其中在第 $t$ 期最优应答的智体在下一期 $t+1$ 不会改变其策略。这种约束仅要求优化智体不更改策略，但并不以任何方式限制其他非最优化智体，因此允许探索。具有此属性的序列被称为满足路径，并在许多 MARL 算法中自然出现。关于战略动态的一个基本问题是：对于给定的博弈和初始策略概况，是否总可以构建一个终止于均衡策略的满足路径？这个问题的解决对应着一些重要含义。

    arXiv:2403.18079v1 Announce Type: cross  Abstract: In multi-agent reinforcement learning (MARL), agents repeatedly interact across time and revise their strategies as new data arrives, producing a sequence of strategy profiles. This paper studies sequences of strategies satisfying a pairwise constraint inspired by policy updating in reinforcement learning, where an agent who is best responding in period $t$ does not switch its strategy in the next period $t+1$. This constraint merely requires that optimizing agents do not switch strategies, but does not constrain the other non-optimizing agents in any way, and thus allows for exploration. Sequences with this property are called satisficing paths, and arise naturally in many MARL algorithms. A fundamental question about strategic dynamics is such: for a given game and initial strategy profile, is it always possible to construct a satisficing path that terminates at an equilibrium strategy? The resolution of this question has implication
    
[^2]: 关于最大可提取价值再分配的动态机制

    On the Redistribution of Maximal Extractable Value: A Dynamic Mechanism

    [https://arxiv.org/abs/2402.15849](https://arxiv.org/abs/2402.15849)

    该论文提出了一个动态机制，通过更新最大可提取价值（MEV）提取率，以在矿工和用户之间维持健康平衡。

    

    最大可提取价值（MEV）已经成为区块链系统设计领域的一个新前沿。去中心化和金融的结合赋予了区块生产者（即矿工）不仅选择和添加交易到区块链的权力，更重要的是，还可以对其进行排序，以尽可能多地从中获利。虽然这种价格对于区块生产者提供的服务来说可能是不可避免的，但链上用户从长远来看可能更愿意使用不那么掠夺性的系统。本文提出将MEV提取率纳入协议设计空间中。我们的目标是利用此参数来维持矿工（需要得到补偿）和用户（需要感到被鼓励进行交易）之间的健康平衡。受EIP-1559引入的交易费原则的启发，我们设计了一个动态机制，旨在通过更新MEV提取率来达到稳定目标。

    arXiv:2402.15849v1 Announce Type: cross  Abstract: Maximal Extractable Value (MEV) has emerged as a new frontier in the design of blockchain systems. The marriage between decentralization and finance gives the power to block producers (a.k.a., miners) not only to select and add transactions to the blockchain but, crucially, also to order them so as to extract as much financial gain as possible for themselves. Whilst this price may be unavoidable for the service provided by block producers, users of the chain may in the long run prefer to use less predatory systems. In this paper, we propose to make the MEV extraction rate part of the protocol design space. Our aim is to leverage this parameter to maintain a healthy balance between miners (who need to be compensated) and users (who need to feel encouraged to transact). Inspired by the principles introduced by EIP-1559 for transaction fees, we design a dynamic mechanism which updates the MEV extraction rate with the goal of stabilizing i
    
[^3]: 鼓励数据共享以进行能源预测：具有相关数据的分析市场

    Incentivizing Data Sharing for Energy Forecasting: Analytics Markets with Correlated Data. (arXiv:2310.06000v1 [econ.GN])

    [http://arxiv.org/abs/2310.06000](http://arxiv.org/abs/2310.06000)

    该论文开发了一个考虑相关性的分析市场，通过采用Shapley值的归因策略来分配收入，促进了数据共享以提高能源预测的准确性。

    

    准确地预测不确定的电力产量对于电力市场的社会福利具有益处，可以减少平衡资源的需求。将这种预测描述为一项分析任务，当前文献提出了以分析市场作为激励手段来改善精度的数据共享方法，例如利用时空相关性。挑战在于，当相关数据用作预测的输入特征时，重叠信息的价值在于收入分配方面使市场设计复杂化，因为这种价值在本质上是组合的。我们为风力预测应用开发了一个考虑相关性的分析市场。为了分配收入，我们采用了基于Shapley值的归因策略，将代理人的特征视为玩家，将他们的相互作用视为一个特征函数博弈。我们说明了描述这种博弈的多种选项，每个选项都有因果细微差别，影响着特征相关时的市场行为。

    Reliably forecasting uncertain power production is beneficial for the social welfare of electricity markets by reducing the need for balancing resources. Describing such forecasting as an analytics task, the current literature proposes analytics markets as an incentive for data sharing to improve accuracy, for instance by leveraging spatio-temporal correlations. The challenge is that, when used as input features for forecasting, correlated data complicates the market design with respect to the revenue allocation, as the value of overlapping information is inherently combinatorial. We develop a correlation-aware analytics market for a wind power forecasting application. To allocate revenue, we adopt a Shapley value-based attribution policy, framing the features of agents as players and their interactions as a characteristic function game. We illustrate that there are multiple options to describe such a game, each having causal nuances that influence market behavior when features are cor
    

