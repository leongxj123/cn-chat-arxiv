# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cooperation and Control in Delegation Games](https://arxiv.org/abs/2402.15821) | 本文在委托游戏中探讨了控制问题（代理人未能按照其委托人的偏好行事）和合作问题（代理人未能良好地协作），并分析了对齐和能力对委托人福利的影响。en_tdlr: This paper explores the issues of control (agents failing to act in line with their principals' preferences) and cooperation (agents failing to work well together) in delegation games, analyzing how alignment and capabilities impact principals' welfare. |
| [^2] | [On Connected Strongly-Proportional Cake-Cutting](https://arxiv.org/abs/2312.15326) | 该论文研究了一种公平划分可划分异质资源的问题，即蛋糕切分。该论文确定了存在一种连通的强比例切分方式，并提供了相应的算法和简单刻画。 |

# 详细

[^1]: 委托游戏中的合作与控制

    Cooperation and Control in Delegation Games

    [https://arxiv.org/abs/2402.15821](https://arxiv.org/abs/2402.15821)

    本文在委托游戏中探讨了控制问题（代理人未能按照其委托人的偏好行事）和合作问题（代理人未能良好地协作），并分析了对齐和能力对委托人福利的影响。en_tdlr: This paper explores the issues of control (agents failing to act in line with their principals' preferences) and cooperation (agents failing to work well together) in delegation games, analyzing how alignment and capabilities impact principals' welfare.

    

    许多涉及人类和机器的感兴趣的场景 - 从虚拟个人助理到自动驾驶车辆 - 可以自然地建模为委托人（人类）委托给代理人（机器），这些代理人之后代表他们的委托人相互交互。我们将这些多委托人，多代理人的情况称为委托游戏。在这类游戏中，存在两种重要的失败模式：控制问题（代理人未能按照其委托人的偏好行事）和合作问题（代理人未能良好地协作）。在本文中，我们形式化和分析这些问题，进一步将其解释为对齐（参与者是否具有相似的偏好？）和能力（参与者在满足这些偏好方面的能力如何？）的问题。我们理论上和经验上展示了这些措施如何确定委托人的福利，如何可以使用有限的观察来估计这些措施，

    arXiv:2402.15821v1 Announce Type: cross  Abstract: Many settings of interest involving humans and machines -- from virtual personal assistants to autonomous vehicles -- can naturally be modelled as principals (humans) delegating to agents (machines), which then interact with each other on their principals' behalf. We refer to these multi-principal, multi-agent scenarios as delegation games. In such games, there are two important failure modes: problems of control (where an agent fails to act in line their principal's preferences) and problems of cooperation (where the agents fail to work well together). In this paper we formalise and analyse these problems, further breaking them down into issues of alignment (do the players have similar preferences?) and capabilities (how competent are the players at satisfying those preferences?). We show -- theoretically and empirically -- how these measures determine the principals' welfare, how they can be estimated using limited observations, and 
    
[^2]: 关于连通且强比例切蛋糕的研究

    On Connected Strongly-Proportional Cake-Cutting

    [https://arxiv.org/abs/2312.15326](https://arxiv.org/abs/2312.15326)

    该论文研究了一种公平划分可划分异质资源的问题，即蛋糕切分。该论文确定了存在一种连通的强比例切分方式，并提供了相应的算法和简单刻画。

    

    我们研究了在一组代理人中如何公平地分配可划分的异质资源，也称为蛋糕。我们确定了存在着一种分配方式，每个代理人都会收到一个价值严格超过他们比例份额的连续部分，也称为*强比例分配*。我们提出了一个算法，可以使用最多$n \cdot 2^{n-1}$个查询来确定是否存在一个连通的强比例分配。对于具有严格正估值的代理人，我们提供了一个更简单的刻画，并且证明了确定是否存在一个连通的强比例分配所需的查询数量是$\Theta(n^2)$。我们的证明是构造性的，并且当存在时，给出了一个连通的强比例分配，使用了类似数量的查询。

    arXiv:2312.15326v2 Announce Type: replace-cross Abstract: We investigate the problem of fairly dividing a divisible heterogeneous resource, also known as a cake, among a set of agents. We characterize the existence of an allocation in which every agent receives a contiguous piece worth strictly more than their proportional share, also known as a *strongly-proportional allocation*. The characterization is supplemented with an algorithm that determines the existence of a connected strongly-proportional allocation using at most $n \cdot 2^{n-1}$ queries. We provide a simpler characterization for agents with strictly positive valuations, and show that the number of queries required to determine the existence of a connected strongly-proportional allocation is in $\Theta(n^2)$. Our proofs are constructive and yield a connected strongly-proportional allocation, when it exists, using a similar number of queries.
    

