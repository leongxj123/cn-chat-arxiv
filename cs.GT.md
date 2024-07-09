# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bandit Profit-maximization for Targeted Marketing](https://arxiv.org/abs/2403.01361) | 该论文研究了针对目标营销的强盗利润最大化问题，并提出了在敌对强盗情境下的近乎最优算法。 |
| [^2] | [Strategically-Robust Learning Algorithms for Bidding in First-Price Auctions](https://arxiv.org/abs/2402.07363) | 本论文提出了一种在一价拍卖中进行竞标的新颖算法，并通过分析证明了其在战略背景下的效果。具体而言，这些算法在面对对策性卖家时表现良好，激励买家进行真实的交易，并获得了最佳的后悔结果。 |
| [^3] | [On Connected Strongly-Proportional Cake-Cutting](https://arxiv.org/abs/2312.15326) | 该论文研究了一种公平划分可划分异质资源的问题，即蛋糕切分。该论文确定了存在一种连通的强比例切分方式，并提供了相应的算法和简单刻画。 |

# 详细

[^1]: 针对目标营销的强盗利润最大化

    Bandit Profit-maximization for Targeted Marketing

    [https://arxiv.org/abs/2403.01361](https://arxiv.org/abs/2403.01361)

    该论文研究了针对目标营销的强盗利润最大化问题，并提出了在敌对强盗情境下的近乎最优算法。

    

    我们研究了一个顺序利润最大化问题，优化价格和像营销支出这样的辅助变量。具体来说，我们旨在在一个任意序列的多个需求曲线上最大化利润，每个曲线依赖于一个不同的辅助变量，但共享相同的价格。一个典型的例子是针对营销，其中一家公司（卖方）希望在多个市场上销售产品。公司可以为不同市场投入不同的营销支出以优化客户获取，但必须在所有市场上保持相同的价格。此外，市场可能具有异质的需求曲线，每个需求曲线对价格和营销支出的响应方式不同。公司的目标是最大化毛利润，即总收入减去营销成本。

    arXiv:2403.01361v1 Announce Type: new  Abstract: We study a sequential profit-maximization problem, optimizing for both price and ancillary variables like marketing expenditures. Specifically, we aim to maximize profit over an arbitrary sequence of multiple demand curves, each dependent on a distinct ancillary variable, but sharing the same price. A prototypical example is targeted marketing, where a firm (seller) wishes to sell a product over multiple markets. The firm may invest different marketing expenditures for different markets to optimize customer acquisition, but must maintain the same price across all markets. Moreover, markets may have heterogeneous demand curves, each responding to prices and marketing expenditures differently. The firm's objective is to maximize its gross profit, the total revenue minus marketing costs.   Our results are near-optimal algorithms for this class of problems in an adversarial bandit setting, where demand curves are arbitrary non-adaptive seque
    
[^2]: 基于策略稳定性的学习算法在一价拍卖中的竞标

    Strategically-Robust Learning Algorithms for Bidding in First-Price Auctions

    [https://arxiv.org/abs/2402.07363](https://arxiv.org/abs/2402.07363)

    本论文提出了一种在一价拍卖中进行竞标的新颖算法，并通过分析证明了其在战略背景下的效果。具体而言，这些算法在面对对策性卖家时表现良好，激励买家进行真实的交易，并获得了最佳的后悔结果。

    

    在游戏理论和机器学习的交界处，学习在重复的一价拍卖中进行竞标是一个基本问题，由于显示广告转向一价拍卖，最近受到了广泛关注。在这项工作中，我们提出了一个新颖的凹函数形式，用于一价拍卖中纯策略的竞标，并将其用于分析这个问题的自然梯度上升算法。重要的是，我们的分析超越了过去工作的差距，还考虑了在线广告市场的战略背景，其中部署了竞标算法 - 我们证明了我们的算法不会被策略性卖家利用，并且它们激励买家诚实交易。具体而言，我们证明了当最高竞争出价通过对抗方式生成时，我们的算法达到了$O(\sqrt{T})$的后悔，并表明没有更好的在线算法可以做得更好。进一步证明了当最高竞争出价通过对抗方式生成时，我们的算法达到了$O(\log T)$的后悔。

    Learning to bid in repeated first-price auctions is a fundamental problem at the interface of game theory and machine learning, which has seen a recent surge in interest due to the transition of display advertising to first-price auctions. In this work, we propose a novel concave formulation for pure-strategy bidding in first-price auctions, and use it to analyze natural Gradient-Ascent-based algorithms for this problem. Importantly, our analysis goes beyond regret, which was the typical focus of past work, and also accounts for the strategic backdrop of online-advertising markets where bidding algorithms are deployed -- we prove that our algorithms cannot be exploited by a strategic seller and that they incentivize truth-telling for the buyer.   Concretely, we show that our algorithms achieve $O(\sqrt{T})$ regret when the highest competing bids are generated adversarially, and show that no online algorithm can do better. We further prove that the regret improves to $O(\log T)$ when th
    
[^3]: 关于连通且强比例切蛋糕的研究

    On Connected Strongly-Proportional Cake-Cutting

    [https://arxiv.org/abs/2312.15326](https://arxiv.org/abs/2312.15326)

    该论文研究了一种公平划分可划分异质资源的问题，即蛋糕切分。该论文确定了存在一种连通的强比例切分方式，并提供了相应的算法和简单刻画。

    

    我们研究了在一组代理人中如何公平地分配可划分的异质资源，也称为蛋糕。我们确定了存在着一种分配方式，每个代理人都会收到一个价值严格超过他们比例份额的连续部分，也称为*强比例分配*。我们提出了一个算法，可以使用最多$n \cdot 2^{n-1}$个查询来确定是否存在一个连通的强比例分配。对于具有严格正估值的代理人，我们提供了一个更简单的刻画，并且证明了确定是否存在一个连通的强比例分配所需的查询数量是$\Theta(n^2)$。我们的证明是构造性的，并且当存在时，给出了一个连通的强比例分配，使用了类似数量的查询。

    arXiv:2312.15326v2 Announce Type: replace-cross Abstract: We investigate the problem of fairly dividing a divisible heterogeneous resource, also known as a cake, among a set of agents. We characterize the existence of an allocation in which every agent receives a contiguous piece worth strictly more than their proportional share, also known as a *strongly-proportional allocation*. The characterization is supplemented with an algorithm that determines the existence of a connected strongly-proportional allocation using at most $n \cdot 2^{n-1}$ queries. We provide a simpler characterization for agents with strictly positive valuations, and show that the number of queries required to determine the existence of a connected strongly-proportional allocation is in $\Theta(n^2)$. Our proofs are constructive and yield a connected strongly-proportional allocation, when it exists, using a similar number of queries.
    

