# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Conversion with Switching Costs: Robust and Learning-Augmented Algorithms.](http://arxiv.org/abs/2310.20598) | 本论文介绍并研究了在线转换及其带有切换成本的问题，并提出了具有竞争力的阈值算法以及学习增强算法，这些算法在最小化和最大化变体中都表现出优越性能。 |

# 详细

[^1]: 在线转换及其带有切换成本：稳健和学习增强算法

    Online Conversion with Switching Costs: Robust and Learning-Augmented Algorithms. (arXiv:2310.20598v2 [cs.DS] UPDATED)

    [http://arxiv.org/abs/2310.20598](http://arxiv.org/abs/2310.20598)

    本论文介绍并研究了在线转换及其带有切换成本的问题，并提出了具有竞争力的阈值算法以及学习增强算法，这些算法在最小化和最大化变体中都表现出优越性能。

    

    我们介绍并研究在线转换及其带有切换成本的问题。这个问题涵盖了能源和可持续性交叉领域中出现的一系列新问题。在这个问题中，在线玩家试图在固定的时间段内购买（或销售）资产的分数份额，每个时间步骤都会公布成本（或价格）函数，并且玩家必须做出不可撤消的决策，决定转换的资产数量。当玩家连续时间步骤中改变决策时，也会产生切换成本，即在购买量增加或减少时。我们介绍了在这个问题的最小化和最大化变体中具有竞争力（稳健）的基于阈值的算法，并且证明它们是确定性在线算法中的最优算法。然后，我们提出了学习增强算法，利用不可信的黑盒建议（例如机器学习模型的预测）来显著改善算法性能。

    We introduce and study online conversion with switching costs, a family of online problems that capture emerging problems at the intersection of energy and sustainability. In this problem, an online player attempts to purchase (alternatively, sell) fractional shares of an asset during a fixed time horizon with length $T$. At each time step, a cost function (alternatively, price function) is revealed, and the player must irrevocably decide an amount of asset to convert. The player also incurs a switching cost whenever their decision changes in consecutive time steps, i.e., when they increase or decrease their purchasing amount. We introduce competitive (robust) threshold-based algorithms for both the minimization and maximization variants of this problem, and show they are optimal among deterministic online algorithms. We then propose learning-augmented algorithms that take advantage of untrusted black-box advice (such as predictions from a machine learning model) to achieve significant
    

