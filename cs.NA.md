# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Optimization with Noise-Free Observations: Improved Regret Bounds via Random Exploration.](http://arxiv.org/abs/2401.17037) | 该论文研究了基于无噪声观测的贝叶斯优化问题，提出了一种基于散乱数据逼近的新算法，并引入随机探索步骤以实现接近最优填充距离的速率衰减。该算法在实现的易用性和累积遗憾边界的性能上超过了传统的GP-UCB算法，并在多个示例中优于其他贝叶斯优化策略。 |

# 详细

[^1]: 基于无噪声观测的贝叶斯优化：通过随机探索改善遗憾边界

    Bayesian Optimization with Noise-Free Observations: Improved Regret Bounds via Random Exploration. (arXiv:2401.17037v1 [cs.LG])

    [http://arxiv.org/abs/2401.17037](http://arxiv.org/abs/2401.17037)

    该论文研究了基于无噪声观测的贝叶斯优化问题，提出了一种基于散乱数据逼近的新算法，并引入随机探索步骤以实现接近最优填充距离的速率衰减。该算法在实现的易用性和累积遗憾边界的性能上超过了传统的GP-UCB算法，并在多个示例中优于其他贝叶斯优化策略。

    

    本文研究了基于无噪声观测的贝叶斯优化。我们引入了新的基于散乱数据逼近的算法，并通过随机探索步骤确保查询点的填充距离以接近最优的速率衰减。我们的算法保留了经典的GP-UCB算法的易实现性，并满足了几乎与arXiv:2002.05096中的猜想相匹配的累积遗憾边界，从而解决了COLT的一个开放问题。此外，新算法在几个示例中优于GP-UCB和其他流行的贝叶斯优化策略。

    This paper studies Bayesian optimization with noise-free observations. We introduce new algorithms rooted in scattered data approximation that rely on a random exploration step to ensure that the fill-distance of query points decays at a near-optimal rate. Our algorithms retain the ease of implementation of the classical GP-UCB algorithm and satisfy cumulative regret bounds that nearly match those conjectured in arXiv:2002.05096, hence solving a COLT open problem. Furthermore, the new algorithms outperform GP-UCB and other popular Bayesian optimization strategies in several examples.
    

