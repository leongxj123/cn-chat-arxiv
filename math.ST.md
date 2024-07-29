# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Differentially Private Synthetic Data Generation](https://arxiv.org/abs/2402.08012) | 本文提出了一种在线差分隐私合成数据生成的多项式时间算法，在超立方体数据流上实现了近乎最优的精度界限，也推广了之前关于计数查询的连续发布模型的工作，仅需要额外的多项式对数因子。 |

# 详细

[^1]: 在线差分隐私合成数据生成

    Online Differentially Private Synthetic Data Generation

    [https://arxiv.org/abs/2402.08012](https://arxiv.org/abs/2402.08012)

    本文提出了一种在线差分隐私合成数据生成的多项式时间算法，在超立方体数据流上实现了近乎最优的精度界限，也推广了之前关于计数查询的连续发布模型的工作，仅需要额外的多项式对数因子。

    

    本文提出了一种用于在线差分隐私合成数据生成的多项式时间算法。对于在超立方体$[0,1]^d$内的数据流和无限时间范围，我们开发了一种在线算法，每个时间$t$都生成一个差分隐私合成数据集。该算法在1-Wasserstein距离上实现了近乎最优的精度界限：当$d\geq 2$时为$O(t^{-1/d}\log(t)$，当$d=1$时为$O(t^{-1}\log^{4.5}(t)$。这个结果将之前关于计数查询的连续发布模型的工作推广到包括Lipschitz查询。与离线情况不同，离线情况下整个数据集一次性可用，我们的方法仅需要在精度界限中额外的多项式对数因子。

    We present a polynomial-time algorithm for online differentially private synthetic data generation. For a data stream within the hypercube $[0,1]^d$ and an infinite time horizon, we develop an online algorithm that generates a differentially private synthetic dataset at each time $t$. This algorithm achieves a near-optimal accuracy bound of $O(t^{-1/d}\log(t))$ for $d\geq 2$ and $O(t^{-1}\log^{4.5}(t))$ for $d=1$ in the 1-Wasserstein distance. This result generalizes the previous work on the continual release model for counting queries to include Lipschitz queries. Compared to the offline case, where the entire dataset is available at once, our approach requires only an extra polylog factor in the accuracy bound.
    

