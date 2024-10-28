# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Bayan Algorithm: Detecting Communities in Networks Through Exact and Approximate Optimization of Modularity.](http://arxiv.org/abs/2209.04562) | 提出了一种名为Bayan的社区检测算法，通过精确或近似优化模块度的方法，它能够返回最优或接近最优的分区，并且比其他算法快数倍，并能够在合成和真实网络数据集上准确地找到地面真实社区。 |

# 详细

[^1]: Bayan算法：通过对模块度的精确和近似优化来检测网络中的社区

    The Bayan Algorithm: Detecting Communities in Networks Through Exact and Approximate Optimization of Modularity. (arXiv:2209.04562v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2209.04562](http://arxiv.org/abs/2209.04562)

    提出了一种名为Bayan的社区检测算法，通过精确或近似优化模块度的方法，它能够返回最优或接近最优的分区，并且比其他算法快数倍，并能够在合成和真实网络数据集上准确地找到地面真实社区。

    

    社区检测是网络科学中的经典问题，具有广泛的应用。在众多方法中，最常见的方法是最大化模块度。尽管启发式模块度最大化算法设计理念和广泛采用，但很少返回最佳分区或类似分区。我们提出了一种专门的算法Bayan，它返回具有最优或接近最优分区保证的分区。Bayan算法的核心是一种分支限界方案，它解决了问题的整数规划公式以达到最优或近似最优的目的。我们证明Bayan在合成基准和真实网络节点标签的检索地面真实社区方面具有独特的准确性和稳定性，比其他21种算法快数倍，可以找到最优分区的实例。

    Community detection is a classic problem in network science with extensive applications in various fields. Among numerous approaches, the most common method is modularity maximization. Despite their design philosophy and wide adoption, heuristic modularity maximization algorithms rarely return an optimal partition or anything similar. We propose a specialized algorithm, Bayan, which returns partitions with a guarantee of either optimality or proximity to an optimal partition. At the core of the Bayan algorithm is a branch-and-cut scheme that solves an integer programming formulation of the problem to optimality or approximate it within a factor. We demonstrate Bayan's distinctive accuracy and stability over 21 other algorithms in retrieving ground-truth communities in synthetic benchmarks and node labels in real networks. Bayan is several times faster than open-source and commercial solvers for modularity maximization making it capable of finding optimal partitions for instances that c
    

