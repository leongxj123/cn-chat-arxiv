# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal partitioning of directed acyclic graphs with dependent costs between clusters.](http://arxiv.org/abs/2308.03970) | 本论文提出了一种名为DCMAP的算法，用于对具有依赖成本的有向无环图进行最优分区。该算法通过优化基于DAG和集群映射的成本函数来寻找所有最优集群，并在途中返回接近最优解。实验证明在复杂系统的DBN模型中，该算法具有时间效率性。 |

# 详细

[^1]: 对具有依赖成本的有向无环图进行最优分区

    Optimal partitioning of directed acyclic graphs with dependent costs between clusters. (arXiv:2308.03970v1 [cs.DS])

    [http://arxiv.org/abs/2308.03970](http://arxiv.org/abs/2308.03970)

    本论文提出了一种名为DCMAP的算法，用于对具有依赖成本的有向无环图进行最优分区。该算法通过优化基于DAG和集群映射的成本函数来寻找所有最优集群，并在途中返回接近最优解。实验证明在复杂系统的DBN模型中，该算法具有时间效率性。

    

    许多统计推断场景，包括贝叶斯网络、马尔可夫过程和隐马尔可夫模型，可以通过将基础的有向无环图（DAG）划分成集群来支持。然而，在统计推断中，最优划分是具有挑战性的，因为要优化的成本取决于集群内的节点以及通过父节点和/或子节点连接的集群之间的映射，我们将其称为依赖集群。我们提出了一种名为DCMAP的新算法，用于具有依赖集群的最优集群映射。在基于DAG和集群映射的任意定义的正成本函数的基础上，我们证明DCMAP收敛于找到所有最优集群，并在途中返回接近最优解。通过实验证明，该算法对使用计算成本函数的一个海草复杂系统的DBN模型具有时间效率性。对于一个25个和50个节点的DBN，搜索空间大小分别为$9.91\times 10^9$和$1.5$

    Many statistical inference contexts, including Bayesian Networks (BNs), Markov processes and Hidden Markov Models (HMMS) could be supported by partitioning (i.e.~mapping) the underlying Directed Acyclic Graph (DAG) into clusters. However, optimal partitioning is challenging, especially in statistical inference as the cost to be optimised is dependent on both nodes within a cluster, and the mapping of clusters connected via parent and/or child nodes, which we call dependent clusters. We propose a novel algorithm called DCMAP for optimal cluster mapping with dependent clusters. Given an arbitrarily defined, positive cost function based on the DAG and cluster mappings, we show that DCMAP converges to find all optimal clusters, and returns near-optimal solutions along the way. Empirically, we find that the algorithm is time-efficient for a DBN model of a seagrass complex system using a computation cost function. For a 25 and 50-node DBN, the search space size was $9.91\times 10^9$ and $1.5
    

