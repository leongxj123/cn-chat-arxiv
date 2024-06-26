# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Best Arm Evades: Near-optimal Multi-pass Streaming Lower Bounds for Pure Exploration in Multi-armed Bandits.](http://arxiv.org/abs/2309.03145) | 该论文通过多通道流算法给出了纯探索多臂赌博机中的近乎最优样本通道交换界限，并回答了一个悬而未决的问题。 |

# 详细

[^1]: 最佳臂躲避：纯探索多臂赌博机中的近乎最优多通道流算法界限

    The Best Arm Evades: Near-optimal Multi-pass Streaming Lower Bounds for Pure Exploration in Multi-armed Bandits. (arXiv:2309.03145v1 [cs.LG])

    [http://arxiv.org/abs/2309.03145](http://arxiv.org/abs/2309.03145)

    该论文通过多通道流算法给出了纯探索多臂赌博机中的近乎最优样本通道交换界限，并回答了一个悬而未决的问题。

    

    我们通过多通道流算法给出了纯探索多臂赌博机（MABs）的近似最优样本通道交换：任何使用子线性内存的流算法，其使用 $O(\frac{n}{\Delta^2})$ 的最优样本复杂度需要 $\Omega(\frac{\log{(1/\Delta)}}{\log\log{(1/\Delta)}})$ 个通道。这里，$n$ 是臂的数量，$\Delta$ 是最佳臂和次佳臂之间的奖励差距。我们的结果与Jin等人[ICML'21]的 $O(\log(\frac{1}{\Delta}))$ 通道算法相匹配（除了低阶项），该算法仅使用 $O(1)$ 内存，并回答了Assadi和Wang[STOC'20]提出的一个悬而未决的问题。

    We give a near-optimal sample-pass trade-off for pure exploration in multi-armed bandits (MABs) via multi-pass streaming algorithms: any streaming algorithm with sublinear memory that uses the optimal sample complexity of $O(\frac{n}{\Delta^2})$ requires $\Omega(\frac{\log{(1/\Delta)}}{\log\log{(1/\Delta)}})$ passes. Here, $n$ is the number of arms and $\Delta$ is the reward gap between the best and the second-best arms. Our result matches the $O(\log(\frac{1}{\Delta}))$-pass algorithm of Jin et al. [ICML'21] (up to lower order terms) that only uses $O(1)$ memory and answers an open question posed by Assadi and Wang [STOC'20].
    

