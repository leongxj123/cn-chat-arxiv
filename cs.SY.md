# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Convex Hull Cheapest Insertion Heuristic for the Non-Euclidean TSP.](http://arxiv.org/abs/2302.06582) | 本文提出了一种适用于非欧几里德旅行商问题的凸包最便宜插入启发式解法，通过使用多维缩放将非欧几里德空间的点近似到欧几里德空间，生成了初始化算法的凸包。在评估中发现，该算法在大多数情况下优于最邻近算法。 |

# 详细

[^1]: 非欧几里德旅行商问题的凸包最便宜插入启发式解法

    A Convex Hull Cheapest Insertion Heuristic for the Non-Euclidean TSP. (arXiv:2302.06582v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.06582](http://arxiv.org/abs/2302.06582)

    本文提出了一种适用于非欧几里德旅行商问题的凸包最便宜插入启发式解法，通过使用多维缩放将非欧几里德空间的点近似到欧几里德空间，生成了初始化算法的凸包。在评估中发现，该算法在大多数情况下优于最邻近算法。

    

    众所周知，凸包最便宜插入启发式算法可以在欧几里德空间中产生良好的旅行商问题解决方案，但还未在非欧几里德情况下进行扩展。为了解决非欧几里德空间中处理障碍物的困难，提出的改进方法使用多维缩放将这些点首先近似到欧几里德空间，从而可以生成初始化算法的凸包。通过修改TSPLIB基准数据集，向其中添加不可通过的分割器来产生非欧几里德空间，评估了所提出的算法。在所研究的案例中，该算法表现出优于常用的最邻近算法的性能，达到96%的情况。

    The convex hull cheapest insertion heuristic is known to generate good solutions to the Traveling Salesperson Problem in Euclidean spaces, but it has not been extended to the non-Euclidean case. To address the difficulty of dealing with obstacles in the non-Euclidean space, the proposed adaptation uses multidimensional scaling to first approximate these points in a Euclidean space, thereby enabling the generation of the convex hull that initializes the algorithm. To evaluate the proposed algorithm, the TSPLIB benchmark data-set is modified by adding impassable separators that produce non-Euclidean spaces. The algorithm is demonstrated to outperform the commonly used Nearest Neighbor algorithm in 96% of the cases studied.
    

