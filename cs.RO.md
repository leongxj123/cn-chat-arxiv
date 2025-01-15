# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Mixed-Integer Conic Program for the Moving-Target Traveling Salesman Problem based on a Graph of Convex Sets](https://arxiv.org/abs/2403.04917) | 本文提出了一个新的公式，用于解决移动目标旅行推销员问题，该公式基于目标在空间-时间坐标系内成为凸集的概念，通过在凸集图中寻找最短路径来实现，在实验中表现出比当前Mixed Integer Conic Program (MICP)求解器更好的效果。 |

# 详细

[^1]: 基于凸集图的移动目标旅行推销员问题的混合整数锥规划

    A Mixed-Integer Conic Program for the Moving-Target Traveling Salesman Problem based on a Graph of Convex Sets

    [https://arxiv.org/abs/2403.04917](https://arxiv.org/abs/2403.04917)

    本文提出了一个新的公式，用于解决移动目标旅行推销员问题，该公式基于目标在空间-时间坐标系内成为凸集的概念，通过在凸集图中寻找最短路径来实现，在实验中表现出比当前Mixed Integer Conic Program (MICP)求解器更好的效果。

    

    本文介绍了一种寻找移动目标旅行推销员问题（MT-TSP）的最佳解决方案的新的公式，该问题旨在找到一个最短路径，使一个从仓库出发的代理访问一组移动目标，并在它们分配的时间窗口内恰好访问一次，然后返回到仓库。该公式依赖于一个关键思想，即当目标沿着线移动时，它们的轨迹在空间-时间坐标系内变为凸集。然后，问题就缩减为在一个凸集图中寻找最短路径，受到一些速度约束的限制。我们将我们的公式与当前最先进的Mixed Integer Conic Program (MICP)求解器进行了比较，结果显示，我们的公式在目标数量最多为20个的情况下性能优于MICP，在运行时间上缩短了两个数量级，并且最优性差距缩小了高达60％。我们还展示了该解法的成本...

    arXiv:2403.04917v1 Announce Type: cross  Abstract: This paper introduces a new formulation that finds the optimum for the Moving-Target Traveling Salesman Problem (MT-TSP), which seeks to find a shortest path for an agent, that starts at a depot, visits a set of moving targets exactly once within their assigned time-windows, and returns to the depot. The formulation relies on the key idea that when the targets move along lines, their trajectories become convex sets within the space-time coordinate system. The problem then reduces to finding the shortest path within a graph of convex sets, subject to some speed constraints. We compare our formulation with the current state-of-the-art Mixed Integer Conic Program (MICP) solver for the MT-TSP. The experimental results show that our formulation outperforms the MICP for instances with up to 20 targets, with up to two orders of magnitude reduction in runtime, and up to a 60\% tighter optimality gap. We also show that the solution cost from th
    

