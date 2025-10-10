# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph-SCP: Accelerating Set Cover Problems with Graph Neural Networks.](http://arxiv.org/abs/2310.07979) | 图形-SCP是一种使用图神经网络加速集合覆盖问题的方法，通过学习识别包含解空间的较小子问题来提高优化求解器的性能，实验结果表明，图形-SCP能够将问题大小减少30-70%，和商业求解器相比加速高达25倍，并且能够在给定的最优性阈值下改进或实现100%的最优性。 |

# 详细

[^1]: 图形-SCP: 用图神经网络加速集合覆盖问题

    Graph-SCP: Accelerating Set Cover Problems with Graph Neural Networks. (arXiv:2310.07979v1 [cs.LG])

    [http://arxiv.org/abs/2310.07979](http://arxiv.org/abs/2310.07979)

    图形-SCP是一种使用图神经网络加速集合覆盖问题的方法，通过学习识别包含解空间的较小子问题来提高优化求解器的性能，实验结果表明，图形-SCP能够将问题大小减少30-70%，和商业求解器相比加速高达25倍，并且能够在给定的最优性阈值下改进或实现100%的最优性。

    

    机器学习方法越来越多地用于加速组合优化问题。我们特别关注集合覆盖问题（SCP），提出了一种名为图形-SCP的图神经网络方法，可以通过学习识别包含解空间的大大较小的子问题来增强现有的优化求解器。我们在具有不同问题特征和复杂度的合成加权和非加权SCP实例上评估了图形-SCP的性能，并在OR Library的实例上进行了评估，这是SCP的一个经典基准。我们展示了图形-SCP将问题大小减少了30-70%，并且相对于商业求解器（Gurobi）实现了高达25倍的运行时间加速。在给定所需的最优性阈值的情况下，图形-SCP将改进或甚至实现100%的最优性。这与快速贪婪解决方案形成了对比，后者在保证多项式运行时间的同时明显损害了解决方案的质量。图形-SCP可以推广到更大的问题规模。

    Machine learning (ML) approaches are increasingly being used to accelerate combinatorial optimization (CO) problems. We look specifically at the Set Cover Problem (SCP) and propose Graph-SCP, a graph neural network method that can augment existing optimization solvers by learning to identify a much smaller sub-problem that contains the solution space. We evaluate the performance of Graph-SCP on synthetic weighted and unweighted SCP instances with diverse problem characteristics and complexities, and on instances from the OR Library, a canonical benchmark for SCP. We show that Graph-SCP reduces the problem size by 30-70% and achieves run time speedups up to~25x when compared to commercial solvers (Gurobi). Given a desired optimality threshold, Graph-SCP will improve upon it or even achieve 100% optimality. This is in contrast to fast greedy solutions that significantly compromise solution quality to achieve guaranteed polynomial run time. Graph-SCP can generalize to larger problem sizes
    

