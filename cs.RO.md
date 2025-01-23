# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast Ergodic Search with Kernel Functions](https://arxiv.org/abs/2403.01536) | 提出了一种使用核函数的快速遍历搜索方法，其在搜索空间维度上具有线性复杂度，可以推广到李群，并且通过数值测试展示比现有算法快两个数量级。 |

# 详细

[^1]: 使用核函数的快速遍历搜索

    Fast Ergodic Search with Kernel Functions

    [https://arxiv.org/abs/2403.01536](https://arxiv.org/abs/2403.01536)

    提出了一种使用核函数的快速遍历搜索方法，其在搜索空间维度上具有线性复杂度，可以推广到李群，并且通过数值测试展示比现有算法快两个数量级。

    

    遍历搜索使得对信息分布进行最佳探索成为可能，同时保证了对搜索空间的渐近覆盖。然而，当前的方法通常在搜索空间维度上具有指数计算复杂度，并且局限于欧几里得空间。我们引入了一种计算高效的遍历搜索方法。我们的贡献是双重的。首先，我们开发了基于核的遍历度量，并将其从欧几里得空间推广到李群上。我们正式证明了所建议的度量与标准遍历度量一致，同时保证了在搜索空间维度上具有线性复杂度。其次，我们推导了非线性系统的核遍历度量的一阶最优性条件，这使得轨迹优化变得更加高效。全面的数值基准测试表明，所提出的方法至少比现有最先进的算法快两个数量级。

    arXiv:2403.01536v1 Announce Type: cross  Abstract: Ergodic search enables optimal exploration of an information distribution while guaranteeing the asymptotic coverage of the search space. However, current methods typically have exponential computation complexity in the search space dimension and are restricted to Euclidean space. We introduce a computationally efficient ergodic search method. Our contributions are two-fold. First, we develop a kernel-based ergodic metric and generalize it from Euclidean space to Lie groups. We formally prove the proposed metric is consistent with the standard ergodic metric while guaranteeing linear complexity in the search space dimension. Secondly, we derive the first-order optimality condition of the kernel ergodic metric for nonlinear systems, which enables efficient trajectory optimization. Comprehensive numerical benchmarks show that the proposed method is at least two orders of magnitude faster than the state-of-the-art algorithm. Finally, we d
    

