# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unexpected Improvements to Expected Improvement for Bayesian Optimization.](http://arxiv.org/abs/2310.20708) | 提出了LogEI作为一类新的贝叶斯优化的获得函数，具有与传统的EI函数相同或近似相等的最优解，但数值上更容易进行优化。 |

# 详细

[^1]: 对贝叶斯优化的期望改进的意外提升

    Unexpected Improvements to Expected Improvement for Bayesian Optimization. (arXiv:2310.20708v1 [cs.LG])

    [http://arxiv.org/abs/2310.20708](http://arxiv.org/abs/2310.20708)

    提出了LogEI作为一类新的贝叶斯优化的获得函数，具有与传统的EI函数相同或近似相等的最优解，但数值上更容易进行优化。

    

    期望改进（EI）可以说是贝叶斯优化中最流行的获得函数，并且已经在很多成功的应用中得到了应用。但是，EI的性能往往被一些新方法超越。尤其是，EI及其变种在并行和多目标设置中很难进行优化，因为它们的获得值在许多区域中数值上变为零。当观测次数增加、搜索空间的维度增加或约束条件的数量增加时，这种困难通常会增加，导致性能在文献中不一致且大多数情况下亚优化。在本论文中，我们提出了LogEI，这是一类新的采样函数。与标准EI相比，这些LogEI函数的成员要么具有相同的最优解，要么具有近似相等的最优解，但数值上更容易进行优化。我们证明了数值病态在“经典”分析EI、期望超体积改进（EHVI）以及它们的...

    Expected Improvement (EI) is arguably the most popular acquisition function in Bayesian optimization and has found countless successful applications, but its performance is often exceeded by that of more recent methods. Notably, EI and its variants, including for the parallel and multi-objective settings, are challenging to optimize because their acquisition values vanish numerically in many regions. This difficulty generally increases as the number of observations, dimensionality of the search space, or the number of constraints grow, resulting in performance that is inconsistent across the literature and most often sub-optimal. Herein, we propose LogEI, a new family of acquisition functions whose members either have identical or approximately equal optima as their canonical counterparts, but are substantially easier to optimize numerically. We demonstrate that numerical pathologies manifest themselves in "classic" analytic EI, Expected Hypervolume Improvement (EHVI), as well as their
    

