# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interaction Screening and Pseudolikelihood Approaches for Tensor Learning in Ising Models.](http://arxiv.org/abs/2310.13232) | 本文研究了在Ising模型中的张量学习中，通过伪似然方法和相互作用筛选方法可以恢复出底层的超网络结构，并且性能比较表明张量恢复速率与最大耦合强度呈指数关系。 |
| [^2] | [Adaptive, Rate-Optimal Hypothesis Testing in Nonparametric IV Models.](http://arxiv.org/abs/2006.09587) | 我们提出了一种自适应检验方法，用于处理非参数仪器变量模型中的结构函数的不等式和等式限制。该方法可以适应未知的平滑度和工具强度，并达到了最小值率的自适应最优检验率。 |

# 详细

[^1]: Ising模型中的张量学习的相互作用筛选和伪似然方法

    Interaction Screening and Pseudolikelihood Approaches for Tensor Learning in Ising Models. (arXiv:2310.13232v1 [stat.ME])

    [http://arxiv.org/abs/2310.13232](http://arxiv.org/abs/2310.13232)

    本文研究了在Ising模型中的张量学习中，通过伪似然方法和相互作用筛选方法可以恢复出底层的超网络结构，并且性能比较表明张量恢复速率与最大耦合强度呈指数关系。

    

    本文研究了在$k$-spin Ising模型中的张量恢复中，伪似然方法和相互作用筛选方法两种已知的Ising结构学习方法。我们证明，在适当的正则化下，这两种方法可以使用样本数对数级别大小的样本恢复出底层的超网络结构，且与最大相互作用强度和最大节点度指数级依赖。我们还对这两种方法的张量恢复速率与交互阶数$k$的确切关系进行了跟踪，并允许$k$随样本数和节点数增长。最后，我们通过仿真研究对这两种方法的性能进行了比较讨论，结果也显示了张量恢复速率与最大耦合强度之间的指数依赖关系。

    In this paper, we study two well known methods of Ising structure learning, namely the pseudolikelihood approach and the interaction screening approach, in the context of tensor recovery in $k$-spin Ising models. We show that both these approaches, with proper regularization, retrieve the underlying hypernetwork structure using a sample size logarithmic in the number of network nodes, and exponential in the maximum interaction strength and maximum node-degree. We also track down the exact dependence of the rate of tensor recovery on the interaction order $k$, that is allowed to grow with the number of samples and nodes, for both the approaches. Finally, we provide a comparative discussion of the performance of the two approaches based on simulation studies, which also demonstrate the exponential dependence of the tensor recovery rate on the maximum coupling strength.
    
[^2]: 非参数IV模型中的自适应高效假设检验

    Adaptive, Rate-Optimal Hypothesis Testing in Nonparametric IV Models. (arXiv:2006.09587v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2006.09587](http://arxiv.org/abs/2006.09587)

    我们提出了一种自适应检验方法，用于处理非参数仪器变量模型中的结构函数的不等式和等式限制。该方法可以适应未知的平滑度和工具强度，并达到了最小值率的自适应最优检验率。

    

    我们提出了一种新的自适应假设检验方法，用于非参数仪器变量（NPIV）模型中结构函数的不等式（如单调性、凸性）和等式（如参数、半参数）限制。我们的检验统计量基于修改版的留一法样本模拟，计算受限和不受限筛子NPIV估计量间的二次距离。我们提供了计算简单、数据驱动的筛子调参和Bonferroni调整卡方临界值的选择。我们的检验适应未知的内生性平滑度和工具强度，达到了$L^2$最小值率的自适应最优检验率。也就是说，在复合零假设下其类型I误差的总体和其类型II误差的总体均不能被任何其他NPIV模型的假设检验所提高。我们还提出了基于数据的置信区间。

    We propose a new adaptive hypothesis test for inequality (e.g., monotonicity, convexity) and equality (e.g., parametric, semiparametric) restrictions on a structural function in a nonparametric instrumental variables (NPIV) model. Our test statistic is based on a modified leave-one-out sample analog of a quadratic distance between the restricted and unrestricted sieve NPIV estimators. We provide computationally simple, data-driven choices of sieve tuning parameters and Bonferroni adjusted chi-squared critical values. Our test adapts to the unknown smoothness of alternative functions in the presence of unknown degree of endogeneity and unknown strength of the instruments. It attains the adaptive minimax rate of testing in $L^2$.  That is, the sum of its type I error uniformly over the composite null and its type II error uniformly over nonparametric alternative models cannot be improved by any other hypothesis test for NPIV models of unknown regularities. Data-driven confidence sets in 
    

