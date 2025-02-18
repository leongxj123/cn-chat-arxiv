# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^2] | [On the Efficiency of Finely Stratified Experiments.](http://arxiv.org/abs/2307.15181) | 本文研究了在实验分析中涉及的大类处理效应参数的有效估计，包括平均处理效应、分位数处理效应、局部平均处理效应等。 |
| [^3] | [Manifold Learning with Sparse Regularised Optimal Transport.](http://arxiv.org/abs/2307.09816) | 这篇论文介绍了一种利用稀疏正则最优传输进行流形学习的方法，该方法构建了一个稀疏自适应的亲和矩阵，并在连续极限下与拉普拉斯型算子一致。 |

# 详细

[^1]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^2]: 关于细分实验效率的研究

    On the Efficiency of Finely Stratified Experiments. (arXiv:2307.15181v1 [econ.EM])

    [http://arxiv.org/abs/2307.15181](http://arxiv.org/abs/2307.15181)

    本文研究了在实验分析中涉及的大类处理效应参数的有效估计，包括平均处理效应、分位数处理效应、局部平均处理效应等。

    

    本文研究了在实验分析中涉及的大类处理效应参数的有效估计。在这里，效率是指对于一类广泛的处理分配方案而言的，其中任何单位被分配到处理的边际概率等于预先指定的值，例如一半。重要的是，我们不要求处理状态是以i.i.d.的方式分配的，因此可以适应实践中使用的复杂处理分配方案，如分层随机化和匹配对。所考虑的参数类别是可以表示为已知观测数据的一个已知函数的期望的约束的解的那些参数，其中可能包括处理分配边际概率的预先指定值。我们证明了这类参数包括平均处理效应、分位数处理效应、局部平均处理效应等。

    This paper studies the efficient estimation of a large class of treatment effect parameters that arise in the analysis of experiments. Here, efficiency is understood to be with respect to a broad class of treatment assignment schemes for which the marginal probability that any unit is assigned to treatment equals a pre-specified value, e.g., one half. Importantly, we do not require that treatment status is assigned in an i.i.d. fashion, thereby accommodating complicated treatment assignment schemes that are used in practice, such as stratified block randomization and matched pairs. The class of parameters considered are those that can be expressed as the solution to a restriction on the expectation of a known function of the observed data, including possibly the pre-specified value for the marginal probability of treatment assignment. We show that this class of parameters includes, among other things, average treatment effects, quantile treatment effects, local average treatment effect
    
[^3]: 用稀疏正则最优传输进行流形学习

    Manifold Learning with Sparse Regularised Optimal Transport. (arXiv:2307.09816v1 [stat.ML])

    [http://arxiv.org/abs/2307.09816](http://arxiv.org/abs/2307.09816)

    这篇论文介绍了一种利用稀疏正则最优传输进行流形学习的方法，该方法构建了一个稀疏自适应的亲和矩阵，并在连续极限下与拉普拉斯型算子一致。

    

    流形学习是现代统计学和数据科学中的一个核心任务。许多数据集（细胞、文档、图像、分子）可以被表示为嵌入在高维环境空间中的点云，然而数据固有的自由度通常远远少于环境维度的数量。检测数据嵌入的潜在流形是许多下游分析的先决条件。现实世界的数据集经常受到噪声观测和抽样的影响，因此提取关于潜在流形的信息是一个重大挑战。我们提出了一种利用对称版本的最优传输和二次正则化的流形学习方法，它构建了一个稀疏自适应的亲和矩阵，可以解释为双随机核归一化的推广。我们证明了在连续极限下产生的核与拉普拉斯型算子一致，并建立了该方法的健壮性。

    Manifold learning is a central task in modern statistics and data science. Many datasets (cells, documents, images, molecules) can be represented as point clouds embedded in a high dimensional ambient space, however the degrees of freedom intrinsic to the data are usually far fewer than the number of ambient dimensions. The task of detecting a latent manifold along which the data are embedded is a prerequisite for a wide family of downstream analyses. Real-world datasets are subject to noisy observations and sampling, so that distilling information about the underlying manifold is a major challenge. We propose a method for manifold learning that utilises a symmetric version of optimal transport with a quadratic regularisation that constructs a sparse and adaptive affinity matrix, that can be interpreted as a generalisation of the bistochastic kernel normalisation. We prove that the resulting kernel is consistent with a Laplace-type operator in the continuous limit, establish robustness
    

