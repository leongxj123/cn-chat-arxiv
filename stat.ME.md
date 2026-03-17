# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^2] | [Skeleton Regression: A Graph-Based Approach to Estimation with Manifold Structure.](http://arxiv.org/abs/2303.11786) | 这是一个处理低维流形数据的回归框架，首先通过构建图形骨架来捕捉潜在的流形几何结构，然后在其上运用非参数回归技术来估计回归函数，除了具有非参数优点之外，在处理多个流形数据，嘈杂观察时也表现出较好的鲁棒性。 |

# 详细

[^1]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^2]: Skeleton Regression：一种基于流形结构估计的基于图形的方法。

    Skeleton Regression: A Graph-Based Approach to Estimation with Manifold Structure. (arXiv:2303.11786v1 [cs.LG])

    [http://arxiv.org/abs/2303.11786](http://arxiv.org/abs/2303.11786)

    这是一个处理低维流形数据的回归框架，首先通过构建图形骨架来捕捉潜在的流形几何结构，然后在其上运用非参数回归技术来估计回归函数，除了具有非参数优点之外，在处理多个流形数据，嘈杂观察时也表现出较好的鲁棒性。

    

    我们引入了一个新的回归框架，旨在处理围绕低维流形的复杂数据。我们的方法首先构建一个图形表示，称为骨架，以捕获潜在的几何结构。然后，我们在骨架图上定义指标，应用非参数回归技术，以及基于图形的特征转换来估计回归函数。除了包括的非参数方法外，我们还讨论了一些非参数回归器在骨架图等一般度量空间方面的限制。所提出的回归框架使我们能够避开维度灾难，具有可以处理多个流形的并集并且鲁棒性能应对加性噪声和嘈杂观察的额外优势。我们为所提出的方法提供了统计保证，并通过模拟和实际数据示例证明了其有效性。

    We introduce a new regression framework designed to deal with large-scale, complex data that lies around a low-dimensional manifold. Our approach first constructs a graph representation, referred to as the skeleton, to capture the underlying geometric structure. We then define metrics on the skeleton graph and apply nonparametric regression techniques, along with feature transformations based on the graph, to estimate the regression function. In addition to the included nonparametric methods, we also discuss the limitations of some nonparametric regressors with respect to the general metric space such as the skeleton graph. The proposed regression framework allows us to bypass the curse of dimensionality and provides additional advantages that it can handle the union of multiple manifolds and is robust to additive noise and noisy observations. We provide statistical guarantees for the proposed method and demonstrate its effectiveness through simulations and real data examples.
    

