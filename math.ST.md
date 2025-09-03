# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jump detection in high-frequency order prices](https://arxiv.org/abs/2403.00819) | 该论文的主要贡献是提出了一种全局跳跃检测方法，利用本地次序统计量检测、定位和估计高频次序价格中的价格跳跃。 |
| [^2] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |

# 详细

[^1]: 高频次序价格中的跳跃检测

    Jump detection in high-frequency order prices

    [https://arxiv.org/abs/2403.00819](https://arxiv.org/abs/2403.00819)

    该论文的主要贡献是提出了一种全局跳跃检测方法，利用本地次序统计量检测、定位和估计高频次序价格中的价格跳跃。

    

    我们提出了一种推断半-martingale跳跃的方法，该方法基于离散、嘈杂、高频次序的观察描述了长期价格动态。与经典的加性、中心的市场微观结构噪声模型不同，我们考虑了限价单委托价格的单侧微观结构噪声。我们开发了使用本地次序统计量估计、定位和检验跳跃的方法。我们提供了一种本地检验，并展示我们可以一致地估计价格跳跃。主要贡献是全局跳跃检验。我们建立了该检验的渐近性质和最优性。基于极值理论，我们推导了无跳跃的零假设下最大统计量的渐近分布。我们证明了在备择假设下的一致性。我们确定并展示了本地备择情况下的收敛速度远远快于标准市场微观结构噪声模型的最优速度。

    arXiv:2403.00819v1 Announce Type: new  Abstract: We propose methods to infer jumps of a semi-martingale, which describes long-term price dynamics based on discrete, noisy, high-frequency observations. Different to the classical model of additive, centered market microstructure noise, we consider one-sided microstructure noise for order prices in a limit order book. We develop methods to estimate, locate and test for jumps using local order statistics. We provide a local test and show that we can consistently estimate price jumps. The main contribution is a global test for jumps. We establish the asymptotic properties and optimality of this test. We derive the asymptotic distribution of a maximum statistic under the null hypothesis of no jumps based on extreme value theory. We prove consistency under the alternative hypothesis. The rate of convergence for local alternatives is determined and shown to be much faster than optimal rates for the standard market microstructure noise model. T
    
[^2]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    

