# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Castor: Causal Temporal Regime Structure Learning.](http://arxiv.org/abs/2311.01412) | “Castor”是一个用于学习多元时间序列数据中因果关系的框架，能够综合学习各个区域的因果图。它通过最大化得分函数来推断区域的数量，并学习每个区域中的线性或非线性因果关系。 |

# 详细

[^1]: Castor: 因果时序区域结构学习

    Castor: Causal Temporal Regime Structure Learning. (arXiv:2311.01412v1 [cs.LG])

    [http://arxiv.org/abs/2311.01412](http://arxiv.org/abs/2311.01412)

    “Castor”是一个用于学习多元时间序列数据中因果关系的框架，能够综合学习各个区域的因果图。它通过最大化得分函数来推断区域的数量，并学习每个区域中的线性或非线性因果关系。

    

    揭示多元时间序列数据之间的因果关系是一个重要且具有挑战性的目标，涉及到从气候科学到医疗保健等各个学科的广泛范围。这些数据包含线性或非线性关系，并且通常遵循多个先验未知的区域。现有的因果发现方法可以从具有已知区域的异构数据中推断出摘要因果图，但在全面学习区域和相应的因果图方面存在不足。在本文中，我们介绍了CASTOR，这是一个新颖的框架，旨在学习由不同因果图统治的各种异构时间序列数据中的因果关系。通过EM算法通过最大化一个得分函数，CASTOR推断出区域的数量并学习每个区域中的线性或非线性因果关系。我们展示了CASTOR的稳健收敛性质，特别突出了其有效性。

    The task of uncovering causal relationships among multivariate time series data stands as an essential and challenging objective that cuts across a broad array of disciplines ranging from climate science to healthcare. Such data entails linear or non-linear relationships, and usually follow multiple a priori unknown regimes. Existing causal discovery methods can infer summary causal graphs from heterogeneous data with known regimes, but they fall short in comprehensively learning both regimes and the corresponding causal graph. In this paper, we introduce CASTOR, a novel framework designed to learn causal relationships in heterogeneous time series data composed of various regimes, each governed by a distinct causal graph. Through the maximization of a score function via the EM algorithm, CASTOR infers the number of regimes and learns linear or non-linear causal relationships in each regime. We demonstrate the robust convergence properties of CASTOR, specifically highlighting its profic
    

