# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlexFringe: Modeling Software Behavior by Learning Probabilistic Automata.](http://arxiv.org/abs/2203.16331) | FlexFringe提供了高效的概率有限自动机学习方法，可用于建模软件行为。该方法在实践中通过实现改进的状态合并策略实现了显著性能提升，并且能够从软件日志中学习可解释的模型，用于异常检测。与基于神经网络的解决方案相比，学习更小更复杂的模型能够提高FlexFringe在异常检测中的性能。 |

# 详细

[^1]: FlexFringe:通过学习概率有限自动机来建模软件行为

    FlexFringe: Modeling Software Behavior by Learning Probabilistic Automata. (arXiv:2203.16331v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.16331](http://arxiv.org/abs/2203.16331)

    FlexFringe提供了高效的概率有限自动机学习方法，可用于建模软件行为。该方法在实践中通过实现改进的状态合并策略实现了显著性能提升，并且能够从软件日志中学习可解释的模型，用于异常检测。与基于神经网络的解决方案相比，学习更小更复杂的模型能够提高FlexFringe在异常检测中的性能。

    

    我们介绍了FlexFringe中可用的概率确定性有限自动机学习方法的高效实现。这些实现了众所周知的状态合并策略，包括几种修改以提高它们在实践中的性能。我们通过实验证明这些算法能够获得有竞争力的结果，并在默认实现上实现了显著的改进。我们还展示了如何使用FlexFringe从软件日志中学习可解释的模型，并将其用于异常检测。虽然这些模型较难解释，但我们展示了学习更小、更复杂的模型如何提高FlexFringe在异常检测中的性能，优于基于神经网络的现有解决方案。

    We present the efficient implementations of probabilistic deterministic finite automaton learning methods available in FlexFringe. These implement well-known strategies for state-merging including several modifications to improve their performance in practice. We show experimentally that these algorithms obtain competitive results and significant improvements over a default implementation. We also demonstrate how to use FlexFringe to learn interpretable models from software logs and use these for anomaly detection. Although less interpretable, we show that learning smaller more convoluted models improves the performance of FlexFringe on anomaly detection, outperforming an existing solution based on neural nets.
    

