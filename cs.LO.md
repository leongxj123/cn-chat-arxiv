# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Boolformer: Symbolic Regression of Logic Functions with Transformers.](http://arxiv.org/abs/2309.12207) | Boolformer是第一个经过训练的Transformer架构，用于执行端到端的布尔函数符号回归。它可以预测复杂函数的简洁公式，并在提供不完整和有噪声观测时找到近似表达式。Boolformer在真实二分类数据集上展现出潜力作为可解释性替代方案，并在基因调控网络动力学建模任务中与最先进的遗传算法相比表现出竞争力。 |

# 详细

[^1]: Boolformer: 用Transformer进行逻辑函数的符号回归

    Boolformer: Symbolic Regression of Logic Functions with Transformers. (arXiv:2309.12207v1 [cs.LG])

    [http://arxiv.org/abs/2309.12207](http://arxiv.org/abs/2309.12207)

    Boolformer是第一个经过训练的Transformer架构，用于执行端到端的布尔函数符号回归。它可以预测复杂函数的简洁公式，并在提供不完整和有噪声观测时找到近似表达式。Boolformer在真实二分类数据集上展现出潜力作为可解释性替代方案，并在基因调控网络动力学建模任务中与最先进的遗传算法相比表现出竞争力。

    

    在这项工作中，我们介绍了Boolformer，这是第一个经过训练的Transformer架构，用于执行端到端的布尔函数符号回归。首先，我们展示了当提供干净的真值表时，它可以预测复杂函数的简洁公式。然后，我们展示了它在提供不完整和有噪声观测时找到近似表达式的能力。我们在广泛的真实二分类数据集上评估了Boolformer，证明了它作为传统机器学习方法的可解释性替代品的潜力。最后，我们将其应用于建模基因调控网络动力学的常见任务。使用最近的基准测试，我们展示了Boolformer与最先进的遗传算法相比，速度提高了几个数量级。我们的代码和模型公开可用。

    In this work, we introduce Boolformer, the first Transformer architecture trained to perform end-to-end symbolic regression of Boolean functions. First, we show that it can predict compact formulas for complex functions which were not seen during training, when provided a clean truth table. Then, we demonstrate its ability to find approximate expressions when provided incomplete and noisy observations. We evaluate the Boolformer on a broad set of real-world binary classification datasets, demonstrating its potential as an interpretable alternative to classic machine learning methods. Finally, we apply it to the widespread task of modelling the dynamics of gene regulatory networks. Using a recent benchmark, we show that Boolformer is competitive with state-of-the art genetic algorithms with a speedup of several orders of magnitude. Our code and models are available publicly.
    

