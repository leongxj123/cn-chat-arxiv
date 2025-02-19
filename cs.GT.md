# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self-Resolving Prediction Markets for Unverifiable Outcomes.](http://arxiv.org/abs/2306.04305) | 该论文提出了一种新颖的预测市场机制，允许在无法验证结果的情况下从一组代理中收集信息和汇总预测，使用机器学习模型，实现自治以及避免了需进一步的验证或干预。 |

# 详细

[^1]: 无法验证结果的自动解决预测市场

    Self-Resolving Prediction Markets for Unverifiable Outcomes. (arXiv:2306.04305v1 [cs.GT])

    [http://arxiv.org/abs/2306.04305](http://arxiv.org/abs/2306.04305)

    该论文提出了一种新颖的预测市场机制，允许在无法验证结果的情况下从一组代理中收集信息和汇总预测，使用机器学习模型，实现自治以及避免了需进一步的验证或干预。

    

    预测市场通过根据预测是否接近可验证的未来结果向代理支付费用来激励和汇总信念。然而，许多重要问题的结果难以验证或不可验证，因为可能很难或不可能获取实际情况。我们提出了一种新颖而又不直观的结果，表明可以通过向代理支付其预测与精心选择的参考代理的预测之间的负交叉熵来运行一个ε-激励兼容的预测市场，从代理池中获取信息并进行有效的汇总，而不需要观察结果。我们的机制利用一个离线的机器学习模型，该模型根据市场设计者已知的一组特征来预测结果，从而使市场能够在观察到结果后自行解决并立即向代理支付报酬，而不需要进一步的验证或干预。我们对我们的机制的效率、激励兼容性和收敛性提供了理论保证，同时在几个真实世界的数据集上进行了验证。

    Prediction markets elicit and aggregate beliefs by paying agents based on how close their predictions are to a verifiable future outcome. However, outcomes of many important questions are difficult to verify or unverifiable, in that the ground truth may be hard or impossible to access. Examples include questions about causal effects where it is infeasible or unethical to run randomized trials; crowdsourcing and content moderation tasks where it is prohibitively expensive to verify ground truth; and questions asked over long time horizons, where the delay until the realization of the outcome skews agents' incentives to report their true beliefs. We present a novel and unintuitive result showing that it is possible to run an $\varepsilon-$incentive compatible prediction market to elicit and efficiently aggregate information from a pool of agents without observing the outcome by paying agents the negative cross-entropy between their prediction and that of a carefully chosen reference agen
    

