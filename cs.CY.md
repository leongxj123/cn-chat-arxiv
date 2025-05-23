# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FairSIN: Achieving Fairness in Graph Neural Networks through Sensitive Information Neutralization](https://arxiv.org/abs/2403.12474) | 通过引入促进公平性的特征（F3）来中和图神经网络中的敏感偏见，进而提高预测性能和公平性的权衡。 |

# 详细

[^1]: FairSIN：通过敏感信息中和实现图神经网络中的公平性

    FairSIN: Achieving Fairness in Graph Neural Networks through Sensitive Information Neutralization

    [https://arxiv.org/abs/2403.12474](https://arxiv.org/abs/2403.12474)

    通过引入促进公平性的特征（F3）来中和图神经网络中的敏感偏见，进而提高预测性能和公平性的权衡。

    

    尽管图神经网络（GNNs）在对图结构数据进行建模方面取得了显著成功，但与其他机器学习模型一样，GNNs也容易根据敏感属性（如种族和性别）做出有偏见的预测。为了公平考虑，最近的最先进方法提出从输入或表示中过滤掉敏感信息，例如删除边或屏蔽特征。然而，我们认为基于此类过滤策略可能也会过滤掉一些非敏感的特征信息，导致在预测性能和公平性之间产生次优的权衡。为解决这一问题，我们提出一种创新的中和基础范式，即在信息传递之前将额外的促进公平性的特征（F3）纳入节点特征或表示中。这些F3预期在统计上中和节点表示中的敏感偏见，并提供额外的非敏感信息。

    arXiv:2403.12474v1 Announce Type: new  Abstract: Despite the remarkable success of graph neural networks (GNNs) in modeling graph-structured data, like other machine learning models, GNNs are also susceptible to making biased predictions based on sensitive attributes, such as race and gender. For fairness consideration, recent state-of-the-art (SOTA) methods propose to filter out sensitive information from inputs or representations, e.g., edge dropping or feature masking. However, we argue that such filtering-based strategies may also filter out some non-sensitive feature information, leading to a sub-optimal trade-off between predictive performance and fairness. To address this issue, we unveil an innovative neutralization-based paradigm, where additional Fairness-facilitating Features (F3) are incorporated into node features or representations before message passing. The F3 are expected to statistically neutralize the sensitive bias in node representations and provide additional nons
    

