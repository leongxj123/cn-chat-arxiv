# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simple Graph Condensation](https://arxiv.org/abs/2403.14951) | 提出了一种简化的图压缩方法，旨在减少图神经网络所带来的不必要复杂性。 |

# 详细

[^1]: 简单图压缩

    Simple Graph Condensation

    [https://arxiv.org/abs/2403.14951](https://arxiv.org/abs/2403.14951)

    提出了一种简化的图压缩方法，旨在减少图神经网络所带来的不必要复杂性。

    

    大规模图上繁重的训练成本已经引起了对图压缩的极大兴趣，涉及调整图神经网络（GNNs）在小尺度压缩图上的训练以在大规模原始图上使用。现有方法主要集中在调整压缩图和原始图之间的关键指标，如梯度、GNNs的分布和轨迹，从而在下游任务上实现了令人满意的性能。然而，这些复杂指标需要复杂的计算，可能会干扰压缩图的优化过程，使得压缩过程非常繁重和不稳定。在各个领域简化模型取得成功的背景下，我们提出了一种简化的图压缩中的指标对准方法，旨在减少从GNNs继承的不必要复杂性。在我们的方法中，我们消除外部参数，仅保留目标的压缩

    arXiv:2403.14951v1 Announce Type: cross  Abstract: The burdensome training costs on large-scale graphs have aroused significant interest in graph condensation, which involves tuning Graph Neural Networks (GNNs) on a small condensed graph for use on the large-scale original graph. Existing methods primarily focus on aligning key metrics between the condensed and original graphs, such as gradients, distribution and trajectory of GNNs, yielding satisfactory performance on downstream tasks. However, these complex metrics necessitate intricate computations and can potentially disrupt the optimization process of the condensation graph, making the condensation process highly demanding and unstable. Motivated by the recent success of simplified models in various fields, we propose a simplified approach to metric alignment in graph condensation, aiming to reduce unnecessary complexity inherited from GNNs. In our approach, we eliminate external parameters and exclusively retain the target conden
    

