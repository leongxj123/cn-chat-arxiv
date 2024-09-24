# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gradient-free neural topology optimization](https://arxiv.org/abs/2403.04937) | 通过提出一种预训练的神经重新参数化策略，在无梯度神经拓扑优化中实现了迭代次数的显著降低，这将开辟一个新的解决路径。 |

# 详细

[^1]: 无梯度神经拓扑优化

    Gradient-free neural topology optimization

    [https://arxiv.org/abs/2403.04937](https://arxiv.org/abs/2403.04937)

    通过提出一种预训练的神经重新参数化策略，在无梯度神经拓扑优化中实现了迭代次数的显著降低，这将开辟一个新的解决路径。

    

    无梯度优化器可以解决问题，无论其目标函数的平滑性或可微性如何，但与基于梯度的算法相比，它们需要更多的迭代才能收敛。这使得它们在拓扑优化中不可行，因为每次迭代的计算成本高，并且问题的维度也很高。我们提出了一种预训练的神经重新参数化策略，当在潜在空间优化设计时，迭代次数至少减少一个数量级，与传统方法不使用潜在重新参数化相比。我们通过对训练数据进行广泛的计算实验，在内部和外部分布中证明了这一点。尽管基于梯度的拓扑优化对于可微的问题，例如结构的合规性优化，仍然更有效，但我们相信这项工作将为那些需要无梯度方法的问题开辟新的道路。

    arXiv:2403.04937v1 Announce Type: new  Abstract: Gradient-free optimizers allow for tackling problems regardless of the smoothness or differentiability of their objective function, but they require many more iterations to converge when compared to gradient-based algorithms. This has made them unviable for topology optimization due to the high computational cost per iteration and high dimensionality of these problems. We propose a pre-trained neural reparameterization strategy that leads to at least one order of magnitude decrease in iteration count when optimizing the designs in latent space, as opposed to the conventional approach without latent reparameterization. We demonstrate this via extensive computational experiments in- and out-of-distribution with the training data. Although gradient-based topology optimization is still more efficient for differentiable problems, such as compliance optimization of structures, we believe this work will open up a new path for problems where gra
    

