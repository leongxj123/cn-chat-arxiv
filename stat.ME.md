# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A comprehensive study of spike and slab shrinkage priors for structurally sparse Bayesian neural networks.](http://arxiv.org/abs/2308.09104) | 本论文研究了在贝叶斯神经网络中使用Lasso和Horseshoe两种缩减技术进行模型压缩的方法。为了实现结构稀疏，通过提出尖峰与块组稀疏Lasso和尖峰与块组Horseshoe先验，并开发了可计算的变分推断方法。该方法可以在保持推理效率的同时实现深度神经网络的模型压缩。 |

# 详细

[^1]: 基于尖峰与块缩减先验的结构稀疏贝叶斯神经网络的全面研究

    A comprehensive study of spike and slab shrinkage priors for structurally sparse Bayesian neural networks. (arXiv:2308.09104v1 [stat.ML])

    [http://arxiv.org/abs/2308.09104](http://arxiv.org/abs/2308.09104)

    本论文研究了在贝叶斯神经网络中使用Lasso和Horseshoe两种缩减技术进行模型压缩的方法。为了实现结构稀疏，通过提出尖峰与块组稀疏Lasso和尖峰与块组Horseshoe先验，并开发了可计算的变分推断方法。该方法可以在保持推理效率的同时实现深度神经网络的模型压缩。

    

    网络复杂度和计算效率已经成为深度学习中越来越重要的方面。稀疏深度学习通过减少过参数化的深度神经网络来恢复底层目标函数的稀疏表示，解决了这些挑战。具体而言，通过结构稀疏（如节点稀疏）压缩的深度神经架构提供了低延迟推理、更高的数据吞吐量和更低的能量消耗。在本文中，我们研究了两种广泛应用的缩减技术，Lasso和Horseshoe，在贝叶斯神经网络中进行模型压缩。为此，我们提出了基于尖峰与块组稀疏Lasso (SS-GL)和基于尖峰与块组Horseshoe (SS-GHS)先验的结构稀疏贝叶斯神经网络，并开发了可计算的变分推断，包括对伯努利变量的连续松弛。我们确定了变分推断的收缩速率。

    Network complexity and computational efficiency have become increasingly significant aspects of deep learning. Sparse deep learning addresses these challenges by recovering a sparse representation of the underlying target function by reducing heavily over-parameterized deep neural networks. Specifically, deep neural architectures compressed via structured sparsity (e.g. node sparsity) provide low latency inference, higher data throughput, and reduced energy consumption. In this paper, we explore two well-established shrinkage techniques, Lasso and Horseshoe, for model compression in Bayesian neural networks. To this end, we propose structurally sparse Bayesian neural networks which systematically prune excessive nodes with (i) Spike-and-Slab Group Lasso (SS-GL), and (ii) Spike-and-Slab Group Horseshoe (SS-GHS) priors, and develop computationally tractable variational inference including continuous relaxation of Bernoulli variables. We establish the contraction rates of the variational 
    

