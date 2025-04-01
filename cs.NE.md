# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Emergent representations in networks trained with the Forward-Forward algorithm.](http://arxiv.org/abs/2305.18353) | 研究表明使用Forward-Forward算法训练的网络内部表征具有高稀疏度，类别特定的集合，这与生物学观察到的皮层表征相似。 |

# 详细

[^1]: Forward-Forward算法训练的网络中的突现表征

    Emergent representations in networks trained with the Forward-Forward algorithm. (arXiv:2305.18353v1 [cs.NE])

    [http://arxiv.org/abs/2305.18353](http://arxiv.org/abs/2305.18353)

    研究表明使用Forward-Forward算法训练的网络内部表征具有高稀疏度，类别特定的集合，这与生物学观察到的皮层表征相似。

    

    Backpropagation算法被广泛用于训练神经网络，但其缺乏生物学上的现实性。为了寻找一种更具生物学可行性的替代方案，并避免反向传播梯度，而是使用本地学习规则，最近介绍的Forward-Forward算法将Backpropagation的传递替换为两个前向传递。本研究表明，使用Forward-Forward算法获得的内部表征组织为稳健的，类别特定的集合，由极少量的有效单元(高稀疏度)组成。这与感觉处理过程中观察到的皮层表征非常相似。虽然在使用标准Backpropagation进行训练的模型中没有发现，但是在使用与Forward-Forward相同的训练目标进行优化的网络中也出现了稀疏性。这些结果表明，Forward-Forward提议的学习过程可能更接近生物学学习的现实情况。

    The Backpropagation algorithm, widely used to train neural networks, has often been criticised for its lack of biological realism. In an attempt to find a more biologically plausible alternative, and avoid to back-propagate gradients in favour of using local learning rules, the recently introduced Forward-Forward algorithm replaces the traditional forward and backward passes of Backpropagation with two forward passes. In this work, we show that internal representations obtained with the Forward-Forward algorithm organize into robust, category-specific ensembles, composed by an extremely low number of active units (high sparsity). This is remarkably similar to what is observed in cortical representations during sensory processing. While not found in models trained with standard Backpropagation, sparsity emerges also in networks optimized by Backpropagation, on the same training objective of Forward-Forward. These results suggest that the learning procedure proposed by Forward-Forward ma
    

