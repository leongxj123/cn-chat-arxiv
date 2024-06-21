# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Bandit Approach with Evolutionary Operators for Model Selection](https://arxiv.org/abs/2402.05144) | 本文提出了一种使用进化算子的强盗方法来进行模型选择，通过将模型选择问题建模为无穷臂赌博机问题，利用部分训练和准确性作为奖励，最终的算法Mutant-UCB在测试中表现出色，优于固定预算下的最先进技术。 |
| [^2] | [Learning from Emergence: A Study on Proactively Inhibiting the Monosemantic Neurons of Artificial Neural Networks](https://arxiv.org/abs/2312.11560) | 本文研究了积极抑制人工神经网络中的单意义神经元，这对于提高性能具有重要意义，并提出了一种基于自发现的方法来实现抑制。 |
| [^3] | [Emergent representations in networks trained with the Forward-Forward algorithm.](http://arxiv.org/abs/2305.18353) | 研究表明使用Forward-Forward算法训练的网络内部表征具有高稀疏度，类别特定的集合，这与生物学观察到的皮层表征相似。 |

# 详细

[^1]: 一种使用进化算子的强盗方法进行模型选择

    A Bandit Approach with Evolutionary Operators for Model Selection

    [https://arxiv.org/abs/2402.05144](https://arxiv.org/abs/2402.05144)

    本文提出了一种使用进化算子的强盗方法来进行模型选择，通过将模型选择问题建模为无穷臂赌博机问题，利用部分训练和准确性作为奖励，最终的算法Mutant-UCB在测试中表现出色，优于固定预算下的最先进技术。

    

    本文将模型选择问题建模为无穷臂赌博机问题。模型是臂，选择一个臂对应部分训练模型（资源分配）。奖励是选择模型在部分训练后的准确性。在这个最佳臂识别问题中，遗憾是最优模型的预期准确性与最终选择模型的准确性之间的差距。我们首先考虑了UCB-E在随机无穷臂赌博机问题上的直接推广，并且证明了在基本假设下，期望遗憾的顺序是$T^{-\alpha}$，其中$\alpha \in (0,1/5)$，$T$是要分配的资源数量。从这个基本算法出发，我们介绍了一种算法Mutant-UCB，它结合了进化算法的操作符。在三个开源图片分类数据集上进行的测试表明了这种新颖的组合方法的相关性，该方法优于固定预算下的国际领先技术。

    This paper formulates model selection as an infinite-armed bandit problem. The models are arms, and picking an arm corresponds to a partial training of the model (resource allocation). The reward is the accuracy of the selected model after its partial training. In this best arm identification problem, regret is the gap between the expected accuracy of the optimal model and that of the model finally chosen. We first consider a straightforward generalization of UCB-E to the stochastic infinite-armed bandit problem and show that, under basic assumptions, the expected regret order is $T^{-\alpha}$ for some $\alpha \in (0,1/5)$ and $T$ the number of resources to allocate. From this vanilla algorithm, we introduce the algorithm Mutant-UCB that incorporates operators from evolutionary algorithms. Tests carried out on three open source image classification data sets attest to the relevance of this novel combining approach, which outperforms the state-of-the-art for a fixed budget.
    
[^2]: 学习自发现：关于积极抑制人工神经网络单意义神经元的研究

    Learning from Emergence: A Study on Proactively Inhibiting the Monosemantic Neurons of Artificial Neural Networks

    [https://arxiv.org/abs/2312.11560](https://arxiv.org/abs/2312.11560)

    本文研究了积极抑制人工神经网络中的单意义神经元，这对于提高性能具有重要意义，并提出了一种基于自发现的方法来实现抑制。

    

    最近，随着大型语言模型的成功，自发现受到了研究界的广泛关注。与现有文献不同，我们提出了一个关键因素的假设，即在规模扩大的过程中高度促进性能的因素：减少只能与特定特征形成一对一关系的单意义神经元。单意义神经元往往更稀疏，并对大型模型的性能产生负面影响。受到这一观点的启发，我们提出了一种直观的思路来识别和抑制单意义神经元。然而，实现这一目标是一个非平凡的任务，因为没有统一的定量评估指标，简单地禁止单意义神经元并不能促进神经网络的多意思性。因此，本文提出了从自发现中学习的方法，并展开了关于积极抑制单意义神经元的研究。具体来说，我们首先提出了一种新的方法

    arXiv:2312.11560v2 Announce Type: replace-cross  Abstract: Recently, emergence has received widespread attention from the research community along with the success of large language models. Different from the literature, we hypothesize a key factor that highly promotes the performance during the increase of scale: the reduction of monosemantic neurons that can only form one-to-one correlations with specific features. Monosemantic neurons tend to be sparser and have negative impacts on the performance in large models. Inspired by this insight, we propose an intuitive idea to identify monosemantic neurons and inhibit them. However, achieving this goal is a non-trivial task as there is no unified quantitative evaluation metric and simply banning monosemantic neurons does not promote polysemanticity in neural networks. Therefore, we propose to learn from emergence and present a study on proactively inhibiting the monosemantic neurons in this paper. More specifically, we first propose a new
    
[^3]: Forward-Forward算法训练的网络中的突现表征

    Emergent representations in networks trained with the Forward-Forward algorithm. (arXiv:2305.18353v1 [cs.NE])

    [http://arxiv.org/abs/2305.18353](http://arxiv.org/abs/2305.18353)

    研究表明使用Forward-Forward算法训练的网络内部表征具有高稀疏度，类别特定的集合，这与生物学观察到的皮层表征相似。

    

    Backpropagation算法被广泛用于训练神经网络，但其缺乏生物学上的现实性。为了寻找一种更具生物学可行性的替代方案，并避免反向传播梯度，而是使用本地学习规则，最近介绍的Forward-Forward算法将Backpropagation的传递替换为两个前向传递。本研究表明，使用Forward-Forward算法获得的内部表征组织为稳健的，类别特定的集合，由极少量的有效单元(高稀疏度)组成。这与感觉处理过程中观察到的皮层表征非常相似。虽然在使用标准Backpropagation进行训练的模型中没有发现，但是在使用与Forward-Forward相同的训练目标进行优化的网络中也出现了稀疏性。这些结果表明，Forward-Forward提议的学习过程可能更接近生物学学习的现实情况。

    The Backpropagation algorithm, widely used to train neural networks, has often been criticised for its lack of biological realism. In an attempt to find a more biologically plausible alternative, and avoid to back-propagate gradients in favour of using local learning rules, the recently introduced Forward-Forward algorithm replaces the traditional forward and backward passes of Backpropagation with two forward passes. In this work, we show that internal representations obtained with the Forward-Forward algorithm organize into robust, category-specific ensembles, composed by an extremely low number of active units (high sparsity). This is remarkably similar to what is observed in cortical representations during sensory processing. While not found in models trained with standard Backpropagation, sparsity emerges also in networks optimized by Backpropagation, on the same training objective of Forward-Forward. These results suggest that the learning procedure proposed by Forward-Forward ma
    

