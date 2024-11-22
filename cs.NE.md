# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target](https://arxiv.org/abs/2403.12116) | 本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。 |
| [^2] | [Criticality-Guided Efficient Pruning in Spiking Neural Networks Inspired by Critical Brain Hypothesis.](http://arxiv.org/abs/2311.16141) | 本研究受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的高效SNN修剪方法，以加强特征提取和加速修剪过程，并取得了比当前最先进方法更好的性能。 |

# 详细

[^1]: 基于自定义生物启发目标的无监督端到端训练

    Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target

    [https://arxiv.org/abs/2403.12116](https://arxiv.org/abs/2403.12116)

    本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。

    

    当前的无监督学习方法依赖于通过深度学习技术（如自监督学习）进行端到端训练，具有较高的计算需求，或者采用通过类似Hebbian学习的生物启发方法逐层训练，使用与监督学习不兼容的局部学习规则。为了解决这一挑战，在这项工作中，我们引入了一种使用网络最终层的胜者通吃（WTA）选择性的“自定义目标”，并通过生物启发的稳态机制进行正则化。

    arXiv:2403.12116v1 Announce Type: cross  Abstract: Current unsupervised learning methods depend on end-to-end training via deep learning techniques such as self-supervised learning, with high computational requirements, or employ layer-by-layer training using bio-inspired approaches like Hebbian learning, using local learning rules incompatible with supervised learning. Both approaches are problematic for edge AI hardware that relies on sparse computational resources and would strongly benefit from alternating between unsupervised and supervised learning phases - thus leveraging widely available unlabeled data from the environment as well as labeled training datasets. To solve this challenge, in this work, we introduce a 'self-defined target' that uses Winner-Take-All (WTA) selectivity at the network's final layer, complemented by regularization through biologically inspired homeostasis mechanism. This approach, framework-agnostic and compatible with both global (Backpropagation) and l
    
[^2]: SNNs中基于关键性的高效修剪方法，受到关键性大脑假设的启发

    Criticality-Guided Efficient Pruning in Spiking Neural Networks Inspired by Critical Brain Hypothesis. (arXiv:2311.16141v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2311.16141](http://arxiv.org/abs/2311.16141)

    本研究受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的高效SNN修剪方法，以加强特征提取和加速修剪过程，并取得了比当前最先进方法更好的性能。

    

    由于其节能和无乘法特性，SNNs已经引起了相当大的关注。深度SNNs规模的不断增长给模型部署带来了挑战。网络修剪通过压缩网络规模来减少模型部署的硬件资源需求。然而，现有的SNN修剪方法由于修剪迭代增加了SNNs的训练难度，导致修剪成本高昂且性能损失严重。本文受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的用于SNN修剪的再生机制，以增强特征提取并加速修剪过程。首先，我们提出了一种SNN中用于关键性的低成本度量方式。然后，在修剪后对所修剪结构进行重新排序，并再生那些具有较高关键性的结构，以获取关键网络。我们的方法表现优于当前的最先进方法。

    Spiking Neural Networks (SNNs) have gained considerable attention due to the energy-efficient and multiplication-free characteristics. The continuous growth in scale of deep SNNs poses challenges for model deployment. Network pruning reduces hardware resource requirements of model deployment by compressing the network scale. However, existing SNN pruning methods cause high pruning costs and performance loss because the pruning iterations amplify the training difficulty of SNNs. In this paper, inspired by the critical brain hypothesis in neuroscience, we propose a regeneration mechanism based on the neuron criticality for SNN pruning to enhance feature extraction and accelerate the pruning process. Firstly, we propose a low-cost metric for the criticality in SNNs. Then, we re-rank the pruned structures after pruning and regenerate those with higher criticality to obtain the critical network. Our method achieves higher performance than the current state-of-the-art (SOTA) method with up t
    

