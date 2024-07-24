# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target](https://arxiv.org/abs/2403.12116) | 本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。 |

# 详细

[^1]: 基于自定义生物启发目标的无监督端到端训练

    Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target

    [https://arxiv.org/abs/2403.12116](https://arxiv.org/abs/2403.12116)

    本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。

    

    当前的无监督学习方法依赖于通过深度学习技术（如自监督学习）进行端到端训练，具有较高的计算需求，或者采用通过类似Hebbian学习的生物启发方法逐层训练，使用与监督学习不兼容的局部学习规则。为了解决这一挑战，在这项工作中，我们引入了一种使用网络最终层的胜者通吃（WTA）选择性的“自定义目标”，并通过生物启发的稳态机制进行正则化。

    arXiv:2403.12116v1 Announce Type: cross  Abstract: Current unsupervised learning methods depend on end-to-end training via deep learning techniques such as self-supervised learning, with high computational requirements, or employ layer-by-layer training using bio-inspired approaches like Hebbian learning, using local learning rules incompatible with supervised learning. Both approaches are problematic for edge AI hardware that relies on sparse computational resources and would strongly benefit from alternating between unsupervised and supervised learning phases - thus leveraging widely available unlabeled data from the environment as well as labeled training datasets. To solve this challenge, in this work, we introduce a 'self-defined target' that uses Winner-Take-All (WTA) selectivity at the network's final layer, complemented by regularization through biologically inspired homeostasis mechanism. This approach, framework-agnostic and compatible with both global (Backpropagation) and l
    

