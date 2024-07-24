# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target](https://arxiv.org/abs/2403.12116) | 本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。 |
| [^2] | [Smooth Tchebycheff Scalarization for Multi-Objective Optimization](https://arxiv.org/abs/2402.19078) | 通过光滑 Tchebycheff 标量化方法，本文提出了一种轻量级的方法，用于梯度型多目标优化，具有更低的计算复杂性但仍能找到所有帕累托解。 |

# 详细

[^1]: 基于自定义生物启发目标的无监督端到端训练

    Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target

    [https://arxiv.org/abs/2403.12116](https://arxiv.org/abs/2403.12116)

    本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。

    

    当前的无监督学习方法依赖于通过深度学习技术（如自监督学习）进行端到端训练，具有较高的计算需求，或者采用通过类似Hebbian学习的生物启发方法逐层训练，使用与监督学习不兼容的局部学习规则。为了解决这一挑战，在这项工作中，我们引入了一种使用网络最终层的胜者通吃（WTA）选择性的“自定义目标”，并通过生物启发的稳态机制进行正则化。

    arXiv:2403.12116v1 Announce Type: cross  Abstract: Current unsupervised learning methods depend on end-to-end training via deep learning techniques such as self-supervised learning, with high computational requirements, or employ layer-by-layer training using bio-inspired approaches like Hebbian learning, using local learning rules incompatible with supervised learning. Both approaches are problematic for edge AI hardware that relies on sparse computational resources and would strongly benefit from alternating between unsupervised and supervised learning phases - thus leveraging widely available unlabeled data from the environment as well as labeled training datasets. To solve this challenge, in this work, we introduce a 'self-defined target' that uses Winner-Take-All (WTA) selectivity at the network's final layer, complemented by regularization through biologically inspired homeostasis mechanism. This approach, framework-agnostic and compatible with both global (Backpropagation) and l
    
[^2]: 光滑 Tchebycheff 标量化用于多目标优化

    Smooth Tchebycheff Scalarization for Multi-Objective Optimization

    [https://arxiv.org/abs/2402.19078](https://arxiv.org/abs/2402.19078)

    通过光滑 Tchebycheff 标量化方法，本文提出了一种轻量级的方法，用于梯度型多目标优化，具有更低的计算复杂性但仍能找到所有帕累托解。

    

    多目标优化问题在许多现实世界应用中都能找到，在这些问题中，目标经常相互冲突，不能通过单个解进行优化。在过去的几十年中，已经提出了许多方法来找到帕累托解，这些解代表了对于给定问题的不同最佳权衡。然而，这些现有方法可能具有较高的计算复杂性，或者可能不能具备解决一般可微分多目标优化问题的良好理论属性。在本项工作中，通过利用光滑优化技术，我们提出了一种新颖且轻量的光滑 Tchebycheff 标量化方法，用于基于梯度的多目标优化。它对于找到所有帕累托解具有良好的理论属性，同时相对于其他方法具有显着较低的计算复杂性。在各种实验结果上

    arXiv:2402.19078v1 Announce Type: cross  Abstract: Multi-objective optimization problems can be found in many real-world applications, where the objectives often conflict each other and cannot be optimized by a single solution. In the past few decades, numerous methods have been proposed to find Pareto solutions that represent different optimal trade-offs among the objectives for a given problem. However, these existing methods could have high computational complexity or may not have good theoretical properties for solving a general differentiable multi-objective optimization problem. In this work, by leveraging the smooth optimization technique, we propose a novel and lightweight smooth Tchebycheff scalarization approach for gradient-based multi-objective optimization. It has good theoretical properties for finding all Pareto solutions with valid trade-off preferences, while enjoying significantly lower computational complexity compared to other methods. Experimental results on variou
    

