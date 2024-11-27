# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics.](http://arxiv.org/abs/2401.15122) | 提出了一种能够促进数值MD模拟并有效模拟蛋白质-配体结合动力学的NeuralMD方法，采用物理信息多级对称框架，实现了准确建模多级蛋白质-配体相互作用。 |
| [^2] | [Getting aligned on representational alignment.](http://arxiv.org/abs/2310.13018) | 该论文研究了生物和人工信息处理系统的表示一致性，探讨了不同系统之间的表示是否一致以及如何调整表示以更好地匹配其他系统。为了改善领域之间的交流，提出了一个统一的框架作为共同语言。 |

# 详细

[^1]: 一种多级对称微分方程模型用于学习蛋白质-配体结合动力学

    A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics. (arXiv:2401.15122v1 [cs.LG])

    [http://arxiv.org/abs/2401.15122](http://arxiv.org/abs/2401.15122)

    提出了一种能够促进数值MD模拟并有效模拟蛋白质-配体结合动力学的NeuralMD方法，采用物理信息多级对称框架，实现了准确建模多级蛋白质-配体相互作用。

    

    在药物发现中，蛋白质-配体结合的分子动力学（MD）模拟提供了一种强大的工具，用于预测结合亲和力，估计运输性能和探索口袋位点。通过改进数值方法以及最近通过机器学习（ML）方法增强MD模拟的效率已经有了很长的历史。然而，仍然存在一些挑战，例如准确建模扩展时间尺度的模拟。为了解决这个问题，我们提出了NeuralMD，这是第一个能够促进数值MD并提供准确的蛋白质-配体结合动力学模拟的ML辅助工具。我们提出了一个合理的方法，将一种新的物理信息多级对称框架纳入模型中。具体而言，我们提出了（1）一个使用向量框架满足群对称性并捕获多级蛋白质-配体相互作用的BindingNet模型，以及（2）一个增强的神经微分方程求解器，学习轨迹的演化。

    In drug discovery, molecular dynamics (MD) simulation for protein-ligand binding provides a powerful tool for predicting binding affinities, estimating transport properties, and exploring pocket sites. There has been a long history of improving the efficiency of MD simulations through better numerical methods and, more recently, by augmenting them with machine learning (ML) methods. Yet, challenges remain, such as accurate modeling of extended-timescale simulations. To address this issue, we propose NeuralMD, the first ML surrogate that can facilitate numerical MD and provide accurate simulations of protein-ligand binding dynamics. We propose a principled approach that incorporates a novel physics-informed multi-grained group symmetric framework. Specifically, we propose (1) a BindingNet model that satisfies group symmetry using vector frames and captures the multi-level protein-ligand interactions, and (2) an augmented neural differential equation solver that learns the trajectory und
    
[^2]: 对表示一致性达成共识

    Getting aligned on representational alignment. (arXiv:2310.13018v1 [q-bio.NC])

    [http://arxiv.org/abs/2310.13018](http://arxiv.org/abs/2310.13018)

    该论文研究了生物和人工信息处理系统的表示一致性，探讨了不同系统之间的表示是否一致以及如何调整表示以更好地匹配其他系统。为了改善领域之间的交流，提出了一个统一的框架作为共同语言。

    

    生物和人工信息处理系统构建可以用来进行分类、推理、规划、导航和决策的世界表示。这些多样化系统所构建的表示在多大程度上是一致的？即使表示不同，是否仍然能够导致相同的行为？系统如何修改它们的表示以更好地匹配另一个系统的表示？这些关于表示一致性研究的问题是当代认知科学、神经科学和机器学习中一些最活跃的研究领域的核心。不幸的是，对于对表示一致性感兴趣的研究社区之间的知识转移有限，其中大部分在一个领域的进展最终会在另一个领域独立地重新发现，而更广泛的领域间交流将是有利的。为了改善领域之间的交流，我们提出了一个统一的框架，可以作为一种共同的语言。

    Biological and artificial information processing systems form representations of the world that they can use to categorize, reason, plan, navigate, and make decisions. To what extent do the representations formed by these diverse systems agree? Can diverging representations still lead to the same behaviors? And how can systems modify their representations to better match those of another system? These questions pertaining to the study of \textbf{\emph{representational alignment}} are at the heart of some of the most active research areas in contemporary cognitive science, neuroscience, and machine learning. Unfortunately, there is limited knowledge-transfer between research communities interested in representational alignment, and much of the progress in one field ends up being rediscovered independently in another, when greater cross-field communication would be advantageous. To improve communication between fields, we propose a unifying framework that can serve as a common language b
    

