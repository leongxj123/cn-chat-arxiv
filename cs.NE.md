# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SWAP-NAS: Sample-Wise Activation Patterns For Ultra-Fast NAS](https://arxiv.org/abs/2403.04161) | 提出了一种新颖的高性能无需训练的度量SWAP-Score，能够在不同搜索空间和任务中测量网络在一批输入样本上的表现能力，并通过正则化进一步提高相关性，实现模型大小的控制。 |
| [^2] | [Two Tales of Single-Phase Contrastive Hebbian Learning](https://arxiv.org/abs/2402.08573) | 两种单相对比海比安学习的故事探索了学习算法的生物合理性，并提出了一种全局学习算法，能够消除与反向传播之间的性能差距，并解决了同步和无限小扰动带来的问题。 |
| [^3] | [Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey](https://arxiv.org/abs/2401.11963) | 通过整合进化算法与强化学习，进化强化学习（ERL）展现出卓越的性能提升，本综述呈现了ERL领域的各个研究分支，突出了EA辅助RL的优化、RL辅助EA的优化以及EA和RL的协同优化这三个主要研究方向。 |

# 详细

[^1]: SWAP-NAS: 适用于超快速NAS的样本级激活模式

    SWAP-NAS: Sample-Wise Activation Patterns For Ultra-Fast NAS

    [https://arxiv.org/abs/2403.04161](https://arxiv.org/abs/2403.04161)

    提出了一种新颖的高性能无需训练的度量SWAP-Score，能够在不同搜索空间和任务中测量网络在一批输入样本上的表现能力，并通过正则化进一步提高相关性，实现模型大小的控制。

    

    无需训练的度量（即零成本代理）被广泛用于避免资源密集型的神经网络训练，尤其是在神经结构搜索（NAS）中。最近的研究表明，现有的无需训练的度量存在一些局限，比如在不同搜索空间和任务之间存在有限的关联性和差劲的泛化能力。因此，我们提出了样本级激活模式及其衍生物SWAP-Score，这是一种新颖的高性能无需训练的度量。它测量了网络在一批输入样本上的表现能力。SWAP-Score与不同搜索空间和任务中的真实性能强相关，在NAS-Bench-101/201/301和TransNAS-Bench-101上胜过了15种现有的无需训练的度量。SWAP-Score可以通过正则化进一步增强，这在基于单元的搜索空间中可以实现更高的相关性，并且在搜索过程中实现模型大小控制。例如，Spearman的排序

    arXiv:2403.04161v1 Announce Type: new  Abstract: Training-free metrics (a.k.a. zero-cost proxies) are widely used to avoid resource-intensive neural network training, especially in Neural Architecture Search (NAS). Recent studies show that existing training-free metrics have several limitations, such as limited correlation and poor generalisation across different search spaces and tasks. Hence, we propose Sample-Wise Activation Patterns and its derivative, SWAP-Score, a novel high-performance training-free metric. It measures the expressivity of networks over a batch of input samples. The SWAP-Score is strongly correlated with ground-truth performance across various search spaces and tasks, outperforming 15 existing training-free metrics on NAS-Bench-101/201/301 and TransNAS-Bench-101. The SWAP-Score can be further enhanced by regularisation, which leads to even higher correlations in cell-based search space and enables model size control during the search. For example, Spearman's rank
    
[^2]: 两种单相对比海比安学习的故事

    Two Tales of Single-Phase Contrastive Hebbian Learning

    [https://arxiv.org/abs/2402.08573](https://arxiv.org/abs/2402.08573)

    两种单相对比海比安学习的故事探索了学习算法的生物合理性，并提出了一种全局学习算法，能够消除与反向传播之间的性能差距，并解决了同步和无限小扰动带来的问题。

    

    对于“生物学上合理”的学习算法的探索已经收敛于将梯度表示为活动差异的想法。然而，大多数方法需要较高程度的同步（学习期间的不同阶段）并引入大量的计算开销，这对于它们的生物学合理性以及其在神经形态计算中的潜在效用产生了疑问。此外，它们通常依赖于对输出单元施加无限小扰动（nudges），这在嘈杂环境中是不切实际的。最近研究发现，通过将人工神经元建模为两个相反扰动的组件，名为“双向传播”的全局学习算法能够弥合到反向传播的性能差距，而不需要分别的学习阶段或无限小扰动。然而，该算法的数值稳定性依赖于对称扰动，这可能在生物学上受到限制。

    The search for "biologically plausible" learning algorithms has converged on the idea of representing gradients as activity differences. However, most approaches require a high degree of synchronization (distinct phases during learning) and introduce substantial computational overhead, which raises doubts regarding their biological plausibility as well as their potential utility for neuromorphic computing. Furthermore, they commonly rely on applying infinitesimal perturbations (nudges) to output units, which is impractical in noisy environments. Recently it has been shown that by modelling artificial neurons as dyads with two oppositely nudged compartments, it is possible for a fully local learning algorithm named ``dual propagation'' to bridge the performance gap to backpropagation, without requiring separate learning phases or infinitesimal nudging. However, the algorithm has the drawback that its numerical stability relies on symmetric nudging, which may be restrictive in biological
    
[^3]: 跨越进化算法和强化学习：一项全面调查

    Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey

    [https://arxiv.org/abs/2401.11963](https://arxiv.org/abs/2401.11963)

    通过整合进化算法与强化学习，进化强化学习（ERL）展现出卓越的性能提升，本综述呈现了ERL领域的各个研究分支，突出了EA辅助RL的优化、RL辅助EA的优化以及EA和RL的协同优化这三个主要研究方向。

    

    进化强化学习（ERL）将进化算法（EAs）和强化学习（RL）相结合进行优化，表现出卓越的性能提升。通过融合两种方法的优势，ERL已经成为一个有前景的研究方向。本调查综述了ERL中不同研究分支的全面概述。具体而言，我们系统总结了相关算法的最新进展，并确定了三个主要研究方向：EA辅助RL的优化，RL辅助EA的优化，以及EA和RL的协同优化。随后，我们深入分析了每个研究方向，组织了多个研究分支。我们阐明了每个分支致力于解决的问题，以及EA和RL的整合如何应对这些挑战。最后，我们讨论了潜在的挑战和未来的研究方向。

    arXiv:2401.11963v2 Announce Type: replace-cross  Abstract: Evolutionary Reinforcement Learning (ERL), which integrates Evolutionary Algorithms (EAs) and Reinforcement Learning (RL) for optimization, has demonstrated remarkable performance advancements. By fusing the strengths of both approaches, ERL has emerged as a promising research direction. This survey offers a comprehensive overview of the diverse research branches in ERL. Specifically, we systematically summarize recent advancements in relevant algorithms and identify three primary research directions: EA-assisted optimization of RL, RL-assisted optimization of EA, and synergistic optimization of EA and RL. Following that, we conduct an in-depth analysis of each research direction, organizing multiple research branches. We elucidate the problems that each branch aims to tackle and how the integration of EA and RL addresses these challenges. In conclusion, we discuss potential challenges and prospective future research directions
    

