# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Two Tales of Single-Phase Contrastive Hebbian Learning](https://arxiv.org/abs/2402.08573) | 两种单相对比海比安学习的故事探索了学习算法的生物合理性，并提出了一种全局学习算法，能够消除与反向传播之间的性能差距，并解决了同步和无限小扰动带来的问题。 |

# 详细

[^1]: 两种单相对比海比安学习的故事

    Two Tales of Single-Phase Contrastive Hebbian Learning

    [https://arxiv.org/abs/2402.08573](https://arxiv.org/abs/2402.08573)

    两种单相对比海比安学习的故事探索了学习算法的生物合理性，并提出了一种全局学习算法，能够消除与反向传播之间的性能差距，并解决了同步和无限小扰动带来的问题。

    

    对于“生物学上合理”的学习算法的探索已经收敛于将梯度表示为活动差异的想法。然而，大多数方法需要较高程度的同步（学习期间的不同阶段）并引入大量的计算开销，这对于它们的生物学合理性以及其在神经形态计算中的潜在效用产生了疑问。此外，它们通常依赖于对输出单元施加无限小扰动（nudges），这在嘈杂环境中是不切实际的。最近研究发现，通过将人工神经元建模为两个相反扰动的组件，名为“双向传播”的全局学习算法能够弥合到反向传播的性能差距，而不需要分别的学习阶段或无限小扰动。然而，该算法的数值稳定性依赖于对称扰动，这可能在生物学上受到限制。

    The search for "biologically plausible" learning algorithms has converged on the idea of representing gradients as activity differences. However, most approaches require a high degree of synchronization (distinct phases during learning) and introduce substantial computational overhead, which raises doubts regarding their biological plausibility as well as their potential utility for neuromorphic computing. Furthermore, they commonly rely on applying infinitesimal perturbations (nudges) to output units, which is impractical in noisy environments. Recently it has been shown that by modelling artificial neurons as dyads with two oppositely nudged compartments, it is possible for a fully local learning algorithm named ``dual propagation'' to bridge the performance gap to backpropagation, without requiring separate learning phases or infinitesimal nudging. However, the algorithm has the drawback that its numerical stability relies on symmetric nudging, which may be restrictive in biological
    

