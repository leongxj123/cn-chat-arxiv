# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Frequentist Guarantees of Distributed (Non)-Bayesian Inference](https://arxiv.org/abs/2311.08214) | 本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，探讨了在适当假设下分布式贝叶斯推断在参数效率和不确定性量化方面的表现，以及通信图设计和大小对后验收缩率的影响。 |
| [^2] | [Generalization Error Curves for Analytic Spectral Algorithms under Power-law Decay.](http://arxiv.org/abs/2401.01599) | 本文研究了核回归方法的泛化误差曲线，对核梯度下降方法和其他分析谱算法在核回归中的泛化误差进行了全面特征化，从而提高了对训练宽神经网络泛化行为的理解，并提出了一种新的技术贡献-分析功能论证。 |

# 详细

[^1]: 分布式(非)贝叶斯推断的频率保证

    Frequentist Guarantees of Distributed (Non)-Bayesian Inference

    [https://arxiv.org/abs/2311.08214](https://arxiv.org/abs/2311.08214)

    本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，探讨了在适当假设下分布式贝叶斯推断在参数效率和不确定性量化方面的表现，以及通信图设计和大小对后验收缩率的影响。

    

    受分析大型分散数据集的需求推动，分布式贝叶斯推断已成为跨多个领域（包括统计学、电气工程和经济学）的关键研究领域。本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，如后验一致性、渐近正态性和后验收缩率。我们的结果表明，在通信图上的适当假设下，分布式贝叶斯推断保留了参数效率，同时在不确定性量化方面增强了鲁棒性。我们还通过研究设计和通信图的大小如何影响后验收缩率来探讨了统计效率和通信效率之间的权衡。此外，我们将我们的分析扩展到时变图，并将结果应用于指数f

    arXiv:2311.08214v2 Announce Type: replace-cross  Abstract: Motivated by the need to analyze large, decentralized datasets, distributed Bayesian inference has become a critical research area across multiple fields, including statistics, electrical engineering, and economics. This paper establishes Frequentist properties, such as posterior consistency, asymptotic normality, and posterior contraction rates, for the distributed (non-)Bayes Inference problem among agents connected via a communication network. Our results show that, under appropriate assumptions on the communication graph, distributed Bayesian inference retains parametric efficiency while enhancing robustness in uncertainty quantification. We also explore the trade-off between statistical efficiency and communication efficiency by examining how the design and size of the communication graph impact the posterior contraction rate. Furthermore, We extend our analysis to time-varying graphs and apply our results to exponential f
    
[^2]: 分析谱算法在幂律衰减下的泛化误差曲线

    Generalization Error Curves for Analytic Spectral Algorithms under Power-law Decay. (arXiv:2401.01599v1 [cs.LG])

    [http://arxiv.org/abs/2401.01599](http://arxiv.org/abs/2401.01599)

    本文研究了核回归方法的泛化误差曲线，对核梯度下降方法和其他分析谱算法在核回归中的泛化误差进行了全面特征化，从而提高了对训练宽神经网络泛化行为的理解，并提出了一种新的技术贡献-分析功能论证。

    

    某些核回归方法的泛化误差曲线旨在确定在不同源条件、噪声水平和正则化参数选择下的泛化误差的确切顺序，而不是最小化率。在本文中，在温和的假设下，我们严格给出了核梯度下降方法（以及大类分析谱算法）在核回归中的泛化误差曲线的完整特征化。因此，我们可以提高核插值的近不一致性，并澄清具有更高资格的核回归算法的饱和效应，等等。由于神经切线核理论的帮助，这些结果极大地提高了我们对训练宽神经网络的泛化行为的理解。一种新颖的技术贡献，即分析功能论证，可能具有独立的兴趣。

    The generalization error curve of certain kernel regression method aims at determining the exact order of generalization error with various source condition, noise level and choice of the regularization parameter rather than the minimax rate. In this work, under mild assumptions, we rigorously provide a full characterization of the generalization error curves of the kernel gradient descent method (and a large class of analytic spectral algorithms) in kernel regression. Consequently, we could sharpen the near inconsistency of kernel interpolation and clarify the saturation effects of kernel regression algorithms with higher qualification, etc. Thanks to the neural tangent kernel theory, these results greatly improve our understanding of the generalization behavior of training the wide neural networks. A novel technical contribution, the analytic functional argument, might be of independent interest.
    

