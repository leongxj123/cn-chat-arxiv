# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Backpropagation through space, time, and the brain](https://arxiv.org/abs/2403.16933) | 提出了 Generalized Latent Equilibrium (GLE)，它是一种针对神经元网络的物理动态局部时空信用分配的计算框架。 |
| [^2] | [Joint-Embedding Masked Autoencoder for Self-supervised Learning of Dynamic Functional Connectivity from the Human Brain](https://arxiv.org/abs/2403.06432) | 提出了一种受到计算机视觉中 JEPA 架构启发的 Spatio-Temporal Joint Embedding Masked Autoencoder（ST-JEMA）用于动态功能连接的自监督学习。 |

# 详细

[^1]: 通过空间、时间和大脑进行反向传播

    Backpropagation through space, time, and the brain

    [https://arxiv.org/abs/2403.16933](https://arxiv.org/abs/2403.16933)

    提出了 Generalized Latent Equilibrium (GLE)，它是一种针对神经元网络的物理动态局部时空信用分配的计算框架。

    

    有效的神经网络学习需要根据它们对解决任务的相对贡献来调整单个突触。然而，无论是生物还是人工的物理神经系统都受到时空局限。这样的网络如何执行高效的信用分配，在很大程度上仍是一个悬而未决的问题。在机器学习中，错误的反向传播算法几乎普遍被空间（BP）和时间（BPTT）两种方式给出答案。然而，BP(TT)被广泛认为依赖于不具生物学意义的假设，特别是关于时空局限性，而正向传播模型，如实时递归学习（RTRL），则受到内存约束的限制。我们引入了广义潜在平衡（GLE），这是一个针对神经元物理动态网络完全局部时空信用分配的计算框架。我们从

    arXiv:2403.16933v1 Announce Type: cross  Abstract: Effective learning in neuronal networks requires the adaptation of individual synapses given their relative contribution to solving a task. However, physical neuronal systems -- whether biological or artificial -- are constrained by spatio-temporal locality. How such networks can perform efficient credit assignment, remains, to a large extent, an open question. In Machine Learning, the answer is almost universally given by the error backpropagation algorithm, through both space (BP) and time (BPTT). However, BP(TT) is well-known to rely on biologically implausible assumptions, in particular with respect to spatiotemporal (non-)locality, while forward-propagation models such as real-time recurrent learning (RTRL) suffer from prohibitive memory constraints. We introduce Generalized Latent Equilibrium (GLE), a computational framework for fully local spatio-temporal credit assignment in physical, dynamical networks of neurons. We start by 
    
[^2]: 人类大脑动态功能连接的自监督学习中的联合嵌入掩蔽自编码器

    Joint-Embedding Masked Autoencoder for Self-supervised Learning of Dynamic Functional Connectivity from the Human Brain

    [https://arxiv.org/abs/2403.06432](https://arxiv.org/abs/2403.06432)

    提出了一种受到计算机视觉中 JEPA 架构启发的 Spatio-Temporal Joint Embedding Masked Autoencoder（ST-JEMA）用于动态功能连接的自监督学习。

    

    arXiv:2403.06432v1 通告类型: 新的 摘要: 图神经网络（GNNs）在学习动态功能连接方面表现出潜力，可以区分人脑网络中的表现型。然而，获得用于训练的大量标记临床数据通常具有资源密集性，这使得实际应用变得困难。因此，在标签稀缺设置中，利用未标记数据对于表示学习变得至关重要。尽管生成式自监督学习技术，特别是掩蔽自编码器，在各个领域的表示学习中展现出了有希望的结果，但它们在动态图形上的应用以及动态功能连接方面仍未得到充分探讨，面临着捕捉高级语义表示方面的挑战。在这里，我们介绍了时空联合嵌入掩蔽自编码器（ST-JEMA），受到计算机视觉中联合嵌入预测架构（JEPA）的启发。ST-JEMA采用了一种受JEPA启发的策略来重构

    arXiv:2403.06432v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have shown promise in learning dynamic functional connectivity for distinguishing phenotypes from human brain networks. However, obtaining extensive labeled clinical data for training is often resource-intensive, making practical application difficult. Leveraging unlabeled data thus becomes crucial for representation learning in a label-scarce setting. Although generative self-supervised learning techniques, especially masked autoencoders, have shown promising results in representation learning in various domains, their application to dynamic graphs for dynamic functional connectivity remains underexplored, facing challenges in capturing high-level semantic representations. Here, we introduce the Spatio-Temporal Joint Embedding Masked Autoencoder (ST-JEMA), drawing inspiration from the Joint Embedding Predictive Architecture (JEPA) in computer vision. ST-JEMA employs a JEPA-inspired strategy for reconstructin
    

