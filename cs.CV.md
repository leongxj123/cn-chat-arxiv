# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variational Bayes image restoration with compressive autoencoders](https://arxiv.org/abs/2311.17744) | 使用压缩自动编码器代替最先进的生成模型，提出了一种在图像恢复中的新方法。 |
| [^2] | [Where We Have Arrived in Proving the Emergence of Sparse Symbolic Concepts in AI Models.](http://arxiv.org/abs/2305.01939) | 证明了对于训练良好的AI模型，如果满足一定条件，将出现稀疏交互概念，这些概念能够描述输入变量之间的相互作用，并对模型推理分数产生影响。 |

# 详细

[^1]: 使用压缩自动编码器的变分贝叶斯图像恢复

    Variational Bayes image restoration with compressive autoencoders

    [https://arxiv.org/abs/2311.17744](https://arxiv.org/abs/2311.17744)

    使用压缩自动编码器代替最先进的生成模型，提出了一种在图像恢复中的新方法。

    

    逆问题的正则化在计算成像中至关重要。近年来，神经网络学习有效图像表示的能力已被利用来设计强大的数据驱动正则化器。本文首先提出使用压缩自动编码器。这些网络可以被看作具有灵活潜在先验的变分自动编码器，比起最先进的生成模型更小更容易训练。

    arXiv:2311.17744v2 Announce Type: replace-cross  Abstract: Regularization of inverse problems is of paramount importance in computational imaging. The ability of neural networks to learn efficient image representations has been recently exploited to design powerful data-driven regularizers. While state-of-the-art plug-and-play methods rely on an implicit regularization provided by neural denoisers, alternative Bayesian approaches consider Maximum A Posteriori (MAP) estimation in the latent space of a generative model, thus with an explicit regularization. However, state-of-the-art deep generative models require a huge amount of training data compared to denoisers. Besides, their complexity hampers the optimization involved in latent MAP derivation. In this work, we first propose to use compressive autoencoders instead. These networks, which can be seen as variational autoencoders with a flexible latent prior, are smaller and easier to train than state-of-the-art generative models. As a
    
[^2]: 证明AI模型中稀疏符号概念的出现

    Where We Have Arrived in Proving the Emergence of Sparse Symbolic Concepts in AI Models. (arXiv:2305.01939v1 [cs.LG])

    [http://arxiv.org/abs/2305.01939](http://arxiv.org/abs/2305.01939)

    证明了对于训练良好的AI模型，如果满足一定条件，将出现稀疏交互概念，这些概念能够描述输入变量之间的相互作用，并对模型推理分数产生影响。

    

    本文旨在证明训练良好的AI模型中出现符号概念的现象。我们证明，如果（1）模型输出相对于输入变量的高阶导数均为零，（2）AI模型可用于遮挡样本且输入样本较少遮挡时会产生更高的置信度，（3）AI模型在遮挡样本上的置信度并不会显著降低，则AI模型将编码稀疏交互概念。每个交互概念表示特定一组输入变量之间的相互作用，并对模型推理分数产生一定的数值影响。具体而言，我们证明了模型的推理分数总是可以表示为所有交互概念的交互效应之和。事实上，我们希望证明出现符号概念的条件非常普遍。这意味着对于大多数AI模型，我们通常可以使用少量的交互概念来模拟模型。

    This paper aims to prove the emergence of symbolic concepts in well-trained AI models. We prove that if (1) the high-order derivatives of the model output w.r.t. the input variables are all zero, (2) the AI model can be used on occluded samples and will yield higher confidence when the input sample is less occluded, and (3) the confidence of the AI model does not significantly degrade on occluded samples, then the AI model will encode sparse interactive concepts. Each interactive concept represents an interaction between a specific set of input variables, and has a certain numerical effect on the inference score of the model. Specifically, it is proved that the inference score of the model can always be represented as the sum of the interaction effects of all interactive concepts. In fact, we hope to prove that conditions for the emergence of symbolic concepts are quite common. It means that for most AI models, we can usually use a small number of interactive concepts to mimic the mode
    

