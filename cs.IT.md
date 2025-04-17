# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Resilience of the quadratic Littlewood-Offord problem](https://arxiv.org/abs/2402.10504) | 论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。 |
| [^2] | [Deep Variational Multivariate Information Bottleneck -- A Framework for Variational Losses.](http://arxiv.org/abs/2310.03311) | 该论文介绍了一个基于信息理论的统一原理，用于重新推导和推广现有的变分降维方法，并设计新的方法。通过将多变量信息瓶颈解释为两个贝叶斯网络的权衡，该框架引入了一个在压缩数据和保留信息之间的权衡参数。 |
| [^3] | [Local Grammar-Based Coding Revisited.](http://arxiv.org/abs/2209.13636) | 本文重新审视了最小局部基于语法的编码问题，并提出了一种新的、更简单、更普遍的证明方法，证明了最小分块编码具有强大的普遍性。同时，通过实验也表明，最小分块编码中的规则数量不能明确区分长记忆和无记忆的源。 |

# 详细

[^1]: 二次Littlewood-Offord问题的弹性

    Resilience of the quadratic Littlewood-Offord problem

    [https://arxiv.org/abs/2402.10504](https://arxiv.org/abs/2402.10504)

    论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。

    

    我们研究了高维数据的统计鲁棒性。我们的结果提供了关于对抗性噪声对二次Radamecher混沌$\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$反集中特性的影响的估计，其中$M$是一个固定的（高维）矩阵，$\boldsymbol{\xi}$是一个共形Rademacher向量。具体来说，我们探讨了$\boldsymbol{\xi}$能够承受多少对抗性符号翻转而不“膨胀”$\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$，从而“去除”原始分布导致更“有粒度”和对抗性偏倚的分布。我们的结果为二次和双线性Rademacher混沌的统计鲁棒性提供了下限估计；这些结果在关键区域被证明是渐近紧的。

    arXiv:2402.10504v1 Announce Type: cross  Abstract: We study the statistical resilience of high-dimensional data. Our results provide estimates as to the effects of adversarial noise over the anti-concentration properties of the quadratic Radamecher chaos $\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$, where $M$ is a fixed (high-dimensional) matrix and $\boldsymbol{\xi}$ is a conformal Rademacher vector. Specifically, we pursue the question of how many adversarial sign-flips can $\boldsymbol{\xi}$ sustain without "inflating" $\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$ and thus "de-smooth" the original distribution resulting in a more "grainy" and adversarially biased distribution. Our results provide lower bound estimations for the statistical resilience of the quadratic and bilinear Rademacher chaos; these are shown to be asymptotically tight across key regimes.
    
[^2]: 深度变分多变量信息瓶颈--一种变分损失的框架

    Deep Variational Multivariate Information Bottleneck -- A Framework for Variational Losses. (arXiv:2310.03311v1 [cs.LG])

    [http://arxiv.org/abs/2310.03311](http://arxiv.org/abs/2310.03311)

    该论文介绍了一个基于信息理论的统一原理，用于重新推导和推广现有的变分降维方法，并设计新的方法。通过将多变量信息瓶颈解释为两个贝叶斯网络的权衡，该框架引入了一个在压缩数据和保留信息之间的权衡参数。

    

    变分降维方法以其高精度、生成能力和鲁棒性而闻名。这些方法有很多理论上的证明。在这里，我们介绍了一种基于信息理论的统一原理，重新推导和推广了现有的变分方法，并设计了新的方法。我们的框架基于多变量信息瓶颈的解释，其中两个贝叶斯网络相互权衡。我们将第一个网络解释为编码器图，它指定了在压缩数据时要保留的信息。我们将第二个网络解释为解码器图，它为数据指定了一个生成模型。使用这个框架，我们重新推导了现有的降维方法，如深度变分信息瓶颈(DVIB)、beta变分自编码器(beta-VAE)和深度变分规范相关分析(DVCCA)。该框架自然地引入了一个在压缩数据和保留信息之间的权衡参数。

    Variational dimensionality reduction methods are known for their high accuracy, generative abilities, and robustness. These methods have many theoretical justifications. Here we introduce a unifying principle rooted in information theory to rederive and generalize existing variational methods and design new ones. We base our framework on an interpretation of the multivariate information bottleneck, in which two Bayesian networks are traded off against one another. We interpret the first network as an encoder graph, which specifies what information to keep when compressing the data. We interpret the second network as a decoder graph, which specifies a generative model for the data. Using this framework, we rederive existing dimensionality reduction methods such as the deep variational information bottleneck (DVIB), beta variational auto-encoders (beta-VAE), and deep variational canonical correlation analysis (DVCCA). The framework naturally introduces a trade-off parameter between compr
    
[^3]: 本文重新审视了局部基于语法的编码问题

    Local Grammar-Based Coding Revisited. (arXiv:2209.13636v2 [cs.IT] UPDATED)

    [http://arxiv.org/abs/2209.13636](http://arxiv.org/abs/2209.13636)

    本文重新审视了最小局部基于语法的编码问题，并提出了一种新的、更简单、更普遍的证明方法，证明了最小分块编码具有强大的普遍性。同时，通过实验也表明，最小分块编码中的规则数量不能明确区分长记忆和无记忆的源。

    

    本文重新审视了最小局部基于语法的编码问题。在这个设置中，局部基于语法的编码器逐个符号地对语法进行编码，而最小语法变换通过局部语法编码的长度在预设的语法类别中最小化语法长度。已知，这样的最小编码对于严格正熵率的情况具有强大的普遍性，而最小语法中的规则数量构成了源的互信息的上界。尽管完全最小编码可能是不可行的，但受限的最小分块编码可以有效计算。本文提出了一种新的、更简单、更普适的最小分块编码强大普遍性的证明方法，不受熵率的限制。该证明基于对排名概率的简单的Zipfian界限。顺便提一下，我们还通过实验证明，最小分块编码中的规则数量不能明确区分长记忆和无记忆的源。

    We revisit the problem of minimal local grammar-based coding. In this setting, the local grammar encoder encodes grammars symbol by symbol, whereas the minimal grammar transform minimizes the grammar length in a preset class of grammars as given by the length of local grammar encoding. It has been known that such minimal codes are strongly universal for a strictly positive entropy rate, whereas the number of rules in the minimal grammar constitutes an upper bound for the mutual information of the source. Whereas the fully minimal code is likely intractable, the constrained minimal block code can be efficiently computed. In this article, we present a new, simpler, and more general proof of strong universality of the minimal block code, regardless of the entropy rate. The proof is based on a simple Zipfian bound for ranked probabilities. By the way, we also show empirically that the number of rules in the minimal block code cannot clearly discriminate between long-memory and memoryless s
    

