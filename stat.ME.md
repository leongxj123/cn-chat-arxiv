# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Algorithmic syntactic causal identification](https://arxiv.org/abs/2403.09580) | 通过替换传统概率论为对称单调范畴的替代基础，可以扩展因果识别技术到更多因果设置中。 |
| [^2] | [Matrix Supermartingales and Randomized Matrix Concentration Inequalities.](http://arxiv.org/abs/2401.15567) | 本文提出了针对鞅相关或可交换随机对称矩阵的新集中不等式，这些不等式在多种尾条件下成立，在洛伊纳顺序表示，并且有时在任意数据相关停止时间都适用。 |
| [^3] | [Uniform Inference on High-dimensional Spatial Panel Networks.](http://arxiv.org/abs/2105.07424) | 本研究在大规模空间面板网络上提出了一种均匀推断理论，该理论能够对感兴趣的参数进行假设检验，包括网络结构中的零或非零元素。 |

# 详细

[^1]: 算法句法因果识别

    Algorithmic syntactic causal identification

    [https://arxiv.org/abs/2403.09580](https://arxiv.org/abs/2403.09580)

    通过替换传统概率论为对称单调范畴的替代基础，可以扩展因果识别技术到更多因果设置中。

    

    在因果贝叶斯网络（CBN）中进行因果识别是因果推断中的一项重要工具，允许从理论上可能的情况下的观测分布推导干预分布。然而，大多数现有的因果识别形式，如使用d分离和do-演算的技术都是在CBN上利用经典概率论的数学语言表达的。然而，在许多因果设置中，概率论和因此目前的因果识别技术不适用，如关系数据库、数据流程序（例如硬件描述语言）、分布式系统和大多数现代机器学习算法。我们表明，可以通过用对称单调范畴的替代公理基础来消除这种限制。在这种替代公理化中，我们展示了如何获得一个明确且清晰的

    arXiv:2403.09580v1 Announce Type: new  Abstract: Causal identification in causal Bayes nets (CBNs) is an important tool in causal inference allowing the derivation of interventional distributions from observational distributions where this is possible in principle. However, most existing formulations of causal identification using techniques such as d-separation and do-calculus are expressed within the mathematical language of classical probability theory on CBNs. However, there are many causal settings where probability theory and hence current causal identification techniques are inapplicable such as relational databases, dataflow programs such as hardware description languages, distributed systems and most modern machine learning algorithms. We show that this restriction can be lifted by replacing the use of classical probability theory with the alternative axiomatic foundation of symmetric monoidal categories. In this alternative axiomatization, we show how an unambiguous and clean
    
[^2]: 矩阵超鞅和随机矩阵集中不等式

    Matrix Supermartingales and Randomized Matrix Concentration Inequalities. (arXiv:2401.15567v1 [math.PR])

    [http://arxiv.org/abs/2401.15567](http://arxiv.org/abs/2401.15567)

    本文提出了针对鞅相关或可交换随机对称矩阵的新集中不等式，这些不等式在多种尾条件下成立，在洛伊纳顺序表示，并且有时在任意数据相关停止时间都适用。

    

    我们在多种尾条件下，提出了针对鞅相关或可交换随机对称矩阵的新集中不等式，包括标准的切尔诺夫上界和自归一化重尾设置。这些不等式通常以洛伊纳顺序表示，并且有时在任意数据相关停止时间都成立。在此过程中，我们探索了矩阵超鞅和极值不等式的理论，可能具有独立的研究价值。

    We present new concentration inequalities for either martingale dependent or exchangeable random symmetric matrices under a variety of tail conditions, encompassing standard Chernoff bounds to self-normalized heavy-tailed settings. These inequalities are often randomized in a way that renders them strictly tighter than existing deterministic results in the literature, are typically expressed in the Loewner order, and are sometimes valid at arbitrary data-dependent stopping times.  Along the way, we explore the theory of matrix supermartingales and maximal inequalities, potentially of independent interest.
    
[^3]: 高维空间面板网络上的均匀推断

    Uniform Inference on High-dimensional Spatial Panel Networks. (arXiv:2105.07424v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2105.07424](http://arxiv.org/abs/2105.07424)

    本研究在大规模空间面板网络上提出了一种均匀推断理论，该理论能够对感兴趣的参数进行假设检验，包括网络结构中的零或非零元素。

    

    我们提出了一种偏差-正则化的高维广义矩方法（GMM）框架，用于对大规模空间面板网络进行推断。特别是，利用偏差机器学习方法估计具有灵活稀疏偏差的网络结构，这可以被视为潜在的或者与预定的邻接矩阵不匹配。理论分析确立了我们提出的估计器的一致性和渐近正态性，考虑了数据生成过程中的一般时间和空间依赖性。讨论了依赖性存在时的维度允许性。我们研究的一个主要贡献是开发了一种均匀推断理论，能够对感兴趣的参数进行假设检验，包括网络结构中的零或非零元素。此外，对估计器的渐近性质进行了线性和非线性时刻的推导。模拟实验证明了所提方法的有效性。

    We propose employing a debiased-regularized, high-dimensional generalized method of moments (GMM) framework to perform inference on large-scale spatial panel networks. In particular, network structure with a flexible sparse deviation, which can be regarded either as latent or as misspecified from a predetermined adjacency matrix, is estimated using debiased machine learning approach. The theoretical analysis establishes the consistency and asymptotic normality of our proposed estimator, taking into account general temporal and spatial dependency inherent in the data-generating processes. The dimensionality allowance in presence of dependency is discussed. A primary contribution of our study is the development of uniform inference theory that enables hypothesis testing on the parameters of interest, including zero or non-zero elements in the network structure. Additionally, the asymptotic properties for the estimator are derived for both linear and nonlinear moments. Simulations demonst
    

