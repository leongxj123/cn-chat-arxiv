# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python](https://arxiv.org/abs/2402.02290) | QuadratiK软件包是一个在R和Python中实现的数据分析工具，它提供了一套全面的拟合度测试和基于核方法的聚类技术，特别适用于处理球形数据。 |
| [^2] | [Sequence Generation via Subsequence Similarity: Theory and Application to UAV Identification.](http://arxiv.org/abs/2301.08403) | 本文探究了一种单次生成模型的多样性，主要聚焦于子序列相似性如何影响整个序列相似性，并通过生成子序列相似的序列来增强数据集。 |

# 详细

[^1]: 球形数据的拟合度和聚类：R和Python中的QuadratiK软件包

    Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python

    [https://arxiv.org/abs/2402.02290](https://arxiv.org/abs/2402.02290)

    QuadratiK软件包是一个在R和Python中实现的数据分析工具，它提供了一套全面的拟合度测试和基于核方法的聚类技术，特别适用于处理球形数据。

    

    我们介绍了QuadratiK软件包，该软件包包含了创新的数据分析方法。该软件包在R和Python中实现，提供了一套全面的适应度拟合测试和基于核方法的二次距离的聚类技术，从而弥合了统计学和机器学习文献之间的差距。我们的软件实现了单样本、双样本和k样本适应度拟合测试，提供了一种高效且数学上合理的方法来评估概率分布的拟合度。我们的软件扩展了功能，包括基于泊松核密度的$d$维球上均匀性测试，以及从泊松核密度中生成随机样本的算法。特别值得注意的是，我们的软件还包括一种针对球形数据而特别量身定制的独特聚类算法，该算法利用了球面上基于泊松核密度的混合模型。同时，我们的软件还包括其他图形功能。

    We introduce the QuadratiK package that incorporates innovative data analysis methodologies. The presented software, implemented in both R and Python, offers a comprehensive set of goodness-of-fit tests and clustering techniques using kernel-based quadratic distances, thereby bridging the gap between the statistical and machine learning literatures. Our software implements one, two and k-sample tests for goodness of fit, providing an efficient and mathematically sound way to assess the fit of probability distributions. Expanded capabilities of our software include supporting tests for uniformity on the $d$-dimensional Sphere based on Poisson kernel densities, and algorithms for generating random samples from Poisson kernel densities. Particularly noteworthy is the incorporation of a unique clustering algorithm specifically tailored for spherical data that leverages a mixture of Poisson-kernel-based densities on the sphere. Alongside this, our software includes additional graphical func
    
[^2]: 通过子序列相似性生成序列：理论及其在无人机识别中的应用

    Sequence Generation via Subsequence Similarity: Theory and Application to UAV Identification. (arXiv:2301.08403v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.08403](http://arxiv.org/abs/2301.08403)

    本文探究了一种单次生成模型的多样性，主要聚焦于子序列相似性如何影响整个序列相似性，并通过生成子序列相似的序列来增强数据集。

    

    生成人工合成序列的能力在广泛的应用中至关重要，而深度学习架构和生成框架的最新进展已经极大地促进了这一过程。本文使用一种单次生成模型来采样，通过相似性生成子序列，并证明了子序列相似性对整个序列相似性的影响，给出了相应的界限。我们使用一种一次性生成模型来从单个序列的范围内取样，并生成子序列相似的序列，证明了数据集增强方面的实用性。

    The ability to generate synthetic sequences is crucial for a wide range of applications, and recent advances in deep learning architectures and generative frameworks have greatly facilitated this process. Particularly, unconditional one-shot generative models constitute an attractive line of research that focuses on capturing the internal information of a single image or video to generate samples with similar contents. Since many of those one-shot models are shifting toward efficient non-deep and non-adversarial approaches, we examine the versatility of a one-shot generative model for augmenting whole datasets. In this work, we focus on how similarity at the subsequence level affects similarity at the sequence level, and derive bounds on the optimal transport of real and generated sequences based on that of corresponding subsequences. We use a one-shot generative model to sample from the vicinity of individual sequences and generate subsequence-similar ones and demonstrate the improvem
    

