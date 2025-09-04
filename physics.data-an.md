# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Variational Multivariate Information Bottleneck -- A Framework for Variational Losses.](http://arxiv.org/abs/2310.03311) | 该论文介绍了一个基于信息理论的统一原理，用于重新推导和推广现有的变分降维方法，并设计新的方法。通过将多变量信息瓶颈解释为两个贝叶斯网络的权衡，该框架引入了一个在压缩数据和保留信息之间的权衡参数。 |

# 详细

[^1]: 深度变分多变量信息瓶颈--一种变分损失的框架

    Deep Variational Multivariate Information Bottleneck -- A Framework for Variational Losses. (arXiv:2310.03311v1 [cs.LG])

    [http://arxiv.org/abs/2310.03311](http://arxiv.org/abs/2310.03311)

    该论文介绍了一个基于信息理论的统一原理，用于重新推导和推广现有的变分降维方法，并设计新的方法。通过将多变量信息瓶颈解释为两个贝叶斯网络的权衡，该框架引入了一个在压缩数据和保留信息之间的权衡参数。

    

    变分降维方法以其高精度、生成能力和鲁棒性而闻名。这些方法有很多理论上的证明。在这里，我们介绍了一种基于信息理论的统一原理，重新推导和推广了现有的变分方法，并设计了新的方法。我们的框架基于多变量信息瓶颈的解释，其中两个贝叶斯网络相互权衡。我们将第一个网络解释为编码器图，它指定了在压缩数据时要保留的信息。我们将第二个网络解释为解码器图，它为数据指定了一个生成模型。使用这个框架，我们重新推导了现有的降维方法，如深度变分信息瓶颈(DVIB)、beta变分自编码器(beta-VAE)和深度变分规范相关分析(DVCCA)。该框架自然地引入了一个在压缩数据和保留信息之间的权衡参数。

    Variational dimensionality reduction methods are known for their high accuracy, generative abilities, and robustness. These methods have many theoretical justifications. Here we introduce a unifying principle rooted in information theory to rederive and generalize existing variational methods and design new ones. We base our framework on an interpretation of the multivariate information bottleneck, in which two Bayesian networks are traded off against one another. We interpret the first network as an encoder graph, which specifies what information to keep when compressing the data. We interpret the second network as a decoder graph, which specifies a generative model for the data. Using this framework, we rederive existing dimensionality reduction methods such as the deep variational information bottleneck (DVIB), beta variational auto-encoders (beta-VAE), and deep variational canonical correlation analysis (DVCCA). The framework naturally introduces a trade-off parameter between compr
    

