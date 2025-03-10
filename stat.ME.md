# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse Bayesian Multidimensional Item Response Theory.](http://arxiv.org/abs/2310.17820) | 本文开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷，并通过贝叶斯非参数方法解决了未知潜在因子维度的问题。 |

# 详细

[^1]: 稀疏贝叶斯多维项目反应理论

    Sparse Bayesian Multidimensional Item Response Theory. (arXiv:2310.17820v1 [stat.ME])

    [http://arxiv.org/abs/2310.17820](http://arxiv.org/abs/2310.17820)

    本文开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷，并通过贝叶斯非参数方法解决了未知潜在因子维度的问题。

    

    多变量项目反应理论（MIRT）被应用研究人员广泛使用，以寻找问卷数据中响应模式背后的可解释（稀疏）解释。然而，在实践中，对于这种稀疏性发现工具的需求尚未得到满足。本文提出了一种用于二元和有序项目MIRT的贝叶斯平台，其需要最少的调整，并且由于其可并行化的特性，在相对较大的数据集上具有良好的可扩展性。MIRT模型的贝叶斯方法传统上依赖于MCMC模拟，在实践中可能既费时又难以通过额外的阈值设定实现精确的稀疏恢复。在本文中，我们开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷。我们利用贝叶斯非参数方法解决了未知潜在因子维度的看似不可逾越的问题，从而使得可以估计因子的数量。通过旋转可以实现稀疏性。

    Multivariate Item Response Theory (MIRT) is sought-after widely by applied researchers looking for interpretable (sparse) explanations underlying response patterns in questionnaire data. There is, however, an unmet demand for such sparsity discovery tools in practice. Our paper develops a Bayesian platform for binary and ordinal item MIRT which requires minimal tuning and scales well on relatively large datasets due to its parallelizable features. Bayesian methodology for MIRT models has traditionally relied on MCMC simulation, which cannot only be slow in practice, but also often renders exact sparsity recovery impossible without additional thresholding. In this work, we develop a scalable Bayesian EM algorithm to estimate sparse factor loadings from binary and ordinal item responses. We address the seemingly insurmountable problem of unknown latent factor dimensionality with tools from Bayesian nonparametrics which enable estimating the number of factors. Rotations to sparsity throug
    

