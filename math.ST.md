# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile](https://arxiv.org/abs/2403.20200) | 研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。 |
| [^2] | [Spectral Regularized Kernel Goodness-of-Fit Tests.](http://arxiv.org/abs/2308.04561) | 本文提出了具有谱正则化的核拟合优度检验方法，用于处理非欧几里得数据。相比之前的方法，本方法在选择适当的正则化参数时能达到最小化最大风险。同时，本方法还克服了之前方法对均值元素为零和积分操作符特征函数均匀有界性条件的限制，并且能够计算更多种类的核函数。 |
| [^3] | [High-Dimensional Canonical Correlation Analysis.](http://arxiv.org/abs/2306.16393) | 本文研究了高维正典相关分析（CCA），发现当数据的两个维度无限增长时，传统的CCA估计过程无法提供一致的估计。我们首次证明了在所有维度都很大时无法识别正典变量的CCA过程的不可能性。并提供了估计误差的量级，可用于评估CCA估计的精确性。 |

# 详细

[^1]: 对具有方差轮廓的非独立同分布数据的岭回归进行高维分析

    High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile

    [https://arxiv.org/abs/2403.20200](https://arxiv.org/abs/2403.20200)

    研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。

    

    针对独立但非独立同分布数据，我们提出研究高维回归模型。假设观测到的预测变量集合是带有方差轮廓的随机矩阵，并且其维度以相应速率增长。在假设随机效应模型的情况下，我们研究了具有这种方差轮廓的岭估计器的线性回归的预测风险。在这种设置下，我们提供了该风险的确定性等价物以及岭估计器的自由度。对于某些方差轮廓类别，我们的工作突出了在岭正则化参数趋于零时，高维回归中的最小模最小二乘估计器出现双谷现象。我们还展示了一些方差轮廓f...

    arXiv:2403.20200v1 Announce Type: cross  Abstract: High-dimensional linear regression has been thoroughly studied in the context of independent and identically distributed data. We propose to investigate high-dimensional regression models for independent but non-identically distributed data. To this end, we suppose that the set of observed predictors (or features) is a random matrix with a variance profile and with dimensions growing at a proportional rate. Assuming a random effect model, we study the predictive risk of the ridge estimator for linear regression with such a variance profile. In this setting, we provide deterministic equivalents of this risk and of the degree of freedom of the ridge estimator. For certain class of variance profile, our work highlights the emergence of the well-known double descent phenomenon in high-dimensional regression for the minimum norm least-squares estimator when the ridge regularization parameter goes to zero. We also exhibit variance profiles f
    
[^2]: 具有谱正则化的核拟合优度检验

    Spectral Regularized Kernel Goodness-of-Fit Tests. (arXiv:2308.04561v1 [math.ST])

    [http://arxiv.org/abs/2308.04561](http://arxiv.org/abs/2308.04561)

    本文提出了具有谱正则化的核拟合优度检验方法，用于处理非欧几里得数据。相比之前的方法，本方法在选择适当的正则化参数时能达到最小化最大风险。同时，本方法还克服了之前方法对均值元素为零和积分操作符特征函数均匀有界性条件的限制，并且能够计算更多种类的核函数。

    

    在许多机器学习和统计应用中，最大均值差异(MMD)因其处理非欧几里得数据的能力而获得了很多成功，包括非参数假设检验。最近，Balasubramanian等人(2021)通过实验证明，基于MMD的拟合优度检验在适当选择正则化参数时，并不是最小化最大风险，而其Tikhonov正则化版本则是最小化最大风险的。然而，Balasubramanian等人(2021)的结果是在均值元素为零的限制性假设和积分操作符特征函数的均匀有界性条件下获得的。此外，Balasubramanian等人(2021)提出的检验在许多核函数中是不可计算的，因此不实用。本文解决了这些问题，并将结果推广到包括Tikhonov正则化在内的一般谱正则化方法中。

    Maximum mean discrepancy (MMD) has enjoyed a lot of success in many machine learning and statistical applications, including non-parametric hypothesis testing, because of its ability to handle non-Euclidean data. Recently, it has been demonstrated in Balasubramanian et al.(2021) that the goodness-of-fit test based on MMD is not minimax optimal while a Tikhonov regularized version of it is, for an appropriate choice of the regularization parameter. However, the results in Balasubramanian et al. (2021) are obtained under the restrictive assumptions of the mean element being zero, and the uniform boundedness condition on the eigenfunctions of the integral operator. Moreover, the test proposed in Balasubramanian et al. (2021) is not practical as it is not computable for many kernels. In this paper, we address these shortcomings and extend the results to general spectral regularizers that include Tikhonov regularization.
    
[^3]: 高维正典相关分析

    High-Dimensional Canonical Correlation Analysis. (arXiv:2306.16393v1 [econ.EM])

    [http://arxiv.org/abs/2306.16393](http://arxiv.org/abs/2306.16393)

    本文研究了高维正典相关分析（CCA），发现当数据的两个维度无限增长时，传统的CCA估计过程无法提供一致的估计。我们首次证明了在所有维度都很大时无法识别正典变量的CCA过程的不可能性。并提供了估计误差的量级，可用于评估CCA估计的精确性。

    

    本文研究了高维正典相关分析（CCA），重点关注了定义正典变量的向量。研究表明，当数据的两个维度共同且成比例地增长到无穷时，传统的CCA估计过程无法提供一致的估计。这是首次研究在所有维度都很大时无法识别正典变量的CCA过程的不可能性。为了弥补这个问题，本文推导出了估计误差的大小，可以在实践中用于评估CCA估计的精确度。还提供了将结果应用于石灰草地数据集的示例。

    This paper studies high-dimensional canonical correlation analysis (CCA) with an emphasis on vectors which define canonical variables. The paper shows that when two dimensions of data grow to infinity jointly and proportionally the classical CCA procedure for estimating those vectors fails to deliver a consistent estimate. This provides the first result on impossibility of the identification of canonical variables in CCA procedure when all dimensions are large. To offset, the paper derives the magnitude of the estimation error, which can be used in practice to assess the precision of CCA estimates. An application of the results to limestone grassland data set is provided.
    

