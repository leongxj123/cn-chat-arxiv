# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions.](http://arxiv.org/abs/2305.08175) | ResidualPlanner是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展，可以优化许多可以写成边际方差的凸函数的损失函数。 |

# 详细

[^1]: 一种优化且可扩展的矩阵机制用于扰动边缘数据下凸损失函数。

    An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions. (arXiv:2305.08175v1 [cs.DB])

    [http://arxiv.org/abs/2305.08175](http://arxiv.org/abs/2305.08175)

    ResidualPlanner是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展，可以优化许多可以写成边际方差的凸函数的损失函数。

    

    扰动的边缘数据是一种常见的保护数据隐私的形式，可用于诸如列联表分析、贝叶斯网络构建和合成数据生成等下游任务。我们提出了ResidualPlanner，这是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展。ResidualPlanner可以优化许多可以写成边际方差的凸函数的损失函数。此外，ResidualPlanner可以在几秒钟内优化大规模设置中的边缘准确性，即使之前的最先进技术（HDMM）也会占用过多的内存。甚至在具有100个属性的数据集上也可以在几分钟内运行。此外，ResidualPlanner还可以有效地计算每个边缘的方差/协方差值（之前的方法会很快失败）。

    Noisy marginals are a common form of confidentiality-protecting data release and are useful for many downstream tasks such as contingency table analysis, construction of Bayesian networks, and even synthetic data generation. Privacy mechanisms that provide unbiased noisy answers to linear queries (such as marginals) are known as matrix mechanisms.  We propose ResidualPlanner, a matrix mechanism for marginals with Gaussian noise that is both optimal and scalable. ResidualPlanner can optimize for many loss functions that can be written as a convex function of marginal variances (prior work was restricted to just one predefined objective function). ResidualPlanner can optimize the accuracy of marginals in large scale settings in seconds, even when the previous state of the art (HDMM) runs out of memory. It even runs on datasets with 100 attributes in a couple of minutes. Furthermore ResidualPlanner can efficiently compute variance/covariance values for each marginal (prior methods quickly
    

