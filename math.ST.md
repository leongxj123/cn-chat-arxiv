# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Under-Parameterized Double Descent for Ridge Regularized Least Squares Denoising of Data on a Line.](http://arxiv.org/abs/2305.14689) | 本文研究了线性数据最小二乘岭正则化的去噪问题，证明了在欠参数化情况下会出现双峰谷现象。 |

# 详细

[^1]: 基于岭正则化的线性数据最小二乘去噪问题的欠参数化双谷效应

    Under-Parameterized Double Descent for Ridge Regularized Least Squares Denoising of Data on a Line. (arXiv:2305.14689v1 [stat.ML])

    [http://arxiv.org/abs/2305.14689](http://arxiv.org/abs/2305.14689)

    本文研究了线性数据最小二乘岭正则化的去噪问题，证明了在欠参数化情况下会出现双峰谷现象。

    

    研究了训练数据点数、统计模型参数数和模型的泛化能力之间的关系。已有的工作表明，过度参数化情况下可能出现双峰谷现象，而在欠参数化情况下则普遍存在标准偏差-方差权衡。本文提出了一个简单的例子，可以证明欠参数化情况下可以发生双峰谷现象。考虑嵌入高维空间中的线性数据最小二乘去噪问题中的岭正则化，通过推导出一种渐近准确的广义误差公式，我们发现了样本和参数的双谷效应，双峰谷位于插值点和过度参数化区域之间。此外，样本双谷曲线的高峰对应于估计量的范数曲线的高峰。

    The relationship between the number of training data points, the number of parameters in a statistical model, and the generalization capabilities of the model has been widely studied. Previous work has shown that double descent can occur in the over-parameterized regime, and believe that the standard bias-variance trade-off holds in the under-parameterized regime. In this paper, we present a simple example that provably exhibits double descent in the under-parameterized regime. For simplicity, we look at the ridge regularized least squares denoising problem with data on a line embedded in high-dimension space. By deriving an asymptotically accurate formula for the generalization error, we observe sample-wise and parameter-wise double descent with the peak in the under-parameterized regime rather than at the interpolation point or in the over-parameterized regime.  Further, the peak of the sample-wise double descent curve corresponds to a peak in the curve for the norm of the estimator,
    

