# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [$L^1$ Estimation: On the Optimality of Linear Estimators.](http://arxiv.org/abs/2309.09129) | 该论文研究了在$L^1$保真度条件下，从噪声观测中估计随机变量$X$的问题。结果表明，唯一能够引入线性条件中位数的先验分布是高斯分布。此外，还研究了其他$L^p$损失，并观察到对于$p \in [1,2]$，高斯分布是唯一引入线性最优贝叶斯估计器的先验分布。扩展还涵盖了特定指数族条件分布的噪声模型。 |
| [^2] | [Nonparametric Linear Feature Learning in Regression Through Regularisation.](http://arxiv.org/abs/2307.12754) | 本研究提出了一种新的非参数线性特征学习方法，对于监督学习中存在于低维线性子空间中的相关信息的预测和解释能力的提升是非常有帮助的。 |

# 详细

[^1]: $L^1$估计：线性估计器的最优性

    $L^1$ Estimation: On the Optimality of Linear Estimators. (arXiv:2309.09129v1 [math.ST])

    [http://arxiv.org/abs/2309.09129](http://arxiv.org/abs/2309.09129)

    该论文研究了在$L^1$保真度条件下，从噪声观测中估计随机变量$X$的问题。结果表明，唯一能够引入线性条件中位数的先验分布是高斯分布。此外，还研究了其他$L^p$损失，并观察到对于$p \in [1,2]$，高斯分布是唯一引入线性最优贝叶斯估计器的先验分布。扩展还涵盖了特定指数族条件分布的噪声模型。

    

    在$L^1$保真度条件下，考虑从噪声观测$Y=X+Z$中估计随机变量$X$的问题，其中$Z$是标准正态分布。众所周知，在这种情况下，最优的贝叶斯估计器是条件中位数。本文表明，在条件中位数中引入线性的唯一先验分布是高斯分布。同时，还提供了其他几个结果。特别地，证明了如果对于所有$y$，条件分布$P_{X|Y=y}$都是对称的，则$X$必须服从高斯分布。此外，我们考虑了其他的$L^p$损失，并观察到以下现象：对于$p \in [1,2]$，高斯分布是唯一引入线性最优贝叶斯估计器的先验分布，对于$p \in (2,\infty)$，有无穷多个先验分布可以引入线性性。最后，还提供了扩展，以涵盖导致特定指数族条件分布的噪声模型。

    Consider the problem of estimating a random variable $X$ from noisy observations $Y = X+ Z$, where $Z$ is standard normal, under the $L^1$ fidelity criterion. It is well known that the optimal Bayesian estimator in this setting is the conditional median. This work shows that the only prior distribution on $X$ that induces linearity in the conditional median is Gaussian.  Along the way, several other results are presented. In particular, it is demonstrated that if the conditional distribution $P_{X|Y=y}$ is symmetric for all $y$, then $X$ must follow a Gaussian distribution. Additionally, we consider other $L^p$ losses and observe the following phenomenon: for $p \in [1,2]$, Gaussian is the only prior distribution that induces a linear optimal Bayesian estimator, and for $p \in (2,\infty)$, infinitely many prior distributions on $X$ can induce linearity. Finally, extensions are provided to encompass noise models leading to conditional distributions from certain exponential families.
    
[^2]: 非参数线性特征学习在回归中的应用通过正则化

    Nonparametric Linear Feature Learning in Regression Through Regularisation. (arXiv:2307.12754v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2307.12754](http://arxiv.org/abs/2307.12754)

    本研究提出了一种新的非参数线性特征学习方法，对于监督学习中存在于低维线性子空间中的相关信息的预测和解释能力的提升是非常有帮助的。

    

    表征学习在自动化特征选择中发挥着关键作用，特别是在高维数据的背景下，非参数方法常常很难应对。在本研究中，我们专注于监督学习场景，其中相关信息存在于数据的低维线性子空间中，即多指数模型。如果已知该子空间，将大大增强预测、计算和解释能力。为了解决这一挑战，我们提出了一种新颖的非参数预测的线性特征学习方法，同时估计预测函数和线性子空间。我们的方法采用经验风险最小化，并加上函数导数的惩罚项，以保证其多样性。通过利用Hermite多项式的正交性和旋转不变性特性，我们引入了我们的估计器RegFeaL。通过利用替代最小化，我们迭代地旋转数据以改善与线性子空间的对齐。

    Representation learning plays a crucial role in automated feature selection, particularly in the context of high-dimensional data, where non-parametric methods often struggle. In this study, we focus on supervised learning scenarios where the pertinent information resides within a lower-dimensional linear subspace of the data, namely the multi-index model. If this subspace were known, it would greatly enhance prediction, computation, and interpretation. To address this challenge, we propose a novel method for linear feature learning with non-parametric prediction, which simultaneously estimates the prediction function and the linear subspace. Our approach employs empirical risk minimisation, augmented with a penalty on function derivatives, ensuring versatility. Leveraging the orthogonality and rotation invariance properties of Hermite polynomials, we introduce our estimator, named RegFeaL. By utilising alternative minimisation, we iteratively rotate the data to improve alignment with 
    

