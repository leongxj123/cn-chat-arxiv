# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interacting Particle Systems on Networks: joint inference of the network and the interaction kernel](https://arxiv.org/abs/2402.08412) | 本文研究了在网络上建模多智体系统的方法，提出了联合推断网络的权重矩阵和相互作用核的估计器，通过解决非凸优化问题并使用交替最小二乘（ALS）算法和交替最小二乘算子回归（ORALS）算法进行求解。在保证可识别性和良定义性的条件下，ALS算法表现出统计效率和鲁棒性，而ORALS算法是一致的，并且在渐近情况下具有正态性。 |
| [^2] | [On Consistency of Signatures Using Lasso.](http://arxiv.org/abs/2305.10413) | 本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。 |
| [^3] | [Theory of Posterior Concentration for Generalized Bayesian Additive Regression Trees.](http://arxiv.org/abs/2304.12505) | 本论文提出了一个广义的贝叶斯树及其加性集成的框架，包括大多数BART的变体，并提出响应分布的充分条件，对BART及其变体的实证成功提供了理论支持。 |

# 详细

[^1]: 在网络上相互作用的粒子系统: 网络和相互作用核的联合推断

    Interacting Particle Systems on Networks: joint inference of the network and the interaction kernel

    [https://arxiv.org/abs/2402.08412](https://arxiv.org/abs/2402.08412)

    本文研究了在网络上建模多智体系统的方法，提出了联合推断网络的权重矩阵和相互作用核的估计器，通过解决非凸优化问题并使用交替最小二乘（ALS）算法和交替最小二乘算子回归（ORALS）算法进行求解。在保证可识别性和良定义性的条件下，ALS算法表现出统计效率和鲁棒性，而ORALS算法是一致的，并且在渐近情况下具有正态性。

    

    在各种学科中，对网络上的多智体系统进行建模是一个基本的挑战。我们从由多条轨迹组成的数据中联合推断网络的权重矩阵和相互作用核，分别确定哪些智体与哪些其他智体相互作用以及这种相互作用的规则。我们提出的估计器自然地导致一个非凸优化问题，并研究了两种解决方案：一种基于交替最小二乘（ALS）算法，另一种基于一种名为交替最小二乘的算子回归（ORALS）的新算法。这两种算法都可扩展到大量数据轨迹。我们建立了保证可识别性和良定义性的强制性条件。尽管ALS算法在小数据情况下缺乏性能和收敛性保证，但表现出统计效率和鲁棒性。在强制性条件下，ORALS估计器是一致的，并且在渐近情况下具有正态性。

    Modeling multi-agent systems on networks is a fundamental challenge in a wide variety of disciplines. We jointly infer the weight matrix of the network and the interaction kernel, which determine respectively which agents interact with which others and the rules of such interactions from data consisting of multiple trajectories. The estimator we propose leads naturally to a non-convex optimization problem, and we investigate two approaches for its solution: one is based on the alternating least squares (ALS) algorithm; another is based on a new algorithm named operator regression with alternating least squares (ORALS). Both algorithms are scalable to large ensembles of data trajectories. We establish coercivity conditions guaranteeing identifiability and well-posedness. The ALS algorithm appears statistically efficient and robust even in the small data regime but lacks performance and convergence guarantees. The ORALS estimator is consistent and asymptotically normal under a coercivity
    
[^2]: 使用Lasso的签名一致性研究

    On Consistency of Signatures Using Lasso. (arXiv:2305.10413v1 [stat.ML])

    [http://arxiv.org/abs/2305.10413](http://arxiv.org/abs/2305.10413)

    本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。

    

    签名变换是连续和离散时间序列数据的迭代路径积分，它们的普遍非线性通过线性化特征选择问题。本文在理论和数值上重新审视了Lasso回归对于签名变换的一致性问题。我们的研究表明，对于更接近布朗运动或具有较弱跨维度相关性的过程和时间序列，签名定义为It\^o积分的Lasso回归更具一致性；对于均值回归过程和时间序列，其签名定义为Stratonovich积分在Lasso回归中具有更高的一致性。我们的发现强调了在统计推断和机器学习中选择适当的签名和随机模型的重要性。

    Signature transforms are iterated path integrals of continuous and discrete-time time series data, and their universal nonlinearity linearizes the problem of feature selection. This paper revisits the consistency issue of Lasso regression for the signature transform, both theoretically and numerically. Our study shows that, for processes and time series that are closer to Brownian motion or random walk with weaker inter-dimensional correlations, the Lasso regression is more consistent for their signatures defined by It\^o integrals; for mean reverting processes and time series, their signatures defined by Stratonovich integrals have more consistency in the Lasso regression. Our findings highlight the importance of choosing appropriate definitions of signatures and stochastic models in statistical inference and machine learning.
    
[^3]: 广义贝叶斯加性回归树的后验集中理论

    Theory of Posterior Concentration for Generalized Bayesian Additive Regression Trees. (arXiv:2304.12505v1 [math.ST])

    [http://arxiv.org/abs/2304.12505](http://arxiv.org/abs/2304.12505)

    本论文提出了一个广义的贝叶斯树及其加性集成的框架，包括大多数BART的变体，并提出响应分布的充分条件，对BART及其变体的实证成功提供了理论支持。

    

    贝叶斯加性回归树（BART）是一种强大的半参数集成学习技术，用于建模非线性回归函数。虽然最初BART仅用于预测连续和二元响应变量，但多年来已经出现了多种扩展，适用于估计更广泛的响应变量（例如分类和计数数据），并且可以应用于很多领域。在本文中，我们描述了一个广义贝叶斯树及其加性集成的框架，其中响应变量来自指数族分布，因此包括BART的大多数变体。 我们推导出响应分布的充分条件，在此条件下，后验以最小化速率集中，最多以对数因子为限。在这方面，我们的结果为BART及其变体的实证成功提供了理论依据。

    Bayesian Additive Regression Trees (BART) are a powerful semiparametric ensemble learning technique for modeling nonlinear regression functions. Although initially BART was proposed for predicting only continuous and binary response variables, over the years multiple extensions have emerged that are suitable for estimating a wider class of response variables (e.g. categorical and count data) in a multitude of application areas. In this paper we describe a Generalized framework for Bayesian trees and their additive ensembles where the response variable comes from an exponential family distribution and hence encompasses a majority of these variants of BART. We derive sufficient conditions on the response distribution, under which the posterior concentrates at a minimax rate, up to a logarithmic factor. In this regard our results provide theoretical justification for the empirical success of BART and its variants.
    

