# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Agnostic Regression: a machine learning method to validate regression models](https://arxiv.org/abs/2402.15213) | 本文提出了一种新的方法，统计无关地评估了线性回归模型，并评估了ML估计在检测方面的表现。 |
| [^2] | [Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits](https://arxiv.org/abs/2402.15171) | 提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。 |
| [^3] | [Thresholded Oja does Sparse PCA?](https://arxiv.org/abs/2402.07240) | 阈值和重新归一化Oja算法的输出可获得一个接近最优的错误率，与未经阈值处理的Oja向量相比，这大大减小了误差。 |

# 详细

[^1]: 统计无偏回归：一种用于验证回归模型的机器学习方法

    Statistical Agnostic Regression: a machine learning method to validate regression models

    [https://arxiv.org/abs/2402.15213](https://arxiv.org/abs/2402.15213)

    本文提出了一种新的方法，统计无关地评估了线性回归模型，并评估了ML估计在检测方面的表现。

    

    回归分析是统计建模中的一个核心主题，旨在估计因变量（通常称为响应变量）与一个或多个自变量（即解释变量）之间的关系。线性回归是迄今为止在预测、预测或因果推断等多个研究领域执行此任务的最流行方法。除了解决线性回归问题的各种传统方法外，如普通最小二乘法、岭回归或套索回归——这些方法往往是更高级机器学习（ML）技术的基础——后者已成功地应用在这种场景中，但没有对统计显著性进行正式定义。最多，基于经验测量（如残差或准确度）进行置换或基于经典分析，以反映ML估计对检测的更高能力。本文介绍了一种新的方法，该方法统计无关地评估了线性回归模型，并对ML估计在检测方面的表现进行了评估。

    arXiv:2402.15213v1 Announce Type: cross  Abstract: Regression analysis is a central topic in statistical modeling, aiming to estimate the relationships between a dependent variable, commonly referred to as the response variable, and one or more independent variables, i.e., explanatory variables. Linear regression is by far the most popular method for performing this task in several fields of research, such as prediction, forecasting, or causal inference. Beyond various classical methods to solve linear regression problems, such as Ordinary Least Squares, Ridge, or Lasso regressions - which are often the foundation for more advanced machine learning (ML) techniques - the latter have been successfully applied in this scenario without a formal definition of statistical significance. At most, permutation or classical analyses based on empirical measures (e.g., residuals or accuracy) have been conducted to reflect the greater ability of ML estimations for detection. In this paper, we introd
    
[^2]: 用于随机组合半臂老虎机的协方差自适应最小二乘算法

    Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits

    [https://arxiv.org/abs/2402.15171](https://arxiv.org/abs/2402.15171)

    提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。

    

    我们解决了随机组合半臂老虎机问题，其中玩家可以从包含d个基本项的P个子集中进行选择。大多数现有算法（如CUCB、ESCB、OLS-UCB）需要对奖励分布有先验知识，比如子高斯代理-方差的上界，这很难准确估计。在这项工作中，我们设计了OLS-UCB的方差自适应版本，依赖于协方差结构的在线估计。在实际设置中，估计协方差矩阵的系数要容易得多，并且相对于基于代理方差的算法，导致改进的遗憾上界。当协方差系数全为非负时，我们展示了我们的方法有效地利用了半臂反馈，并且可以明显优于老虎机反馈方法，在指数级别P≫d以及P≤d的情况下，这一点并不来自大多数现有分析。

    arXiv:2402.15171v1 Announce Type: new  Abstract: We address the problem of stochastic combinatorial semi-bandits, where a player can select from P subsets of a set containing d base items. Most existing algorithms (e.g. CUCB, ESCB, OLS-UCB) require prior knowledge on the reward distribution, like an upper bound on a sub-Gaussian proxy-variance, which is hard to estimate tightly. In this work, we design a variance-adaptive version of OLS-UCB, relying on an online estimation of the covariance structure. Estimating the coefficients of a covariance matrix is much more manageable in practical settings and results in improved regret upper bounds compared to proxy variance-based algorithms. When covariance coefficients are all non-negative, we show that our approach efficiently leverages the semi-bandit feedback and provably outperforms bandit feedback approaches, not only in exponential regimes where P $\gg$ d but also when P $\le$ d, which is not straightforward from most existing analyses.
    
[^3]: 阈值Oja是否适用于稀疏PCA？

    Thresholded Oja does Sparse PCA?

    [https://arxiv.org/abs/2402.07240](https://arxiv.org/abs/2402.07240)

    阈值和重新归一化Oja算法的输出可获得一个接近最优的错误率，与未经阈值处理的Oja向量相比，这大大减小了误差。

    

    我们考虑了当比值$d/n \rightarrow c > 0$时稀疏主成分分析（PCA）的问题。在离线设置下，关于稀疏PCA的最优率已经有很多研究，其中所有数据都可以用于多次传递。相比之下，当人口特征向量是$s$-稀疏时，具有$O(d)$存储和$O(nd)$时间复杂度的流算法通常要求强初始化条件，否则会有次优错误。我们展示了一种简单的算法，对Oja算法的输出（Oja向量）进行阈值和重新归一化，从而获得接近最优的错误率。这非常令人惊讶，因为没有阈值，Oja向量的误差很大。我们的分析集中在限制未归一化的Oja向量的项上，这涉及将一组独立随机矩阵的乘积在随机初始向量上的投影。 这是非平凡且新颖的，因为以前的Oja算法分析没有考虑这一点。

    arXiv:2402.07240v2 Announce Type: cross  Abstract: We consider the problem of Sparse Principal Component Analysis (PCA) when the ratio $d/n \rightarrow c > 0$. There has been a lot of work on optimal rates on sparse PCA in the offline setting, where all the data is available for multiple passes. In contrast, when the population eigenvector is $s$-sparse, streaming algorithms that have $O(d)$ storage and $O(nd)$ time complexity either typically require strong initialization conditions or have a suboptimal error. We show that a simple algorithm that thresholds and renormalizes the output of Oja's algorithm (the Oja vector) obtains a near-optimal error rate. This is very surprising because, without thresholding, the Oja vector has a large error. Our analysis centers around bounding the entries of the unnormalized Oja vector, which involves the projection of a product of independent random matrices on a random initial vector. This is nontrivial and novel since previous analyses of Oja's al
    

