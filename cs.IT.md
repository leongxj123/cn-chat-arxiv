# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Fidelity Bayesian Optimization With Across-Task Transferable Max-Value Entropy Search](https://arxiv.org/abs/2403.09570) | 本文引入了一种新颖的信息理论获取函数，用于平衡在连续的优化任务中获得最优值或解信息的需求。 |
| [^2] | [Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression.](http://arxiv.org/abs/2309.07810) | 这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。 |
| [^3] | [Batches Stabilize the Minimum Norm Risk in High Dimensional Overparameterized Linear Regression.](http://arxiv.org/abs/2306.08432) | 本文研究了将数据分成批次的学习算法，在高维超参数线性回归模型中提供了隐式正则化，通过适当的批量大小选择，稳定了风险行为，消除了插值点处的膨胀和双峰现象 |

# 详细

[^1]: 基于多保真度的贝叶斯优化方法及跨任务可转移的最大值熵搜索

    Multi-Fidelity Bayesian Optimization With Across-Task Transferable Max-Value Entropy Search

    [https://arxiv.org/abs/2403.09570](https://arxiv.org/abs/2403.09570)

    本文引入了一种新颖的信息理论获取函数，用于平衡在连续的优化任务中获得最优值或解信息的需求。

    

    在许多应用中，设计者面临一系列优化任务，任务的目标是昂贵评估的黑盒函数形式。本文介绍了一种新的信息理论获取函数，用于平衡需要获取不同任务的最优值或解的信息和通过参数的转移传递。

    arXiv:2403.09570v1 Announce Type: new  Abstract: In many applications, ranging from logistics to engineering, a designer is faced with a sequence of optimization tasks for which the objectives are in the form of black-box functions that are costly to evaluate. For example, the designer may need to tune the hyperparameters of neural network models for different learning tasks over time. Rather than evaluating the objective function for each candidate solution, the designer may have access to approximations of the objective functions, for which higher-fidelity evaluations entail a larger cost. Existing multi-fidelity black-box optimization strategies select candidate solutions and fidelity levels with the goal of maximizing the information accrued about the optimal value or solution for the current task. Assuming that successive optimization tasks are related, this paper introduces a novel information-theoretic acquisition function that balances the need to acquire information about the 
    
[^2]: Spectrum-Aware Adjustment: 一种新的去偏方法框架及其在主成分回归中的应用

    Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression. (arXiv:2309.07810v1 [math.ST])

    [http://arxiv.org/abs/2309.07810](http://arxiv.org/abs/2309.07810)

    这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。

    

    我们引入了一个新的去偏方法框架，用于解决高维线性回归中现代去偏技术对协变量分布的约束问题。我们研究了特征数和样本数都很大且相近的普遍情况。在这种情况下，现代去偏技术使用自由度校正来除去正则化估计量的收缩偏差并进行推断。然而，该方法要求观测样本是独立同分布的，协变量遵循均值为零的高斯分布，并且能够获得可靠的特征协方差矩阵估计。当（i）协变量具有非高斯分布、重尾或非对称分布，（ii）设计矩阵的行呈异质性或存在依赖性，以及（iii）缺乏可靠的特征协方差估计时，这种方法就会遇到困难。为了应对这些问题，我们提出了一种新的策略，其中去偏校正是一步缩放的梯度下降步骤（适当缩放）。

    We introduce a new debiasing framework for high-dimensional linear regression that bypasses the restrictions on covariate distributions imposed by modern debiasing technology. We study the prevalent setting where the number of features and samples are both large and comparable. In this context, state-of-the-art debiasing technology uses a degrees-of-freedom correction to remove shrinkage bias of regularized estimators and conduct inference. However, this method requires that the observed samples are i.i.d., the covariates follow a mean zero Gaussian distribution, and reliable covariance matrix estimates for observed features are available. This approach struggles when (i) covariates are non-Gaussian with heavy tails or asymmetric distributions, (ii) rows of the design exhibit heterogeneity or dependencies, and (iii) reliable feature covariance estimates are lacking.  To address these, we develop a new strategy where the debiasing correction is a rescaled gradient descent step (suitably
    
[^3]: 批次使高维超参数线性回归的最小规范风险稳定

    Batches Stabilize the Minimum Norm Risk in High Dimensional Overparameterized Linear Regression. (arXiv:2306.08432v1 [cs.LG])

    [http://arxiv.org/abs/2306.08432](http://arxiv.org/abs/2306.08432)

    本文研究了将数据分成批次的学习算法，在高维超参数线性回归模型中提供了隐式正则化，通过适当的批量大小选择，稳定了风险行为，消除了插值点处的膨胀和双峰现象

    

    将数据分成批次的学习算法在许多机器学习应用中很常见，通常在计算效率和性能之间提供有用的权衡。本文通过具有各向同性高斯特征的最小规范超参数线性回归模型的视角来研究批量分区的好处。我们建议最小规范估计量的自然小批量版本，并推导出其二次风险的上界，表明其与噪声水平以及过度参数化比例成反比，对于最佳批量大小的选择。与最小规范相比，我们的估计器具有稳定的风险行为，其在过度参数化比例上单调递增，消除了插值点处的膨胀和双峰现象。有趣的是，我们观察到批处理所提供的隐式正则化在一定程度上可以通过特征重叠来解释。

    Learning algorithms that divide the data into batches are prevalent in many machine-learning applications, typically offering useful trade-offs between computational efficiency and performance. In this paper, we examine the benefits of batch-partitioning through the lens of a minimum-norm overparameterized linear regression model with isotropic Gaussian features. We suggest a natural small-batch version of the minimum-norm estimator, and derive an upper bound on its quadratic risk, showing it is inversely proportional to the noise level as well as to the overparameterization ratio, for the optimal choice of batch size. In contrast to minimum-norm, our estimator admits a stable risk behavior that is monotonically increasing in the overparameterization ratio, eliminating both the blowup at the interpolation point and the double-descent phenomenon. Interestingly, we observe that this implicit regularization offered by the batch partition is partially explained by feature overlap between t
    

