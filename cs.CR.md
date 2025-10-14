# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-aware Gaussian Process Regression.](http://arxiv.org/abs/2305.16541) | 以高斯过程回归为基础，我们提出了实现数据隐私保护的方法。方法将综合噪声添加到数据中，使得高斯过程预测模型达到特定的隐私级别。我们还通过核方法介绍了连续隐私约束下的隐私感知解形式，以及研究了其理论性质。所提出的方法应用于卫星轨迹跟踪模型。 |

# 详细

[^1]: 面向隐私的高斯过程回归

    Privacy-aware Gaussian Process Regression. (arXiv:2305.16541v1 [cs.LG])

    [http://arxiv.org/abs/2305.16541](http://arxiv.org/abs/2305.16541)

    以高斯过程回归为基础，我们提出了实现数据隐私保护的方法。方法将综合噪声添加到数据中，使得高斯过程预测模型达到特定的隐私级别。我们还通过核方法介绍了连续隐私约束下的隐私感知解形式，以及研究了其理论性质。所提出的方法应用于卫星轨迹跟踪模型。

    

    我们提出了第一个在隐私约束条件下的高斯过程回归的理论和方法框架。所提出的方法可以在数据所有者因隐私担忧而不愿与公众分享其从其数据构建的高保真监督学习模型时使用。所提出方法的关键思想是通过添加综合噪声来使高斯过程预测模型的预测方差达到预先指定的隐私级别。合成噪声的最优协方差矩阵以半定编程的形式给出。我们还介绍了基于核的方法来研究在连续约束隐私条件下的隐私感知解的形式，并研究了它们的理论属性。所提出的方法使用跟踪卫星轨迹的模型进行了说明。

    We propose the first theoretical and methodological framework for Gaussian process regression subject to privacy constraints. The proposed method can be used when a data owner is unwilling to share a high-fidelity supervised learning model built from their data with the public due to privacy concerns. The key idea of the proposed method is to add synthetic noise to the data until the predictive variance of the Gaussian process model reaches a prespecified privacy level. The optimal covariance matrix of the synthetic noise is formulated in terms of semi-definite programming. We also introduce the formulation of privacy-aware solutions under continuous privacy constraints using kernel-based approaches, and study their theoretical properties. The proposed method is illustrated by considering a model that tracks the trajectories of satellites.
    

