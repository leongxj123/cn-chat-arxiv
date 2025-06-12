# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simulation-based inference using surjective sequential neural likelihood estimation.](http://arxiv.org/abs/2308.01054) | 我们提出了一种使用全射序列神经似然估计（SSNL）进行基于仿真的推断的新方法，在模型中无法计算似然函数并且只能使用模拟器生成数据的情况下，SSNL通过拟合降维的全射归一化流模型，并将其作为替代似然函数，解决了先前基于似然方法在高维数据集中遇到的问题，并在各种实验中展示了其优越性能。 |

# 详细

[^1]: 使用全射序列神经似然估计进行基于仿真的推断

    Simulation-based inference using surjective sequential neural likelihood estimation. (arXiv:2308.01054v1 [stat.ML])

    [http://arxiv.org/abs/2308.01054](http://arxiv.org/abs/2308.01054)

    我们提出了一种使用全射序列神经似然估计（SSNL）进行基于仿真的推断的新方法，在模型中无法计算似然函数并且只能使用模拟器生成数据的情况下，SSNL通过拟合降维的全射归一化流模型，并将其作为替代似然函数，解决了先前基于似然方法在高维数据集中遇到的问题，并在各种实验中展示了其优越性能。

    

    我们提出了全射序列神经似然（SSNL）估计方法，这是一种在模型中无法计算似然函数并且只能使用可以生成合成数据的模拟器时进行基于仿真的推断的新方法。SSNL拟合一个降维的全射归一化流模型，并将其用作替代似然函数，从而可以使用传统的贝叶斯推断方法，包括马尔科夫链蒙特卡罗方法或变分推断。通过将数据嵌入到低维空间中，SSNL解决了先前基于似然方法在应用于高维数据集时遇到的几个问题，例如包含无信息数据维度或位于较低维流形上的数据。我们对SSNL在各种实验中进行了评估，并表明它通常优于在基于仿真推断中使用的现代方法，例如在一项来自天体物理学的具有挑战性的真实世界例子上对磁场模型的建模。

    We present Surjective Sequential Neural Likelihood (SSNL) estimation, a novel method for simulation-based inference in models where the evaluation of the likelihood function is not tractable and only a simulator that can generate synthetic data is available. SSNL fits a dimensionality-reducing surjective normalizing flow model and uses it as a surrogate likelihood function which allows for conventional Bayesian inference using either Markov chain Monte Carlo methods or variational inference. By embedding the data in a low-dimensional space, SSNL solves several issues previous likelihood-based methods had when applied to high-dimensional data sets that, for instance, contain non-informative data dimensions or lie along a lower-dimensional manifold. We evaluate SSNL on a wide variety of experiments and show that it generally outperforms contemporary methods used in simulation-based inference, for instance, on a challenging real-world example from astrophysics which models the magnetic fi
    

