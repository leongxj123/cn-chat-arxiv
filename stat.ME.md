# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modular Learning of Deep Causal Generative Models for High-dimensional Causal Inference.](http://arxiv.org/abs/2401.01426) | 本文提出了一种用于高维因果推断的模块化深度生成模型学习算法，该算法利用预训练的模型来回答由高维数据引起的因果查询。 |
| [^2] | [Covariate Shift Adaptation Robust to Density-Ratio Estimation.](http://arxiv.org/abs/2310.16638) | 该论文研究了在协变量偏移下的密度比估计的罕见问题，提出了一种适应性方法来减轻密度比估计的偏差对模型的影响。 |
| [^3] | [Theoretical guarantees for neural control variates in MCMC.](http://arxiv.org/abs/2304.01111) | 本文提出了一种利用神经控制变量的方差缩减方法，推导并得出了在各种遍历性假设下渐近方差的最优收敛速率。 |

# 详细

[^1]: 高维因果推断的模块化深度生成模型学习

    Modular Learning of Deep Causal Generative Models for High-dimensional Causal Inference. (arXiv:2401.01426v1 [cs.LG])

    [http://arxiv.org/abs/2401.01426](http://arxiv.org/abs/2401.01426)

    本文提出了一种用于高维因果推断的模块化深度生成模型学习算法，该算法利用预训练的模型来回答由高维数据引起的因果查询。

    

    Pearl的因果层次结构在观测、干预和反事实问题之间建立了明确的分离。研究人员提出了计算可辨识因果查询的声音和完整算法，在给定层次的因果结构和数据的情况下使用较低层次的层次的数据。然而，大多数这些算法假设我们可以准确估计数据的概率分布，这对于如图像这样的高维变量是一个不切实际的假设。另一方面，现代生成式深度学习架构可以被训练来学习如何准确地从这样的高维分布中采样。特别是随着图像基模型的最近兴起，利用预训练模型来回答带有这样高维数据的因果查询是非常有吸引力的。为了解决这个问题，我们提出了一个顺序训练算法，给定因果结构和预训练的条件生成模型，可以训练一个模型来估计由高维数据引起的因果关系。

    Pearl's causal hierarchy establishes a clear separation between observational, interventional, and counterfactual questions. Researchers proposed sound and complete algorithms to compute identifiable causal queries at a given level of the hierarchy using the causal structure and data from the lower levels of the hierarchy. However, most of these algorithms assume that we can accurately estimate the probability distribution of the data, which is an impractical assumption for high-dimensional variables such as images. On the other hand, modern generative deep learning architectures can be trained to learn how to accurately sample from such high-dimensional distributions. Especially with the recent rise of foundation models for images, it is desirable to leverage pre-trained models to answer causal queries with such high-dimensional data. To address this, we propose a sequential training algorithm that, given the causal structure and a pre-trained conditional generative model, can train a
    
[^2]: 适应密度比估计的协变量偏移适应

    Covariate Shift Adaptation Robust to Density-Ratio Estimation. (arXiv:2310.16638v1 [stat.ME])

    [http://arxiv.org/abs/2310.16638](http://arxiv.org/abs/2310.16638)

    该论文研究了在协变量偏移下的密度比估计的罕见问题，提出了一种适应性方法来减轻密度比估计的偏差对模型的影响。

    

    在一种情况下，我们可以访问具有协变量和结果的训练数据，而测试数据只包含协变量。在这种情况下，我们的主要目标是预测测试数据中缺失的结果。为了实现这个目标，我们在协变量偏移下训练参数回归模型，其中训练数据和测试数据之间的协变量分布不同。对于这个问题，现有研究提出了通过使用密度比的重要性加权来进行协变量偏移适应的方法。该方法通过对训练数据损失进行加权平均，每个权重是训练数据和测试数据之间的协变量密度比的估计，以近似测试数据的风险。尽管它允许我们获得一个最小化测试数据风险的模型，但其性能严重依赖于密度比估计的准确性。此外，即使密度比可以一致地估计，密度比的估计误差也会导致回归模型的估计器产生偏差。

    Consider a scenario where we have access to train data with both covariates and outcomes while test data only contains covariates. In this scenario, our primary aim is to predict the missing outcomes of the test data. With this objective in mind, we train parametric regression models under a covariate shift, where covariate distributions are different between the train and test data. For this problem, existing studies have proposed covariate shift adaptation via importance weighting using the density ratio. This approach averages the train data losses, each weighted by an estimated ratio of the covariate densities between the train and test data, to approximate the test-data risk. Although it allows us to obtain a test-data risk minimizer, its performance heavily relies on the accuracy of the density ratio estimation. Moreover, even if the density ratio can be consistently estimated, the estimation errors of the density ratio also yield bias in the estimators of the regression model's 
    
[^3]: 神经控制变量在MCMC中的理论保证

    Theoretical guarantees for neural control variates in MCMC. (arXiv:2304.01111v1 [math.ST])

    [http://arxiv.org/abs/2304.01111](http://arxiv.org/abs/2304.01111)

    本文提出了一种利用神经控制变量的方差缩减方法，推导并得出了在各种遍历性假设下渐近方差的最优收敛速率。

    

    本文提出了一种基于加性控制变量和最小化渐近方差的马尔可夫链方差缩减方法。我们专注于控制变量表示为深度神经网络的特定情况。在基础马尔可夫链的各种遍历性假设下，推导了渐近方差的最优收敛速率。该方法依赖于方差缩减算法和函数逼近理论的随机误差的最新成果。

    In this paper, we propose a variance reduction approach for Markov chains based on additive control variates and the minimization of an appropriate estimate for the asymptotic variance. We focus on the particular case when control variates are represented as deep neural networks. We derive the optimal convergence rate of the asymptotic variance under various ergodicity assumptions on the underlying Markov chain. The proposed approach relies upon recent results on the stochastic errors of variance reduction algorithms and function approximation theory.
    

