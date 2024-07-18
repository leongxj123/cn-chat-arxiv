# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linear and nonlinear system identification under $\ell_1$- and group-Lasso regularization via L-BFGS-B](https://arxiv.org/abs/2403.03827) | 本文提出了一种基于L-BFGS-B算法的方法，可用于在$\ell_1$和group-Lasso正则化下识别线性和非线性系统，相比传统线性子空间方法，该方法在结果、损失和正则化项使用的通用性以及数值稳定性方面通常提供更好的表现，并且可以广泛应用于各种参数化非线性状态空间模型的识别。 |
| [^2] | [Learning-assisted Stochastic Capacity Expansion Planning: A Bayesian Optimization Approach.](http://arxiv.org/abs/2401.10451) | 本研究提出了一种学习辅助的贝叶斯优化方法，用于解决大规模容量扩展问题。通过构建和求解可行的时间聚合代理问题，识别出低成本的规划决策。通过在验证集和测试预测上评估解决的规划结果，实现了随机容量扩展问题的可行解决。 |

# 详细

[^1]: 在L-BFGS-B算法的$\ell_1$和group-Lasso正则化下进行线性和非线性系统识别

    Linear and nonlinear system identification under $\ell_1$- and group-Lasso regularization via L-BFGS-B

    [https://arxiv.org/abs/2403.03827](https://arxiv.org/abs/2403.03827)

    本文提出了一种基于L-BFGS-B算法的方法，可用于在$\ell_1$和group-Lasso正则化下识别线性和非线性系统，相比传统线性子空间方法，该方法在结果、损失和正则化项使用的通用性以及数值稳定性方面通常提供更好的表现，并且可以广泛应用于各种参数化非线性状态空间模型的识别。

    

    在本文中，我们提出了一种基于L-BFGS-B算法的方法，用于识别可能在$\ell_1$和group-Lasso正则化下的线性和非线性离散时间状态空间模型。针对线性模型的识别，我们展示了与经典线性子空间方法相比，该方法通常提供更好的结果，在损失和正则化项的使用方面更加通用，也在数值上更加稳定。该方法不仅丰富了现有的线性系统识别工具集，还可以应用于识别包括循环神经网络在内的非常广泛的参数化非线性状态空间模型。我们在合成和实验数据集上演示了该方法，并将其应用于解决Weigand等人（2022年）提出的具有挑战性的工业机器人基准的非线性多输入/多输出系统识别。

    arXiv:2403.03827v1 Announce Type: cross  Abstract: In this paper, we propose an approach for identifying linear and nonlinear discrete-time state-space models, possibly under $\ell_1$- and group-Lasso regularization, based on the L-BFGS-B algorithm. For the identification of linear models, we show that, compared to classical linear subspace methods, the approach often provides better results, is much more general in terms of the loss and regularization terms used, and is also more stable from a numerical point of view. The proposed method not only enriches the existing set of linear system identification tools but can be also applied to identifying a very broad class of parametric nonlinear state-space models, including recurrent neural networks. We illustrate the approach on synthetic and experimental datasets and apply it to solve the challenging industrial robot benchmark for nonlinear multi-input/multi-output system identification proposed by Weigand et al. (2022). A Python impleme
    
[^2]: 学习辅助的随机容量扩展规划：一种贝叶斯优化方法

    Learning-assisted Stochastic Capacity Expansion Planning: A Bayesian Optimization Approach. (arXiv:2401.10451v1 [eess.SY])

    [http://arxiv.org/abs/2401.10451](http://arxiv.org/abs/2401.10451)

    本研究提出了一种学习辅助的贝叶斯优化方法，用于解决大规模容量扩展问题。通过构建和求解可行的时间聚合代理问题，识别出低成本的规划决策。通过在验证集和测试预测上评估解决的规划结果，实现了随机容量扩展问题的可行解决。

    

    解决大规模的容量扩展问题对于区域能源系统的成本效益低碳化至关重要。为了确保容量扩展问题的预期结果，建模考虑到天气相关的可再生能源供应和能源需求的不确定性变得至关重要。然而，由此产生的随机优化模型通常比确定性模型难以计算。在这里，我们提出了一种学习辅助的近似解法来可行地解决两阶段随机容量扩展问题。我们的方法通过构建和求解一系列可行的时间聚合代理问题，识别出低成本的规划决策。我们采用贝叶斯优化方法搜索时间序列聚合超参数的空间，并计算在供需预测的验证集上最小化成本的近似解。重要的是，我们在一组保留的测试预测上评估解决的规划结果。

    Solving large-scale capacity expansion problems (CEPs) is central to cost-effective decarbonization of regional-scale energy systems. To ensure the intended outcomes of CEPs, modeling uncertainty due to weather-dependent variable renewable energy (VRE) supply and energy demand becomes crucially important. However, the resulting stochastic optimization models are often less computationally tractable than their deterministic counterparts. Here, we propose a learning-assisted approximate solution method to tractably solve two-stage stochastic CEPs. Our method identifies low-cost planning decisions by constructing and solving a sequence of tractable temporally aggregated surrogate problems. We adopt a Bayesian optimization approach to searching the space of time series aggregation hyperparameters and compute approximate solutions that minimize costs on a validation set of supply-demand projections. Importantly, we evaluate solved planning outcomes on a held-out set of test projections. We 
    

