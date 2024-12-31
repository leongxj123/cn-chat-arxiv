# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric Regression under Cluster Sampling](https://arxiv.org/abs/2403.04766) | 本文在簇相关性存在的情况下为非参数核回归模型发展了一般渐近理论，并提出了有效的带宽选择和推断方法，引入了渐近方差的估计量，并验证了集群稳健带宽选择的有效性。 |
| [^2] | [The Fragility of Sparsity.](http://arxiv.org/abs/2311.02299) | 稀疏性的线性回归估计在选择回归矩阵和假设检验上存在脆弱性，OLS能够提供更健壮的结果而效率损失较小。 |
| [^3] | [Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods.](http://arxiv.org/abs/2305.04634) | 研究提出了使用CNN学习空间过程的似然函数。即使在没有确切似然函数的情况下，通过分类任务进行的神经网络的训练，可以隐式地学习似然函数。使用Platt缩放可以提高神经似然面的准确性。 |
| [^4] | [CF-VAE: Causal Disentangled Representation Learning with VAE and Causal Flows.](http://arxiv.org/abs/2304.09010) | 本文提出了一种新的因果流以进行因果分离表示学习，设计了一个新模型CF-VAE，利用因果流增强了VAE编码器的分离能力，并展示了在合成和真实数据集上实现因果分离并进行干预实验的结果。 |
| [^5] | [Online Joint Assortment-Inventory Optimization under MNL Choices.](http://arxiv.org/abs/2304.02022) | 本文提出了一个算法解决在线联合组合库存优化问题，能够在平衡探索与开发的措施下实现最大化预期总利润，并为该算法建立了遗憾上界。 |
| [^6] | [Clustered Covariate Regression.](http://arxiv.org/abs/2302.09255) | 本文提出了一种聚类协变量回归方法，该方法通过使用聚类和紧凑参数支持的自然限制来解决高维度协变量问题。与竞争估计器相比，该方法在偏差减小和尺寸控制方面表现出色，并在估计汽油需求的价格和收入弹性方面具有实用性。 |

# 详细

[^1]: 集群抽样下的非参数回归

    Nonparametric Regression under Cluster Sampling

    [https://arxiv.org/abs/2403.04766](https://arxiv.org/abs/2403.04766)

    本文在簇相关性存在的情况下为非参数核回归模型发展了一般渐近理论，并提出了有效的带宽选择和推断方法，引入了渐近方差的估计量，并验证了集群稳健带宽选择的有效性。

    

    本文在簇相关性存在的情况下为非参数核回归模型发展了一般渐近理论。我们研究了非参数密度估计、Nadaraya-Watson核回归和局部线性估计。我们的理论考虑了增长和异质的簇大小。我们推导了渐近条件偏差和方差，确立了一致收敛性，并证明了渐近正态性。我们的发现表明，在异质的簇大小下，渐近方差包括一个反映簇内相关性的新项，当假定簇大小有界时被忽略。我们提出了有效的带宽选择和推断方法，引入了渐近方差的估计量，并证明了它们的一致性。在模拟中，我们验证了集群稳健带宽选择的有效性，并展示了推导的集群稳健置信区间提高了覆盖率。

    arXiv:2403.04766v1 Announce Type: new  Abstract: This paper develops a general asymptotic theory for nonparametric kernel regression in the presence of cluster dependence. We examine nonparametric density estimation, Nadaraya-Watson kernel regression, and local linear estimation. Our theory accommodates growing and heterogeneous cluster sizes. We derive asymptotic conditional bias and variance, establish uniform consistency, and prove asymptotic normality. Our findings reveal that under heterogeneous cluster sizes, the asymptotic variance includes a new term reflecting within-cluster dependence, which is overlooked when cluster sizes are presumed to be bounded. We propose valid approaches for bandwidth selection and inference, introduce estimators of the asymptotic variance, and demonstrate their consistency. In simulations, we verify the effectiveness of the cluster-robust bandwidth selection and show that the derived cluster-robust confidence interval improves the coverage ratio. We 
    
[^2]: 稀疏性的脆弱性

    The Fragility of Sparsity. (arXiv:2311.02299v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2311.02299](http://arxiv.org/abs/2311.02299)

    稀疏性的线性回归估计在选择回归矩阵和假设检验上存在脆弱性，OLS能够提供更健壮的结果而效率损失较小。

    

    我们使用三个实证应用展示了线性回归估计在依赖稀疏性假设时存在两种脆弱性。首先，我们证明在不影响普通最小二乘(OLS)估计的情况下，如基线类别的选择与分类控制相关，可能会使稀疏性估计值移动超过两个标准误。其次，我们开发了两个基于将稀疏性估计与OLS估计进行比较的稀疏性假设检验。在所有三个应用中，这些检验倾向于拒绝稀疏性假设。除非自变量的数量与样本量相当或超过样本量，否则OLS能够以较小的效率损失产生更健壮的结果。

    We show, using three empirical applications, that linear regression estimates which rely on the assumption of sparsity are fragile in two ways. First, we document that different choices of the regressor matrix that do not impact ordinary least squares (OLS) estimates, such as the choice of baseline category with categorical controls, can move sparsity-based estimates two standard errors or more. Second, we develop two tests of the sparsity assumption based on comparing sparsity-based estimators with OLS. The tests tend to reject the sparsity assumption in all three applications. Unless the number of regressors is comparable to or exceeds the sample size, OLS yields more robust results at little efficiency cost.
    
[^3]: 空间过程的神经似然面

    Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods. (arXiv:2305.04634v1 [stat.ME])

    [http://arxiv.org/abs/2305.04634](http://arxiv.org/abs/2305.04634)

    研究提出了使用CNN学习空间过程的似然函数。即使在没有确切似然函数的情况下，通过分类任务进行的神经网络的训练，可以隐式地学习似然函数。使用Platt缩放可以提高神经似然面的准确性。

    

    在空间统计中，当拟合空间过程到真实世界的数据时，快速准确的参数估计和可靠的不确定性量化手段可能是一项具有挑战性的任务，因为似然函数可能评估缓慢或难以处理。 在本研究中，我们提出使用卷积神经网络（CNN）学习空间过程的似然函数。通过特定设计的分类任务，我们的神经网络隐式地学习似然函数，即使在没有显式可用的确切似然函数的情况下也可以实现。一旦在分类任务上进行了训练，我们的神经网络使用Platt缩放进行校准，从而提高了神经似然面的准确性。为了展示我们的方法，我们比较了来自神经似然面的最大似然估计和近似置信区间与两个不同空间过程（高斯过程和对数高斯Cox过程）的相应精确或近似的似然函数构成的等效物。

    In spatial statistics, fast and accurate parameter estimation coupled with a reliable means of uncertainty quantification can be a challenging task when fitting a spatial process to real-world data because the likelihood function might be slow to evaluate or intractable. In this work, we propose using convolutional neural networks (CNNs) to learn the likelihood function of a spatial process. Through a specifically designed classification task, our neural network implicitly learns the likelihood function, even in situations where the exact likelihood is not explicitly available. Once trained on the classification task, our neural network is calibrated using Platt scaling which improves the accuracy of the neural likelihood surfaces. To demonstrate our approach, we compare maximum likelihood estimates and approximate confidence regions constructed from the neural likelihood surface with the equivalent for exact or approximate likelihood for two different spatial processes: a Gaussian Pro
    
[^4]: CF-VAE：基于VAE和因果流的因果分离表示学习

    CF-VAE: Causal Disentangled Representation Learning with VAE and Causal Flows. (arXiv:2304.09010v1 [cs.LG])

    [http://arxiv.org/abs/2304.09010](http://arxiv.org/abs/2304.09010)

    本文提出了一种新的因果流以进行因果分离表示学习，设计了一个新模型CF-VAE，利用因果流增强了VAE编码器的分离能力，并展示了在合成和真实数据集上实现因果分离并进行干预实验的结果。

    

    学习分离表示在表示学习中至关重要，旨在学习数据的低维表示，其中每个维度对应一个潜在的生成因素。由于生成因素之间可能存在因果关系，因果分离表示学习已经受到广泛关注。本文首先提出了一种新的可以将因果结构信息引入模型中的流，称为因果流。基于广泛用于分离表示学习的变分自编码器（VAE），我们设计了一个新模型CF-VAE，利用因果流增强了VAE编码器的分离能力。通过进一步引入基准因素的监督，我们展示了我们模型的分离可识别性。在合成和真实数据集上的实验结果表明，CF-VAE可以实现因果分离并进行干预实验。

    Learning disentangled representations is important in representation learning, aiming to learn a low dimensional representation of data where each dimension corresponds to one underlying generative factor. Due to the possibility of causal relationships between generative factors, causal disentangled representation learning has received widespread attention. In this paper, we first propose a new flows that can incorporate causal structure information into the model, called causal flows. Based on the variational autoencoders(VAE) commonly used in disentangled representation learning, we design a new model, CF-VAE, which enhances the disentanglement ability of the VAE encoder by utilizing the causal flows. By further introducing the supervision of ground-truth factors, we demonstrate the disentanglement identifiability of our model. Experimental results on both synthetic and real datasets show that CF-VAE can achieve causal disentanglement and perform intervention experiments. Moreover, C
    
[^5]: 基于MNL选择模型的在线联合组合库存优化问题研究

    Online Joint Assortment-Inventory Optimization under MNL Choices. (arXiv:2304.02022v1 [cs.LG])

    [http://arxiv.org/abs/2304.02022](http://arxiv.org/abs/2304.02022)

    本文提出了一个算法解决在线联合组合库存优化问题，能够在平衡探索与开发的措施下实现最大化预期总利润，并为该算法建立了遗憾上界。

    

    本文研究了一种在线联合组合库存优化问题，在该问题中，我们假设每个顾客的选择行为都遵循Multinomial Logit（MNL）选择模型，吸引力参数是先验未知的。零售商进行周期性组合和库存决策，以动态地从实现的需求中学习吸引力参数，同时在时间上最大化预期的总利润。本文提出了一种新算法，可以有效地平衡组合和库存在线决策中的探索和开发。我们的算法建立在一个新的MNL吸引力参数估计器，一种通过自适应调整某些已知和未知参数来激励探索的新方法，以及一个用于静态单周期组合库存规划问题的优化oracle基础之上。我们为我们的算法建立了遗憾上界，以及关于在线联合组合库存优化问题的下界。

    We study an online joint assortment-inventory optimization problem, in which we assume that the choice behavior of each customer follows the Multinomial Logit (MNL) choice model, and the attraction parameters are unknown a priori. The retailer makes periodic assortment and inventory decisions to dynamically learn from the realized demands about the attraction parameters while maximizing the expected total profit over time. In this paper, we propose a novel algorithm that can effectively balance the exploration and exploitation in the online decision-making of assortment and inventory. Our algorithm builds on a new estimator for the MNL attraction parameters, a novel approach to incentivize exploration by adaptively tuning certain known and unknown parameters, and an optimization oracle to static single-cycle assortment-inventory planning problems with given parameters. We establish a regret upper bound for our algorithm and a lower bound for the online joint assortment-inventory optimi
    
[^6]: 聚类协变量回归

    Clustered Covariate Regression. (arXiv:2302.09255v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.09255](http://arxiv.org/abs/2302.09255)

    本文提出了一种聚类协变量回归方法，该方法通过使用聚类和紧凑参数支持的自然限制来解决高维度协变量问题。与竞争估计器相比，该方法在偏差减小和尺寸控制方面表现出色，并在估计汽油需求的价格和收入弹性方面具有实用性。

    

    模型估计中协变量维度的高度增加，解决这个问题的现有技术通常需要无序性或不可观测参数向量的离散异质性。然而，在某些经验背景下，经济理论可能不支持任何限制，这可能导致严重的偏差和误导性推断。本文介绍的基于聚类的分组参数估计器（GPE）放弃这两个限制，而选择参数支持是紧凑的自然限制。在标准条件下，GPE具有稳健的大样本性质，并适应了支持可以远离零点的稀疏和非稀疏参数。广泛的蒙特卡洛模拟证明了与竞争估计器相比，GPE在偏差减小和尺寸控制方面的出色性能。对于估计汽油需求的价格和收入弹性的实证应用突显了GPE的实用性。

    High covariate dimensionality is increasingly occurrent in model estimation, and existing techniques to address this issue typically require sparsity or discrete heterogeneity of the unobservable parameter vector. However, neither restriction may be supported by economic theory in some empirical contexts, leading to severe bias and misleading inference. The clustering-based grouped parameter estimator (GPE) introduced in this paper drops both restrictions in favour of the natural one that the parameter support be compact. GPE exhibits robust large sample properties under standard conditions and accommodates both sparse and non-sparse parameters whose support can be bounded away from zero. Extensive Monte Carlo simulations demonstrate the excellent performance of GPE in terms of bias reduction and size control compared to competing estimators. An empirical application of GPE to estimating price and income elasticities of demand for gasoline highlights its practical utility.
    

