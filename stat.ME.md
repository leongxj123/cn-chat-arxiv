# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distributionally Robust Machine Learning with Multi-source Data.](http://arxiv.org/abs/2309.02211) | 本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。 |
| [^2] | [Non-parametric Hypothesis Tests for Distributional Group Symmetry.](http://arxiv.org/abs/2307.15834) | 该论文提出了用于分布对称性的非参数假设检验方法，适用于具有对称性的数据集。具体而言，该方法在紧致群作用下测试边际或联合分布的不变性，并提出了一种易于实施的条件蒙特卡罗检验。 |
| [^3] | [Variational Sequential Optimal Experimental Design using Reinforcement Learning.](http://arxiv.org/abs/2306.10430) | 该研究提出了一种基于贝叶斯框架和信息增益效用的变分序列最优实验设计方法，通过强化学习求解最优设计策略，适用于多种OED问题，结果具有更高的样本效率和更少的前向模型模拟次数。 |
| [^4] | [Conditionally Calibrated Predictive Distributions by Probability-Probability Map: Application to Galaxy Redshift Estimation and Probabilistic Forecasting.](http://arxiv.org/abs/2205.14568) | 本研究提出了一种名为Cal-PIT的方法，通过学习一个概率-概率映射，解决了预测分布的诊断和校准问题，来实现有条件校准。 |

# 详细

[^1]: 基于多源数据的分布鲁棒机器学习

    Distributionally Robust Machine Learning with Multi-source Data. (arXiv:2309.02211v1 [stat.ML])

    [http://arxiv.org/abs/2309.02211](http://arxiv.org/abs/2309.02211)

    本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。

    

    当目标分布与源数据集不同时，传统的机器学习方法可能导致较差的预测性能。本文利用多个数据源，并引入了一种基于组分布鲁棒预测模型来优化关于目标分布类的可解释方差的对抗性奖励。与传统的经验风险最小化相比，所提出的鲁棒预测模型改善了具有分布偏移的目标人群的预测准确性。我们证明了组分布鲁棒预测模型是源数据集条件结果模型的加权平均。我们利用这一关键鉴别结果来提高任意机器学习算法的鲁棒性，包括随机森林和神经网络等。我们设计了一种新的偏差校正估计器来估计通用机器学习算法的最优聚合权重，并展示了其在c方面的改进。

    Classical machine learning methods may lead to poor prediction performance when the target distribution differs from the source populations. This paper utilizes data from multiple sources and introduces a group distributionally robust prediction model defined to optimize an adversarial reward about explained variance with respect to a class of target distributions. Compared to classical empirical risk minimization, the proposed robust prediction model improves the prediction accuracy for target populations with distribution shifts. We show that our group distributionally robust prediction model is a weighted average of the source populations' conditional outcome models. We leverage this key identification result to robustify arbitrary machine learning algorithms, including, for example, random forests and neural networks. We devise a novel bias-corrected estimator to estimate the optimal aggregation weight for general machine-learning algorithms and demonstrate its improvement in the c
    
[^2]: 非参数假设检验对分配群对称性的研究

    Non-parametric Hypothesis Tests for Distributional Group Symmetry. (arXiv:2307.15834v1 [stat.ME])

    [http://arxiv.org/abs/2307.15834](http://arxiv.org/abs/2307.15834)

    该论文提出了用于分布对称性的非参数假设检验方法，适用于具有对称性的数据集。具体而言，该方法在紧致群作用下测试边际或联合分布的不变性，并提出了一种易于实施的条件蒙特卡罗检验。

    

    对称性在科学、机器学习和统计学中起着重要的作用。对于已知遵循对称性的数据，已经开发出了许多利用对称性的方法。然而，对于普遍群对称性的存在或不存在的统计检验几乎不存在。本研究提出了一种非参数假设检验方法，基于单个独立同分布样本，用于针对特定群的分布对称性。我们提供了适用于两种广泛情况的对称性检验的一般公式。第一种情况是测试在紧致群作用下的边际或联合分布的不变性。在这里，一个渐近无偏的检验只需要一个可计算的概率分布空间上的度量和能够均匀随机采样群元素的能力。在此基础上，我们提出了一种易于实施的条件蒙特卡罗检验，并证明它可以实现精确的p值。

    Symmetry plays a central role in the sciences, machine learning, and statistics. For situations in which data are known to obey a symmetry, a multitude of methods that exploit symmetry have been developed. Statistical tests for the presence or absence of general group symmetry, however, are largely non-existent. This work formulates non-parametric hypothesis tests, based on a single independent and identically distributed sample, for distributional symmetry under a specified group. We provide a general formulation of tests for symmetry that apply to two broad settings. The first setting tests for the invariance of a marginal or joint distribution under the action of a compact group. Here, an asymptotically unbiased test only requires a computable metric on the space of probability distributions and the ability to sample uniformly random group elements. Building on this, we propose an easy-to-implement conditional Monte Carlo test and prove that it achieves exact $p$-values with finitel
    
[^3]: 基于强化学习的变分序列最优实验设计方法

    Variational Sequential Optimal Experimental Design using Reinforcement Learning. (arXiv:2306.10430v1 [stat.ML])

    [http://arxiv.org/abs/2306.10430](http://arxiv.org/abs/2306.10430)

    该研究提出了一种基于贝叶斯框架和信息增益效用的变分序列最优实验设计方法，通过强化学习求解最优设计策略，适用于多种OED问题，结果具有更高的样本效率和更少的前向模型模拟次数。

    

    我们引入了变分序列最优实验设计 (vsOED) 的新方法，通过贝叶斯框架和信息增益效用来最优地设计有限序列的实验。具体而言，我们通过变分近似贝叶斯后验的下界估计期望效用。通过同时最大化变分下界和执行策略梯度更新来数值解决最优设计策略。我们将这种方法应用于一系列面向参数推断、模型区分和目标导向预测的OED问题。这些案例涵盖了显式和隐式似然函数、麻烦参数和基于物理的偏微分方程模型。我们的vsOED结果表明，与以前的顺序设计算法相比，样本效率大大提高，所需前向模型模拟次数减少了。

    We introduce variational sequential Optimal Experimental Design (vsOED), a new method for optimally designing a finite sequence of experiments under a Bayesian framework and with information-gain utilities. Specifically, we adopt a lower bound estimator for the expected utility through variational approximation to the Bayesian posteriors. The optimal design policy is solved numerically by simultaneously maximizing the variational lower bound and performing policy gradient updates. We demonstrate this general methodology for a range of OED problems targeting parameter inference, model discrimination, and goal-oriented prediction. These cases encompass explicit and implicit likelihoods, nuisance parameters, and physics-based partial differential equation models. Our vsOED results indicate substantially improved sample efficiency and reduced number of forward model simulations compared to previous sequential design algorithms.
    
[^4]: 通过概率-概率映射实现有条件校准的预测分布：在银河红移估计和概率预测中的应用

    Conditionally Calibrated Predictive Distributions by Probability-Probability Map: Application to Galaxy Redshift Estimation and Probabilistic Forecasting. (arXiv:2205.14568v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.14568](http://arxiv.org/abs/2205.14568)

    本研究提出了一种名为Cal-PIT的方法，通过学习一个概率-概率映射，解决了预测分布的诊断和校准问题，来实现有条件校准。

    

    不确定性量化对于评估AI算法的预测能力至关重要。过去的研究致力于描述目标变量$y \in \mathbb{R}$在给定复杂输入特征$\mathbf{x} \in \mathcal{X}$的条件下的预测分布$F(y|\mathbf{x})$。然而，现有的预测分布（例如，归一化流和贝叶斯神经网络）往往缺乏条件校准，即给定输入$\mathbf{x}$的事件发生的概率与预测概率显著不同。当前的校准方法不能完全评估和实施有条件校准的预测分布。在这里，我们提出了一种名为Cal-PIT的方法，它通过从校准数据中学习一个概率-概率映射来同时解决预测分布的诊断和校准问题。关键思想是对概率积分变换分数进行$\mathbf{x}$的回归。估计的回归提供了对特征空间中条件覆盖的可解释诊断。

    Uncertainty quantification is crucial for assessing the predictive ability of AI algorithms. Much research has been devoted to describing the predictive distribution (PD) $F(y|\mathbf{x})$ of a target variable $y \in \mathbb{R}$ given complex input features $\mathbf{x} \in \mathcal{X}$. However, off-the-shelf PDs (from, e.g., normalizing flows and Bayesian neural networks) often lack conditional calibration with the probability of occurrence of an event given input $\mathbf{x}$ being significantly different from the predicted probability. Current calibration methods do not fully assess and enforce conditionally calibrated PDs. Here we propose \texttt{Cal-PIT}, a method that addresses both PD diagnostics and recalibration by learning a single probability-probability map from calibration data. The key idea is to regress probability integral transform scores against $\mathbf{x}$. The estimated regression provides interpretable diagnostics of conditional coverage across the feature space. 
    

