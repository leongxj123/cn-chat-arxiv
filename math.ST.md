# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift.](http://arxiv.org/abs/2302.10160) | 该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。 |
| [^2] | [Local Polynomial Estimation of Time-Varying Parameters in Nonlinear Models.](http://arxiv.org/abs/1904.05209) | 本论文发展了一种局部多项式估计时间变化参数的新理论，并证明在弱正则条件下，所提出的估计器是一致的，且服从正态分布。此外，该方法对于不同模型具有普适性，并能够在泊松自回归模型中得到应用。 |

# 详细

[^1]: 核岭回归下伪标签的协变量转移策略

    Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift. (arXiv:2302.10160v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.10160](http://arxiv.org/abs/2302.10160)

    该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。

    

    我们提出并分析了一种基于协变量转移的核岭回归方法。我们的目标是在目标分布上学习一个均方误差最小的回归函数，基于从目标分布采样的未标记数据和可能具有不同特征分布的已标记数据。我们将已标记数据分成两个子集，并分别进行核岭回归，以获得候选模型集合和一个填充模型。我们使用后者填充缺失的标签，然后相应地选择最佳的候选模型。我们的非渐近性过量风险界表明，在相当一般的情况下，我们的估计器能够适应目标分布以及协变量转移的结构。它能够实现渐近正态误差率直到对数因子的最小极限优化。在模型选择中使用伪标签不会产生主要负面影响。

    We develop and analyze a principled approach to kernel ridge regression under covariate shift. The goal is to learn a regression function with small mean squared error over a target distribution, based on unlabeled data from there and labeled data that may have a different feature distribution. We propose to split the labeled data into two subsets and conduct kernel ridge regression on them separately to obtain a collection of candidate models and an imputation model. We use the latter to fill the missing labels and then select the best candidate model accordingly. Our non-asymptotic excess risk bounds show that in quite general scenarios, our estimator adapts to the structure of the target distribution as well as the covariate shift. It achieves the minimax optimal error rate up to a logarithmic factor. The use of pseudo-labels in model selection does not have major negative impacts.
    
[^2]: 非线性模型中时间变化参数的局部多项式估计

    Local Polynomial Estimation of Time-Varying Parameters in Nonlinear Models. (arXiv:1904.05209v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/1904.05209](http://arxiv.org/abs/1904.05209)

    本论文发展了一种局部多项式估计时间变化参数的新理论，并证明在弱正则条件下，所提出的估计器是一致的，且服从正态分布。此外，该方法对于不同模型具有普适性，并能够在泊松自回归模型中得到应用。

    

    我们在广泛类别的非线性时间序列模型中发展了一种新的渐近理论来估计时间变化参数的局部多项式（准）最大似然估计器。在弱正则条件下，我们证明所提出的估计器在大样本下是一致的，并且服从正态分布。与现有理论相比，我们的条件对于数据生成过程及其似然函数的光滑性和矩条件要求较低。此外，估计器的偏差项具有更简单的形式。我们通过将理论应用于局部准最大似然估计的时间变化VAR、ARCH和GARCH以及泊松自回归模型，展示了我们普适结果的有益性。对于前三个模型，我们能够大大减弱现有文献中的条件要求。对于泊松自回归模型，现有理论不能应用，而我们的新方法使我们能够分析它。

    We develop a novel asymptotic theory for local polynomial (quasi-) maximum-likelihood estimators of time-varying parameters in a broad class of nonlinear time series models. Under weak regularity conditions, we show the proposed estimators are consistent and follow normal distributions in large samples. Our conditions impose weaker smoothness and moment conditions on the data-generating process and its likelihood compared to existing theories. Furthermore, the bias terms of the estimators take a simpler form. We demonstrate the usefulness of our general results by applying our theory to local (quasi-)maximum-likelihood estimators of a time-varying VAR's, ARCH and GARCH, and Poisson autogressions. For the first three models, we are able to substantially weaken the conditions found in the existing literature. For the Poisson autogression, existing theories cannot be be applied while our novel approach allows us to analyze it.
    

