# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation.](http://arxiv.org/abs/2308.02293) | 通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。 |
| [^2] | [The extended Ville's inequality for nonintegrable nonnegative supermartingales.](http://arxiv.org/abs/2304.01163) | 本文提出了一种新的理论来描述非负超马氏过程，并推导出一个新的极大不等式，适用于非可积情况，并说明了混合方法的扩展以及该理论在顺序统计中的应用。 |
| [^3] | [Double Robust Bayesian Inference on Average Treatment Effects.](http://arxiv.org/abs/2211.16298) | 本文研究了双重鲁棒贝叶斯推断程序，实现了平均处理效应的偏差校正并形成了可信区间。 |

# 详细

[^1]: 用正则化高阶总变差的随机优化方法训练非线性神经网络

    A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation. (arXiv:2308.02293v1 [stat.ME])

    [http://arxiv.org/abs/2308.02293](http://arxiv.org/abs/2308.02293)

    通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。

    

    尽管包括深度神经网络在内的高度表达的参数模型可以更好地建模复杂概念，但训练这种高度非线性模型已知会导致严重的过拟合风险。针对这个问题，本研究考虑了一种k阶总变差（k-TV）正则化，它被定义为要训练的参数模型的k阶导数的平方积分，通过惩罚k-TV来产生一个更平滑的函数，从而避免过拟合。尽管将k-TV项应用于一般的参数模型由于积分而导致计算复杂，本研究提供了一种随机优化算法，可以高效地训练带有k-TV正则化的一般模型，而无需进行显式的数值积分。这种方法可以应用于结构任意的深度神经网络的训练，因为它只需要进行简单的随机梯度优化即可实现。

    While highly expressive parametric models including deep neural networks have an advantage to model complicated concepts, training such highly non-linear models is known to yield a high risk of notorious overfitting. To address this issue, this study considers a $k$th order total variation ($k$-TV) regularization, which is defined as the squared integral of the $k$th order derivative of the parametric models to be trained; penalizing the $k$-TV is expected to yield a smoother function, which is expected to avoid overfitting. While the $k$-TV terms applied to general parametric models are computationally intractable due to the integration, this study provides a stochastic optimization algorithm, that can efficiently train general models with the $k$-TV regularization without conducting explicit numerical integration. The proposed approach can be applied to the training of even deep neural networks whose structure is arbitrary, as it can be implemented by only a simple stochastic gradien
    
[^2]: 非可积非负超马氏过程的扩展维尔不等式

    The extended Ville's inequality for nonintegrable nonnegative supermartingales. (arXiv:2304.01163v1 [math.PR])

    [http://arxiv.org/abs/2304.01163](http://arxiv.org/abs/2304.01163)

    本文提出了一种新的理论来描述非负超马氏过程，并推导出一个新的极大不等式，适用于非可积情况，并说明了混合方法的扩展以及该理论在顺序统计中的应用。

    

    本文在 Robbins 的初始工作基础上，严密地提出了一种非负超马氏过程的扩展理论，不需要可积性或有限性。特别地，我们推导了 Robbins 预示的一个关键极大不等式，称为扩展维尔不等式，它加强了经典的维尔不等式（适用于可积非负超马氏过程），并适用于我们的非可积设置。我们推导了混合方法的扩展，适用于我们扩展的非负超马氏过程的 $\sigma$- 有限混合。我们介绍了我们理论在顺序统计中的一些应用，如在推导非参数置信序列和（扩展）e-过程中使用不适当混合（先验）。

    Following initial work by Robbins, we rigorously present an extended theory of nonnegative supermartingales, requiring neither integrability nor finiteness. In particular, we derive a key maximal inequality foreshadowed by Robbins, which we call the extended Ville's inequality, that strengthens the classical Ville's inequality (for integrable nonnegative supermartingales), and also applies to our nonintegrable setting. We derive an extension of the method of mixtures, which applies to $\sigma$-finite mixtures of our extended nonnegative supermartingales. We present some implications of our theory for sequential statistics, such as the use of improper mixtures (priors) in deriving nonparametric confidence sequences and (extended) e-processes.
    
[^3]: 平均处理效应的双重鲁棒贝叶斯推断

    Double Robust Bayesian Inference on Average Treatment Effects. (arXiv:2211.16298v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.16298](http://arxiv.org/abs/2211.16298)

    本文研究了双重鲁棒贝叶斯推断程序，实现了平均处理效应的偏差校正并形成了可信区间。

    

    我们研究了无偏性下的平均处理效应（ATE）的双重鲁棒贝叶斯推断程序。我们的鲁棒贝叶斯方法包括两个调整步骤：首先，我们对条件均值函数的先验分布进行校正；其次，我们在产生的ATE的后验分布上引入一个重新居中术语。我们通过建立双重鲁棒性下的半参数Bernstein-von Mises定理，证明了我们的贝叶斯估计量和双重鲁棒频率估计量的渐近等价性；即，条件均值函数的缺乏平滑性可以通过概率得分的高规则性进行补偿，反之亦然。因此，产生的贝叶斯点估计内在化了频率型双重鲁棒估计量的偏差校正，而贝叶斯可信集形成的置信区间具有渐近精确的覆盖概率。在模拟中，我们发现这种鲁棒的贝叶斯程序导致了显着的...

    We study a double robust Bayesian inference procedure on the average treatment effect (ATE) under unconfoundedness. Our robust Bayesian approach involves two adjustment steps: first, we make a correction for prior distributions of the conditional mean function; second, we introduce a recentering term on the posterior distribution of the resulting ATE. We prove asymptotic equivalence of our Bayesian estimator and double robust frequentist estimators by establishing a new semiparametric Bernstein-von Mises theorem under double robustness; i.e., the lack of smoothness of conditional mean functions can be compensated by high regularity of the propensity score and vice versa. Consequently, the resulting Bayesian point estimator internalizes the bias correction as the frequentist-type doubly robust estimator, and the Bayesian credible sets form confidence intervals with asymptotically exact coverage probability. In simulations, we find that this robust Bayesian procedure leads to significant
    

