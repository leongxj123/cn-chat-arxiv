# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation for sigmoidal classification models](https://arxiv.org/abs/2402.08151) | 本研究引入了渐变流自适应重要性抽样的方法，用于稳定贝叶斯分类模型的留一交叉验证预测的蒙特卡罗近似，以评估模型的普适性。 |
| [^2] | [A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation.](http://arxiv.org/abs/2306.16297) | 这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。 |
| [^3] | [Inference on Extreme Quantiles of Unobserved Individual Heterogeneity.](http://arxiv.org/abs/2210.08524) | 本文提出了一种可计算个体异质性极值分位数置信区间的方法，该方法通过构造极值定理和中间阶定理，以及适当的速率和动量条件来解决只有带有噪声的估计可用的情况下，推断个体异质性的极值分位数的问题。 |

# 详细

[^1]: 渐变流自适应重要性抽样用于sigmoid分类模型的贝叶斯留一交叉验证

    Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation for sigmoidal classification models

    [https://arxiv.org/abs/2402.08151](https://arxiv.org/abs/2402.08151)

    本研究引入了渐变流自适应重要性抽样的方法，用于稳定贝叶斯分类模型的留一交叉验证预测的蒙特卡罗近似，以评估模型的普适性。

    

    我们引入了一组梯度流引导的自适应重要性抽样（IS）变换，用于稳定贝叶斯分类模型的点级留一交叉验证（LOO）预测的蒙特卡罗近似。可以利用这种方法来评估模型的普适性，例如计算与AIC类似的LOO或计算LOO ROC / PRC曲线以及派生的度量指标，如AUROC和AUPRC。通过变分法和梯度流，我们推导出两个简单的非线性单步变换，利用梯度信息将模型的预训练完整数据后验靠近目标LOO后验预测分布。这样，变换稳定了重要性权重。因为变换涉及到似然函数的梯度，所以结果的蒙特卡罗积分依赖于模型Hessian的Jacobian行列式。我们推导出了这些Jacobian行列式的闭合精确公式。

    We introduce a set of gradient-flow-guided adaptive importance sampling (IS) transformations to stabilize Monte-Carlo approximations of point-wise leave one out cross-validated (LOO) predictions for Bayesian classification models. One can leverage this methodology for assessing model generalizability by for instance computing a LOO analogue to the AIC or computing LOO ROC/PRC curves and derived metrics like the AUROC and AUPRC. By the calculus of variations and gradient flow, we derive two simple nonlinear single-step transformations that utilize gradient information to shift a model's pre-trained full-data posterior closer to the target LOO posterior predictive distributions. In doing so, the transformations stabilize importance weights. Because the transformations involve the gradient of the likelihood function, the resulting Monte Carlo integral depends on Jacobian determinants with respect to the model Hessian. We derive closed-form exact formulae for these Jacobian determinants in
    
[^2]: 一种用于评估时变调节因素的因果偏离效应估计的元学习方法

    A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation. (arXiv:2306.16297v1 [stat.ME])

    [http://arxiv.org/abs/2306.16297](http://arxiv.org/abs/2306.16297)

    这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。

    

    可穿戴技术和智能手机提供的数字化健康干预的双重革命显著增加了移动健康（mHealth）干预在各个健康科学领域的可及性和采纳率。顺序随机实验称为微随机试验（MRTs）已经越来越受欢迎，用于实证评估这些mHealth干预组成部分的有效性。MRTs产生了一类新的因果估计量，称为“因果偏离效应”，使健康科学家能够评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。然而，目前用于估计因果偏离效应的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型。虽然机器学习算法在自动特征构建方面具有优势，但其朴素应用导致了问题。

    Twin revolutions in wearable technologies and smartphone-delivered digital health interventions have significantly expanded the accessibility and uptake of mobile health (mHealth) interventions across various health science domains. Sequentially randomized experiments called micro-randomized trials (MRTs) have grown in popularity to empirically evaluate the effectiveness of these mHealth intervention components. MRTs have given rise to a new class of causal estimands known as "causal excursion effects", which enable health scientists to assess how intervention effectiveness changes over time or is moderated by individual characteristics, context, or responses in the past. However, current data analysis methods for estimating causal excursion effects require pre-specified features of the observed high-dimensional history to construct a working model of an important nuisance parameter. While machine learning algorithms are ideal for automatic feature construction, their naive application
    
[^3]: 论个体异质性极值分位数推断

    Inference on Extreme Quantiles of Unobserved Individual Heterogeneity. (arXiv:2210.08524v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.08524](http://arxiv.org/abs/2210.08524)

    本文提出了一种可计算个体异质性极值分位数置信区间的方法，该方法通过构造极值定理和中间阶定理，以及适当的速率和动量条件来解决只有带有噪声的估计可用的情况下，推断个体异质性的极值分位数的问题。

    

    在面板数据或元分析设置中，我们提出了一种方法，用于推断个体异质性（异质性系数、异质性处理效应等）的极值分位数。在这样的设置中，推断是具有挑战性的：只有关于未观测异质性的带有噪声的估计可用，而基于中心极限定理的逼近在极值分位数上效果不佳。针对这种情况，我们在弱假设下推导出噪声估计的极值定理和中间阶定理，以及适当的速率和动量条件。然后，我们使用这两个定理构造极值分位数的置信区间。这些区间易于构造，无需进行优化。基于中间阶定理的推断涉及一种新颖的自标准化中间阶定理。在模拟中，我们的极值置信区间在尾部具有良好的覆盖性质。我们的方法是通过使用来自1979年国家纵向青年调查的数据，对教育回报率分布的估计进行演示。

    We develop a methodology for conducting inference on extreme quantiles of unobserved individual heterogeneity (heterogeneous coefficients, heterogeneous treatment effects, etc.) in a panel data or meta-analysis setting. Inference in such settings is challenging: only noisy estimates of unobserved heterogeneity are available, and approximations based on the central limit theorem work poorly for extreme quantiles. For this situation, under weak assumptions we derive an extreme value theorem and an intermediate order theorem for noisy estimates and appropriate rate and moment conditions. Both theorems are then used to construct confidence intervals for extremal quantiles. The intervals are simple to construct and require no optimization. Inference based on the intermediate order theorem involves a novel self-normalized intermediate order theorem. In simulations, our extremal confidence intervals have favorable coverage properties in the tail. Our methodology is illustrated with an applica
    

