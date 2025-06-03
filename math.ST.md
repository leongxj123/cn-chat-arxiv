# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation for sigmoidal classification models](https://arxiv.org/abs/2402.08151) | 本研究引入了渐变流自适应重要性抽样的方法，用于稳定贝叶斯分类模型的留一交叉验证预测的蒙特卡罗近似，以评估模型的普适性。 |
| [^2] | [Kernel $\epsilon$-Greedy for Contextual Bandits.](http://arxiv.org/abs/2306.17329) | 本文提出了基于核的$\epsilon$-贪心策略应用于情境脉冲中的方法，通过在线加权核岭回归估计器实现对奖励函数的估计，并证明了其一致性和依赖于RKHS维度的次线性后悔率，在有限维RKHS的边际条件下实现了最优后悔率。 |

# 详细

[^1]: 渐变流自适应重要性抽样用于sigmoid分类模型的贝叶斯留一交叉验证

    Gradient-flow adaptive importance sampling for Bayesian leave one out cross-validation for sigmoidal classification models

    [https://arxiv.org/abs/2402.08151](https://arxiv.org/abs/2402.08151)

    本研究引入了渐变流自适应重要性抽样的方法，用于稳定贝叶斯分类模型的留一交叉验证预测的蒙特卡罗近似，以评估模型的普适性。

    

    我们引入了一组梯度流引导的自适应重要性抽样（IS）变换，用于稳定贝叶斯分类模型的点级留一交叉验证（LOO）预测的蒙特卡罗近似。可以利用这种方法来评估模型的普适性，例如计算与AIC类似的LOO或计算LOO ROC / PRC曲线以及派生的度量指标，如AUROC和AUPRC。通过变分法和梯度流，我们推导出两个简单的非线性单步变换，利用梯度信息将模型的预训练完整数据后验靠近目标LOO后验预测分布。这样，变换稳定了重要性权重。因为变换涉及到似然函数的梯度，所以结果的蒙特卡罗积分依赖于模型Hessian的Jacobian行列式。我们推导出了这些Jacobian行列式的闭合精确公式。

    We introduce a set of gradient-flow-guided adaptive importance sampling (IS) transformations to stabilize Monte-Carlo approximations of point-wise leave one out cross-validated (LOO) predictions for Bayesian classification models. One can leverage this methodology for assessing model generalizability by for instance computing a LOO analogue to the AIC or computing LOO ROC/PRC curves and derived metrics like the AUROC and AUPRC. By the calculus of variations and gradient flow, we derive two simple nonlinear single-step transformations that utilize gradient information to shift a model's pre-trained full-data posterior closer to the target LOO posterior predictive distributions. In doing so, the transformations stabilize importance weights. Because the transformations involve the gradient of the likelihood function, the resulting Monte Carlo integral depends on Jacobian determinants with respect to the model Hessian. We derive closed-form exact formulae for these Jacobian determinants in
    
[^2]: 基于核的$\epsilon$-贪心策略在情境脉冲中的应用

    Kernel $\epsilon$-Greedy for Contextual Bandits. (arXiv:2306.17329v1 [stat.ML])

    [http://arxiv.org/abs/2306.17329](http://arxiv.org/abs/2306.17329)

    本文提出了基于核的$\epsilon$-贪心策略应用于情境脉冲中的方法，通过在线加权核岭回归估计器实现对奖励函数的估计，并证明了其一致性和依赖于RKHS维度的次线性后悔率，在有限维RKHS的边际条件下实现了最优后悔率。

    

    我们考虑了情境脉冲中的基于核的$\epsilon$-贪心策略。更具体地说，在有限数量的臂的情况下，我们认为平均奖励函数位于再生核希尔伯特空间（RKHS）中。我们提出了一种用于奖励函数的在线加权核岭回归估计器。在对探索概率序列$\{\epsilon_t\}_t$和正则化参数$\{\lambda_t\}_t$的一些条件下，我们证明了所提出的估计器的一致性。我们还证明，对于任何核和相应的RKHS的选择，我们可以实现依赖于RKHS内在维度的次线性后悔率。此外，在有限维RKHS的边际条件下，我们实现了$\sqrt{T}$的最优后悔率。

    We consider a kernelized version of the $\epsilon$-greedy strategy for contextual bandits. More precisely, in a setting with finitely many arms, we consider that the mean reward functions lie in a reproducing kernel Hilbert space (RKHS). We propose an online weighted kernel ridge regression estimator for the reward functions. Under some conditions on the exploration probability sequence, $\{\epsilon_t\}_t$, and choice of the regularization parameter, $\{\lambda_t\}_t$, we show that the proposed estimator is consistent. We also show that for any choice of kernel and the corresponding RKHS, we achieve a sub-linear regret rate depending on the intrinsic dimensionality of the RKHS. Furthermore, we achieve the optimal regret rate of $\sqrt{T}$ under a margin condition for finite-dimensional RKHS.
    

