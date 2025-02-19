# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synthetic Control Methods by Density Matching under Implicit Endogeneitiy.](http://arxiv.org/abs/2307.11127) | 本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。 |
| [^2] | [Adjustment with Many Regressors Under Covariate-Adaptive Randomizations.](http://arxiv.org/abs/2304.08184) | 本文关注协变量自适应随机化中的因果推断，在使用回归调整时需要权衡效率与估计误差成本。作者提供了关于经过回归调整的平均处理效应（ATE）估计器的统一推断理论。 |

# 详细

[^1]: 通过密度匹配实现的合成对照方法下的隐式内生性问题

    Synthetic Control Methods by Density Matching under Implicit Endogeneitiy. (arXiv:2307.11127v1 [econ.EM])

    [http://arxiv.org/abs/2307.11127](http://arxiv.org/abs/2307.11127)

    本文提出了一种新型的合成对照方法，通过密度匹配来解决现有SCMs中的隐式内生性问题。该方法通过将经过处理单元的结果密度与未处理单元的密度进行加权平均来估计SC权重。

    

    合成对照方法（SCMs）已成为比较案例研究中因果推断的重要工具。SCMs的基本思想是通过使用来自未处理单元的观测结果的加权和来估计经过处理单元的反事实结果。合成对照（SC）的准确性对于估计因果效应至关重要，因此，SC权重的估计成为了研究的焦点。在本文中，我们首先指出现有的SCMs存在一个隐式内生性问题，即未处理单元的结果与反事实结果模型中的误差项之间的相关性。我们展示了这个问题会对因果效应估计器产生偏差。然后，我们提出了一种基于密度匹配的新型SCM，假设经过处理单元的结果密度可以用未处理单元的密度的加权平均来近似（即混合模型）。基于这一假设，我们通过匹配来估计SC权重。

    Synthetic control methods (SCMs) have become a crucial tool for causal inference in comparative case studies. The fundamental idea of SCMs is to estimate counterfactual outcomes for a treated unit by using a weighted sum of observed outcomes from untreated units. The accuracy of the synthetic control (SC) is critical for estimating the causal effect, and hence, the estimation of SC weights has been the focus of much research. In this paper, we first point out that existing SCMs suffer from an implicit endogeneity problem, which is the correlation between the outcomes of untreated units and the error term in the model of a counterfactual outcome. We show that this problem yields a bias in the causal effect estimator. We then propose a novel SCM based on density matching, assuming that the density of outcomes of the treated unit can be approximated by a weighted average of the densities of untreated units (i.e., a mixture model). Based on this assumption, we estimate SC weights by matchi
    
[^2]: 协变量自适应随机化下的多个回归器的调整

    Adjustment with Many Regressors Under Covariate-Adaptive Randomizations. (arXiv:2304.08184v1 [econ.EM])

    [http://arxiv.org/abs/2304.08184](http://arxiv.org/abs/2304.08184)

    本文关注协变量自适应随机化中的因果推断，在使用回归调整时需要权衡效率与估计误差成本。作者提供了关于经过回归调整的平均处理效应（ATE）估计器的统一推断理论。

    

    本文针对协变量自适应随机化（CAR）中的因果推断使用回归调整（RA）时存在的权衡进行了研究。RA可以通过整合未用于随机分配的协变量信息来提高因果估计器的效率。但是，当回归器数量与样本量同阶时，RA的估计误差不能渐近忽略，会降低估计效率。没有考虑RA成本的结果可能导致在零假设下过度拒绝因果推断。为了解决这个问题，我们在CAR下为经过回归调整的平均处理效应（ATE）估计器开发了一种统一的推断理论。我们的理论具有两个关键特征：（1）确保在零假设下的精确渐近大小，无论协变量数量是固定还是最多以样本大小的速度发散，（2）确保在协变量维度方面都比未调整的估计器弱效提高.

    Our paper identifies a trade-off when using regression adjustments (RAs) in causal inference under covariate-adaptive randomizations (CARs). On one hand, RAs can improve the efficiency of causal estimators by incorporating information from covariates that are not used in the randomization. On the other hand, RAs can degrade estimation efficiency due to their estimation errors, which are not asymptotically negligible when the number of regressors is of the same order as the sample size. Failure to account for the cost of RAs can result in over-rejection of causal inference under the null hypothesis. To address this issue, we develop a unified inference theory for the regression-adjusted average treatment effect (ATE) estimator under CARs. Our theory has two key features: (1) it ensures the exact asymptotic size under the null hypothesis, regardless of whether the number of covariates is fixed or diverges at most at the rate of the sample size, and (2) it guarantees weak efficiency impro
    

