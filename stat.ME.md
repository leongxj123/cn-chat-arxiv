# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Closer Look at AUROC and AUPRC under Class Imbalance.](http://arxiv.org/abs/2401.06091) | 通过数学分析，研究发现AUROC和AUPRC在类别不平衡情况下可以以概率术语简洁地相关联。相比于人们普遍认为的AUPRC优越性，结果表明AUPRC并不如人们预期的有优势，并且可能是一种有害的指标。研究还通过分析大量文献验证了这一结论。 |
| [^2] | [Simple Estimation of Semiparametric Models with Measurement Errors.](http://arxiv.org/abs/2306.14311) | 本文提出了一种解决广义矩量方法（GMM）框架下变量误差（EIV）问题的方法，对于任何初始矩条件，该方法提供了纠正后对EIV具有鲁棒性的矩条件集，这使得GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论，对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。 |

# 详细

[^1]: AUROC和AUPRC在类不平衡下的深入研究

    A Closer Look at AUROC and AUPRC under Class Imbalance. (arXiv:2401.06091v1 [cs.LG])

    [http://arxiv.org/abs/2401.06091](http://arxiv.org/abs/2401.06091)

    通过数学分析，研究发现AUROC和AUPRC在类别不平衡情况下可以以概率术语简洁地相关联。相比于人们普遍认为的AUPRC优越性，结果表明AUPRC并不如人们预期的有优势，并且可能是一种有害的指标。研究还通过分析大量文献验证了这一结论。

    

    在机器学习中，一个广泛的观点是，在二分类任务中，面积受限制的准确率曲线（AUPRC）比受试者工作特征曲线下的面积（AUROC）更好地用于模型比较，尤其是在存在类别不平衡的情况下。本文通过新颖的数学分析挑战了这一观点，并说明了AUROC和AUPRC可以以概率术语简洁地相关联。我们证明了AUPRC并不如人们普遍认为的在类别不平衡的情况下更优，甚至可能是一种有害的指标，因为它倾向于过分偏向于在正样本较为频繁的子群中改善模型。这种偏差可能会无意中增加算法的差异。在这些洞见的推动下，我们对现有的机器学习文献进行了彻底的回顾，并利用大型语言模型对arXiv上的150多万篇论文进行了分析。我们的调查重点是验证和证明声称的AUPRC优越性的普遍性。

    In machine learning (ML), a widespread adage is that the area under the precision-recall curve (AUPRC) is a superior metric for model comparison to the area under the receiver operating characteristic (AUROC) for binary classification tasks with class imbalance. This paper challenges this notion through novel mathematical analysis, illustrating that AUROC and AUPRC can be concisely related in probabilistic terms. We demonstrate that AUPRC, contrary to popular belief, is not superior in cases of class imbalance and might even be a harmful metric, given its inclination to unduly favor model improvements in subpopulations with more frequent positive labels. This bias can inadvertently heighten algorithmic disparities. Prompted by these insights, a thorough review of existing ML literature was conducted, utilizing large language models to analyze over 1.5 million papers from arXiv. Our investigation focused on the prevalence and substantiation of the purported AUPRC superiority. The result
    
[^2]: 测量误差中半参数模型的简单估计

    Simple Estimation of Semiparametric Models with Measurement Errors. (arXiv:2306.14311v1 [econ.EM])

    [http://arxiv.org/abs/2306.14311](http://arxiv.org/abs/2306.14311)

    本文提出了一种解决广义矩量方法（GMM）框架下变量误差（EIV）问题的方法，对于任何初始矩条件，该方法提供了纠正后对EIV具有鲁棒性的矩条件集，这使得GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论，对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。

    

    我们在广义矩量方法（GMM）框架下开发了一种解决变量误差（EIV）问题的实用方法。我们关注的是EIV的可变性是测量误差变量的一小部分的情况，这在实证应用中很常见。对于任何初始矩条件，我们的方法提供了纠正后对EIV具有鲁棒性的矩条件集。我们表明，基于这些矩的GMM估计量是根号下n一致的，标准检验和置信区间提供有效的推论。即使EIV很大，朴素估计量（忽略EIV问题）可能严重偏误并且置信区间的覆盖率为0％，我们的方法也能处理。我们的方法不涉及非参数估计，这对于具有多个协变量和多元的，序贯相关或非经典EIV的应用程序特别重要。

    We develop a practical way of addressing the Errors-In-Variables (EIV) problem in the Generalized Method of Moments (GMM) framework. We focus on the settings in which the variability of the EIV is a fraction of that of the mismeasured variables, which is typical for empirical applications. For any initial set of moment conditions our approach provides a corrected set of moment conditions that are robust to the EIV. We show that the GMM estimator based on these moments is root-n-consistent, with the standard tests and confidence intervals providing valid inference. This is true even when the EIV are so large that naive estimators (that ignore the EIV problem) may be heavily biased with the confidence intervals having 0% coverage. Our approach involves no nonparametric estimation, which is particularly important for applications with multiple covariates, and settings with multivariate, serially correlated, or non-classical EIV.
    

