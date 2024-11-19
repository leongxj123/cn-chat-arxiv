# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^2] | [AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression](https://arxiv.org/abs/2403.13565) | 提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。 |
| [^3] | [Interpretable Machine Learning for Survival Analysis](https://arxiv.org/abs/2403.10250) | 可解释的机器学习在生存分析中的应用促进了透明度和公平性，揭示了模型的潜在偏见和限制，并提供了更符合数学原理的特征影响和风险因素预测方法。 |
| [^4] | [Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective.](http://arxiv.org/abs/2311.02043) | 本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。 |
| [^5] | [Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects.](http://arxiv.org/abs/2310.08115) | 提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。 |

# 详细

[^1]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^2]: AdaTrans：针对高维回归的特征自适应与样本自适应迁移学习

    AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression

    [https://arxiv.org/abs/2403.13565](https://arxiv.org/abs/2403.13565)

    提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。

    

    我们考虑高维背景下的迁移学习问题，在该问题中，特征维度大于样本大小。为了学习可迁移的信息，该信息可能在特征或源样本之间变化，我们提出一种自适应迁移学习方法，可以检测和聚合特征-wise (F-AdaTrans)或样本-wise (S-AdaTrans)可迁移结构。我们通过采用一种新颖的融合惩罚方法，结合权重，可以根据可迁移结构进行调整。为了选择权重，我们提出了一个在理论上建立，数据驱动的过程，使得 F-AdaTrans 能够选择性地将可迁移的信号与目标融合在一起，同时滤除非可迁移的信号，S-AdaTrans则可以获得每个源样本传递的信息的最佳组合。我们建立了非渐近速率，可以在特殊情况下恢复现有的近最小似乎最优速率。效果证明...

    arXiv:2403.13565v1 Announce Type: cross  Abstract: We consider the transfer learning problem in the high dimensional setting, where the feature dimension is larger than the sample size. To learn transferable information, which may vary across features or the source samples, we propose an adaptive transfer learning method that can detect and aggregate the feature-wise (F-AdaTrans) or sample-wise (S-AdaTrans) transferable structures. We achieve this by employing a novel fused-penalty, coupled with weights that can adapt according to the transferable structure. To choose the weight, we propose a theoretically informed, data-driven procedure, enabling F-AdaTrans to selectively fuse the transferable signals with the target while filtering out non-transferable signals, and S-AdaTrans to obtain the optimal combination of information transferred from each source sample. The non-asymptotic rates are established, which recover existing near-minimax optimal rates in special cases. The effectivene
    
[^3]: 可解释的机器学习用于生存分析

    Interpretable Machine Learning for Survival Analysis

    [https://arxiv.org/abs/2403.10250](https://arxiv.org/abs/2403.10250)

    可解释的机器学习在生存分析中的应用促进了透明度和公平性，揭示了模型的潜在偏见和限制，并提供了更符合数学原理的特征影响和风险因素预测方法。

    

    随着黑盒机器学习模型的传播和快速进步，可解释的机器学习（IML）领域或可解释的人工智能（XAI）在过去十年中变得越来越重要。 这在生存分析领域尤为重要，其中采用IML技术促进了透明度、问责制和公平性，特别是在临床决策过程、有针对性疗法的开发、干预或其他医学或与医疗保健相关的环境中。 具体来说，可解释性可以揭示生存模型的潜在偏见和局限性，并提供更符合数学原理的方法来理解哪些特征对预测有影响或构成风险因素。 然而，缺乏即时可用的IML方法可能已经阻碍了医学从业者和公共卫生政策制定者充分利用机器学习的潜力。

    arXiv:2403.10250v1 Announce Type: cross  Abstract: With the spread and rapid advancement of black box machine learning models, the field of interpretable machine learning (IML) or explainable artificial intelligence (XAI) has become increasingly important over the last decade. This is particularly relevant for survival analysis, where the adoption of IML techniques promotes transparency, accountability and fairness in sensitive areas, such as clinical decision making processes, the development of targeted therapies, interventions or in other medical or healthcare related contexts. More specifically, explainability can uncover a survival model's potential biases and limitations and provide more mathematically sound ways to understand how and which features are influential for prediction or constitute risk factors. However, the lack of readily available IML methods may have deterred medical practitioners and policy makers in public health from leveraging the full potential of machine lea
    
[^4]: 基于子集选择的贝叶斯分位回归：后验总结视角

    Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective. (arXiv:2311.02043v1 [stat.ME])

    [http://arxiv.org/abs/2311.02043](http://arxiv.org/abs/2311.02043)

    本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。

    

    分位回归是一种强大的工具，用于推断协变量如何影响响应分布的特定分位数。现有方法要么分别估计每个感兴趣分位数的条件分位数，要么使用半参数或非参数模型估计整个条件分布。前者经常产生不适合实际数据的模型，并且不在分位数之间共享信息，而后者则以复杂且受限制的模型为特点，难以解释和计算效率低下。此外，这两种方法都不适合于特定分位数的子集选择。相反，我们从贝叶斯决策分析的角度出发，提出了线性分位估计、不确定性量化和子集选择的基本问题。对于任何贝叶斯回归模型，我们为每个基于模型的条件分位数推导出最佳和可解释的线性估计值和不确定性量化。我们的方法引入了一种分位数聚焦的方法。

    Quantile regression is a powerful tool for inferring how covariates affect specific percentiles of the response distribution. Existing methods either estimate conditional quantiles separately for each quantile of interest or estimate the entire conditional distribution using semi- or non-parametric models. The former often produce inadequate models for real data and do not share information across quantiles, while the latter are characterized by complex and constrained models that can be difficult to interpret and computationally inefficient. Further, neither approach is well-suited for quantile-specific subset selection. Instead, we pose the fundamental problems of linear quantile estimation, uncertainty quantification, and subset selection from a Bayesian decision analysis perspective. For any Bayesian regression model, we derive optimal and interpretable linear estimates and uncertainty quantification for each model-based conditional quantile. Our approach introduces a quantile-focu
    
[^5]: 模型不可知的辅助推断方法在部分可辨识因果效应上的应用

    Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects. (arXiv:2310.08115v1 [econ.EM])

    [http://arxiv.org/abs/2310.08115](http://arxiv.org/abs/2310.08115)

    提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。

    

    很多因果估计是部分可辨识的，因为它们依赖于潜在结果之间的不可观察联合分布。基于前处理协变量的分层可以获得更明确的部分可辨识性范围；然而，除非协变量为离散且支撑度相对较小，否则这种方法通常需要对给定协变量的潜在结果的条件分布进行一致估计。因此，现有的方法在模型错误或一致性假设被违反时可能失败。在本研究中，我们提出了一种基于最优输运问题的对偶理论的统一且模型不可知的推断方法，适用于广泛类别的部分可辨识估计。在随机实验中，我们的方法可以结合任何对条件分布的估计，并提供统一有效的推断，即使初始估计是任意不准确的。此外，我们的方法在观测研究中也是双重鲁棒的。

    Many causal estimands are only partially identifiable since they depend on the unobservable joint distribution between potential outcomes. Stratification on pretreatment covariates can yield sharper partial identification bounds; however, unless the covariates are discrete with relatively small support, this approach typically requires consistent estimation of the conditional distributions of the potential outcomes given the covariates. Thus, existing approaches may fail under model misspecification or if consistency assumptions are violated. In this study, we propose a unified and model-agnostic inferential approach for a wide class of partially identified estimands, based on duality theory for optimal transport problems. In randomized experiments, our approach can wrap around any estimates of the conditional distributions and provide uniformly valid inference, even if the initial estimates are arbitrarily inaccurate. Also, our approach is doubly robust in observational studies. Notab
    

