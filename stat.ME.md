# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Does AI help humans make better decisions? A methodological framework for experimental evaluation](https://arxiv.org/abs/2403.12108) | 引入一种新的实验框架用于评估人类是否通过使用AI可以做出更好的决策，在单盲实验设计中比较了三种决策系统的表现 |
| [^2] | [Multivariate Tie-breaker Designs](https://arxiv.org/abs/2202.10030) | 该研究探讨了一种多元回归的打破平局设计，通过优化D-最优准则的治疗概率，实现了资源分配效率和统计效率的权衡。 |
| [^3] | [Simultaneous inference for generalized linear models with unmeasured confounders.](http://arxiv.org/abs/2309.07261) | 本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。 |
| [^4] | [Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA.](http://arxiv.org/abs/2303.06198) | 本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。 |
| [^5] | [Uncertainty Quantification in Synthetic Controls with Staggered Treatment Adoption.](http://arxiv.org/abs/2210.05026) | 该论文提出了一种基于原则的预测区间方法，用于量化合成对照预测或估计在错位处理采用的情况下的不确定性。 |

# 详细

[^1]: AI是否有助于人类做出更好的决策？一种用于实验评估的方法论框架

    Does AI help humans make better decisions? A methodological framework for experimental evaluation

    [https://arxiv.org/abs/2403.12108](https://arxiv.org/abs/2403.12108)

    引入一种新的实验框架用于评估人类是否通过使用AI可以做出更好的决策，在单盲实验设计中比较了三种决策系统的表现

    

    基于数据驱动算法的人工智能（AI）在当今社会变得无处不在。然而，在许多情况下，尤其是当利益高昂时，人类仍然作出最终决策。因此，关键问题是AI是否有助于人类比单独的人类或单独的AI做出更好的决策。我们引入了一种新的方法论框架，用于实验性地回答这个问题，而不需要额外的假设。我们使用基于基准潜在结果的标准分类指标测量决策者做出正确决策的能力。我们考虑了一个单盲实验设计，在这个设计中，提供AI生成的建议在不同案例中被随机分配给最终决策的人类。在这种实验设计下，我们展示了如何比较三种替代决策系统的性能--仅人类、人类与AI、仅AI。

    arXiv:2403.12108v1 Announce Type: new  Abstract: The use of Artificial Intelligence (AI) based on data-driven algorithms has become ubiquitous in today's society. Yet, in many cases and especially when stakes are high, humans still make final decisions. The critical question, therefore, is whether AI helps humans make better decisions as compared to a human alone or AI an alone. We introduce a new methodological framework that can be used to answer experimentally this question with no additional assumptions. We measure a decision maker's ability to make correct decisions using standard classification metrics based on the baseline potential outcome. We consider a single-blinded experimental design, in which the provision of AI-generated recommendations is randomized across cases with a human making final decisions. Under this experimental design, we show how to compare the performance of three alternative decision-making systems--human-alone, human-with-AI, and AI-alone. We apply the pr
    
[^2]: 多元化的打破平局设计

    Multivariate Tie-breaker Designs

    [https://arxiv.org/abs/2202.10030](https://arxiv.org/abs/2202.10030)

    该研究探讨了一种多元回归的打破平局设计，通过优化D-最优准则的治疗概率，实现了资源分配效率和统计效率的权衡。

    

    在打破平局设计（TBD）中，具有一定高值的运行变量的受试者接受某种（通常是理想的）治疗，低值的受试者不接受治疗，而中间的受试者被随机分配。 TBD介于回归断点设计（RDD）和随机对照试验（RCT）之间，通过允许在RDD的资源分配效率与RCT的统计效率之间进行权衡。 我们研究了一个模型，其中被治疗受试者的预期反应是一个多元回归，而对照受试者则是另一个。 对于给定的协变量，我们展示了如何使用凸优化来选择优化D-最优准则的治疗概率。 我们可以结合多种受经济和伦理考虑激发的约束条件。 在我们的模型中，对于治疗效应的D-最优性与整体回归的D-最优性重合，在没有经济约束的情况下，RCT即为最优选择。

    arXiv:2202.10030v4 Announce Type: replace-cross  Abstract: In a tie-breaker design (TBD), subjects with high values of a running variable are given some (usually desirable) treatment, subjects with low values are not, and subjects in the middle are randomized. TBDs are intermediate between regression discontinuity designs (RDDs) and randomized controlled trials (RCTs) by allowing a tradeoff between the resource allocation efficiency of an RDD and the statistical efficiency of an RCT. We study a model where the expected response is one multivariate regression for treated subjects and another for control subjects. For given covariates, we show how to use convex optimization to choose treatment probabilities that optimize a D-optimality criterion. We can incorporate a variety of constraints motivated by economic and ethical considerations. In our model, D-optimality for the treatment effect coincides with D-optimality for the whole regression, and without economic constraints, an RCT is g
    
[^3]: 具有未测混淆因素的广义线性模型的同时推断

    Simultaneous inference for generalized linear models with unmeasured confounders. (arXiv:2309.07261v1 [stat.ME])

    [http://arxiv.org/abs/2309.07261](http://arxiv.org/abs/2309.07261)

    本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。

    

    在基因组研究中，常常进行成千上万个同时假设检验，以确定差异表达的基因。然而，由于存在未测混淆因素，许多标准统计方法可能存在严重的偏差。本文研究了存在混淆效应时的多元广义线性模型的大规模假设检验问题。在任意混淆机制下，我们提出了一个统一的统计估计和推断方法，利用正交结构并将线性投影整合到三个关键阶段中。首先，利用多元响应变量分离边际和不相关的混淆效应，恢复混淆系数的列空间。随后，利用$\ell_1$正则化进行稀疏性估计，并强加正交性限制于混淆系数，联合估计潜在因子和主要效应。最后，我们结合投影和加权偏差校正步骤。

    Tens of thousands of simultaneous hypothesis tests are routinely performed in genomic studies to identify differentially expressed genes. However, due to unmeasured confounders, many standard statistical approaches may be substantially biased. This paper investigates the large-scale hypothesis testing problem for multivariate generalized linear models in the presence of confounding effects. Under arbitrary confounding mechanisms, we propose a unified statistical estimation and inference framework that harnesses orthogonal structures and integrates linear projections into three key stages. It first leverages multivariate responses to separate marginal and uncorrelated confounding effects, recovering the confounding coefficients' column space. Subsequently, latent factors and primary effects are jointly estimated, utilizing $\ell_1$-regularization for sparsity while imposing orthogonality onto confounding coefficients. Finally, we incorporate projected and weighted bias-correction steps 
    
[^4]: 克服异方差PCA中病态问题的缩减算法

    Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA. (arXiv:2303.06198v1 [math.ST])

    [http://arxiv.org/abs/2303.06198](http://arxiv.org/abs/2303.06198)

    本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。

    This paper proposes a novel algorithm, called Deflated-HeteroPCA, that overcomes the curse of ill-conditioning in heteroskedastic PCA while achieving near-optimal and condition-number-free theoretical guarantees.

    本文关注于从受污染的数据中估计低秩矩阵X*的列子空间。当存在异方差噪声和不平衡的维度（即n2 >> n1）时，如何在容纳最广泛的信噪比范围的同时获得最佳的统计精度变得特别具有挑战性。虽然最先进的算法HeteroPCA成为解决这个问题的强有力的解决方案，但它遭受了“病态问题的诅咒”，即随着X*的条件数增长，其性能会下降。为了克服这个关键问题而不影响允许的信噪比范围，我们提出了一种新的算法，称为缩减异方差PCA，它在$\ell_2$和$\ell_{2,\infty}$统计精度方面实现了近乎最优和无条件数的理论保证。所提出的算法将谱分成两部分

    This paper is concerned with estimating the column subspace of a low-rank matrix $\boldsymbol{X}^\star \in \mathbb{R}^{n_1\times n_2}$ from contaminated data. How to obtain optimal statistical accuracy while accommodating the widest range of signal-to-noise ratios (SNRs) becomes particularly challenging in the presence of heteroskedastic noise and unbalanced dimensionality (i.e., $n_2\gg n_1$). While the state-of-the-art algorithm $\textsf{HeteroPCA}$ emerges as a powerful solution for solving this problem, it suffers from "the curse of ill-conditioning," namely, its performance degrades as the condition number of $\boldsymbol{X}^\star$ grows. In order to overcome this critical issue without compromising the range of allowable SNRs, we propose a novel algorithm, called $\textsf{Deflated-HeteroPCA}$, that achieves near-optimal and condition-number-free theoretical guarantees in terms of both $\ell_2$ and $\ell_{2,\infty}$ statistical accuracy. The proposed algorithm divides the spectrum
    
[^5]: 带有错位处理采用的合成对照中的不确定性量化

    Uncertainty Quantification in Synthetic Controls with Staggered Treatment Adoption. (arXiv:2210.05026v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.05026](http://arxiv.org/abs/2210.05026)

    该论文提出了一种基于原则的预测区间方法，用于量化合成对照预测或估计在错位处理采用的情况下的不确定性。

    

    我们提出了基于原则的预测区间，用于量化在错位处理采用的情况下大类合成对照预测或估计的不确定性，提供精确的非渐近覆盖概率保证。从方法论的角度来看，我们提供了对需要预测的不同因果量进行详细讨论，我们称其为“因果预测量”，允许在可能不同时刻进行多个处理单元的处理采用。从理论的角度来看，我们的不确定性量化方法提高了之前文献的水平，具体表现在：（i）覆盖了错位采用设置中的大类因果预测量，（ii）允许具有可能非线性约束的合成对照方法，（iii）提出可扩展的鲁棒锥优化方法和基于原则的数据驱动调参选择，（iv）提供了在后处理期间进行有效均匀推断。我们通过实证应用展示了我们的方法。

    We propose principled prediction intervals to quantify the uncertainty of a large class of synthetic control predictions or estimators in settings with staggered treatment adoption, offering precise non-asymptotic coverage probability guarantees. From a methodological perspective, we provide a detailed discussion of different causal quantities to be predicted, which we call `causal predictands', allowing for multiple treated units with treatment adoption at possibly different points in time. From a theoretical perspective, our uncertainty quantification methods improve on prior literature by (i) covering a large class of causal predictands in staggered adoption settings, (ii) allowing for synthetic control methods with possibly nonlinear constraints, (iii) proposing scalable robust conic optimization methods and principled data-driven tuning parameter selection, and (iv) offering valid uniform inference across post-treatment periods. We illustrate our methodology with an empirical appl
    

