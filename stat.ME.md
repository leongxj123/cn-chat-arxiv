# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Auditing Fairness under Unobserved Confounding](https://arxiv.org/abs/2403.14713) | 在未观测混杂因素的情况下，本文展示了即使在放宽或甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以给出对高风险个体分配率的信息丰富的界限。 |
| [^2] | [A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding](https://arxiv.org/abs/2402.02306) | 本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。 |
| [^3] | [Comparative Study of Causal Discovery Methods for Cyclic Models with Hidden Confounders.](http://arxiv.org/abs/2401.13009) | 对于循环模型中含有隐藏因变量的因果发现，已经出现了能够处理这种情况的多种技术方法。 |
| [^4] | [Estimating Effects of Long-Term Treatments.](http://arxiv.org/abs/2308.08152) | 本论文介绍了一个纵向替代框架，用于准确估计长期治疗的效果。论文通过分解长期治疗效果为一系列函数，考虑用户属性、短期中间指标和治疗分配等因素。 |
| [^5] | [One-step nonparametric instrumental regression using smoothing splines.](http://arxiv.org/abs/2307.14867) | 这个论文提出了一种一步非参数的方法，使用平滑样条来处理内生性和仪器变量，同时解决了单调性限制的问题，并在估计Engel曲线时表现出良好性能。 |
| [^6] | [Statistical Tests for Replacing Human Decision Makers with Algorithms.](http://arxiv.org/abs/2306.11689) | 本文提出了一种利用人工智能改善人类决策的统计框架，通过基准测试与机器预测，替换部分人类决策者的决策制定，并经过实验检验得出算法具有更高的真阳性率和更低的假阳性率，尤其是来自农村地区的医生的诊断更容易被替代。 |

# 详细

[^1]: 在未观测混杂因素下审计公平性

    Auditing Fairness under Unobserved Confounding

    [https://arxiv.org/abs/2403.14713](https://arxiv.org/abs/2403.14713)

    在未观测混杂因素的情况下，本文展示了即使在放宽或甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以给出对高风险个体分配率的信息丰富的界限。

    

    决策系统中的一个基本问题是跨越人口统计线存在不公平性。然而，不公平性可能难以量化，特别是如果我们对公平性的理解依赖于难以衡量的风险等观念（例如，对于那些没有其治疗就会死亡的人平等获得治疗）。审计这种不公平性需要准确测量个体风险，而在未观测混杂的现实环境中，难以估计。在这些未观测到的因素“解释”明显差异的情况下，我们可能低估或高估不公平性。在本文中，我们展示了即使在放宽或（令人惊讶地）甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以对高风险个体的分配率给出信息丰富的界限。我们利用了在许多实际环境中（例如引入新型治疗）我们拥有在任何分配之前的数据的事实。

    arXiv:2403.14713v1 Announce Type: cross  Abstract: A fundamental problem in decision-making systems is the presence of inequity across demographic lines. However, inequity can be difficult to quantify, particularly if our notion of equity relies on hard-to-measure notions like risk (e.g., equal access to treatment for those who would die without it). Auditing such inequity requires accurate measurements of individual risk, which is difficult to estimate in the realistic setting of unobserved confounding. In the case that these unobservables "explain" an apparent disparity, we may understate or overstate inequity. In this paper, we show that one can still give informative bounds on allocation rates among high-risk individuals, even while relaxing or (surprisingly) even when eliminating the assumption that all relevant risk factors are observed. We utilize the fact that in many real-world settings (e.g., the introduction of a novel treatment) we have data from a period prior to any alloc
    
[^2]: 弹性贝叶斯g形式在具有时变混杂的因果生存分析中的应用

    A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding

    [https://arxiv.org/abs/2402.02306](https://arxiv.org/abs/2402.02306)

    本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。

    

    在具有时间至事件结果的纵向观察性研究中，因果分析的常见目标是在研究群体中估计在假设干预情景下的因果生存曲线。g形式是这种分析的一个特别有用的工具。为了增强传统的参数化g形式方法，我们开发了一种更灵活的贝叶斯g形式估计器。该估计器同时支持纵向预测和因果推断。它在模拟时变生成组件的建模中引入了贝叶斯附加回归树，旨在减轻由于模型错误规范造成的偏差。具体而言，我们引入了一类更通用的离散生存数据g形式。这些公式可以引入纵向平衡分数，这在处理越来越多的时变混杂因素时是一种有效的降维方法。

    In longitudinal observational studies with a time-to-event outcome, a common objective in causal analysis is to estimate the causal survival curve under hypothetical intervention scenarios within the study cohort. The g-formula is a particularly useful tool for this analysis. To enhance the traditional parametric g-formula approach, we developed a more adaptable Bayesian g-formula estimator. This estimator facilitates both longitudinal predictive and causal inference. It incorporates Bayesian additive regression trees in the modeling of the time-evolving generative components, aiming to mitigate bias due to model misspecification. Specifically, we introduce a more general class of g-formulas for discrete survival data. These formulas can incorporate the longitudinal balancing scores, which serve as an effective method for dimension reduction and are vital when dealing with an expanding array of time-varying confounders. The minimum sufficient formulation of these longitudinal balancing
    
[^3]: 循环模型中含有隐藏因变量的因果发现方法的比较研究

    Comparative Study of Causal Discovery Methods for Cyclic Models with Hidden Confounders. (arXiv:2401.13009v1 [cs.LG])

    [http://arxiv.org/abs/2401.13009](http://arxiv.org/abs/2401.13009)

    对于循环模型中含有隐藏因变量的因果发现，已经出现了能够处理这种情况的多种技术方法。

    

    如今，对因果发现的需求无处不在。理解系统中部分之间的随机依赖性以及实际的因果关系对科学的各个部分都至关重要。因此，寻找可靠的方法来检测因果方向的需求不断增长。在过去的50年里，出现了许多因果发现算法，但大多数仅适用于系统没有反馈环路并且具有因果充分性的假设，即没有未测量的子系统能够影响多个已测量变量。这是不幸的，因为这些限制在实践中往往不能假定。反馈是许多过程的一个重要特性，现实世界的系统很少是完全隔离和完全测量的。幸运的是，在最近几年中，已经发展了几种能够处理循环的、因果不充分的系统的技术。随着多种方法的出现，一种实际的应用方法开始变得可能。

    Nowadays, the need for causal discovery is ubiquitous. A better understanding of not just the stochastic dependencies between parts of a system, but also the actual cause-effect relations, is essential for all parts of science. Thus, the need for reliable methods to detect causal directions is growing constantly. In the last 50 years, many causal discovery algorithms have emerged, but most of them are applicable only under the assumption that the systems have no feedback loops and that they are causally sufficient, i.e. that there are no unmeasured subsystems that can affect multiple measured variables. This is unfortunate since those restrictions can often not be presumed in practice. Feedback is an integral feature of many processes, and real-world systems are rarely completely isolated and fully measured. Fortunately, in recent years, several techniques, that can cope with cyclic, causally insufficient systems, have been developed. And with multiple methods available, a practical ap
    
[^4]: 估计长期治疗效果

    Estimating Effects of Long-Term Treatments. (arXiv:2308.08152v1 [econ.EM])

    [http://arxiv.org/abs/2308.08152](http://arxiv.org/abs/2308.08152)

    本论文介绍了一个纵向替代框架，用于准确估计长期治疗的效果。论文通过分解长期治疗效果为一系列函数，考虑用户属性、短期中间指标和治疗分配等因素。

    

    在A/B测试中，估计长期治疗的效果是一个巨大的挑战。这种治疗措施包括产品功能的更新、用户界面设计和推荐算法等，旨在在其发布后长期存在系统中。然而，由于长期试验的限制，从业者通常依赖短期实验结果来做产品发布决策。如何使用短期实验数据准确估计长期治疗效果仍然是一个未解决的问题。为了解决这个问题，我们引入了一个纵向替代框架。我们展示了，在标准假设下，长期治疗效果可以分解为一系列函数，这些函数依赖于用户属性、短期中间指标和治疗分配。我们描述了识别假设、估计策略和推理技术。

    Estimating the effects of long-term treatments in A/B testing presents a significant challenge. Such treatments -- including updates to product functions, user interface designs, and recommendation algorithms -- are intended to remain in the system for a long period after their launches. On the other hand, given the constraints of conducting long-term experiments, practitioners often rely on short-term experimental results to make product launch decisions. It remains an open question how to accurately estimate the effects of long-term treatments using short-term experimental data. To address this question, we introduce a longitudinal surrogate framework. We show that, under standard assumptions, the effects of long-term treatments can be decomposed into a series of functions, which depend on the user attributes, the short-term intermediate metrics, and the treatment assignments. We describe the identification assumptions, the estimation strategies, and the inference technique under thi
    
[^5]: 一步非参数仪器回归使用平滑样条

    One-step nonparametric instrumental regression using smoothing splines. (arXiv:2307.14867v1 [econ.EM])

    [http://arxiv.org/abs/2307.14867](http://arxiv.org/abs/2307.14867)

    这个论文提出了一种一步非参数的方法，使用平滑样条来处理内生性和仪器变量，同时解决了单调性限制的问题，并在估计Engel曲线时表现出良好性能。

    

    我们将非参数回归平滑样条扩展到一种情境，即存在内生性和使用仪器变量。与流行的现有估计方法不同，结果估计器是一步的，并依赖于唯一的正则化参数。我们导出了估计器及其一阶导数的均匀收敛速率。我们还解决了在估计中施加单调性的问题。模拟结果证实了我们的估计器与两步程序相比的良好性能。当用于估计Engel曲线时，我们的方法产生了经济上有意义的结果。

    We extend nonparametric regression smoothing splines to a context where there is endogeneity and instrumental variables are available. Unlike popular existing estimators, the resulting estimator is one-step and relies on a unique regularization parameter. We derive uniform rates of the convergence for the estimator and its first derivative. We also address the issue of imposing monotonicity in estimation. Simulations confirm the good performances of our estimator compared to two-step procedures. Our method yields economically sensible results when used to estimate Engel curves.
    
[^6]: 统计测试替代人类决策者的算法

    Statistical Tests for Replacing Human Decision Makers with Algorithms. (arXiv:2306.11689v1 [econ.EM])

    [http://arxiv.org/abs/2306.11689](http://arxiv.org/abs/2306.11689)

    本文提出了一种利用人工智能改善人类决策的统计框架，通过基准测试与机器预测，替换部分人类决策者的决策制定，并经过实验检验得出算法具有更高的真阳性率和更低的假阳性率，尤其是来自农村地区的医生的诊断更容易被替代。

    

    本文提出了一个统计框架，可以通过人工智能来改善人类的决策。首先将每个人类决策者的表现与机器预测进行基准测试；然后用所提出的人工智能算法的建议替换决策制定者的一个子集所做出的决策。利用全国大型孕产结果和繁殖年龄夫妇孕前检查的医生诊断数据集，我们试验了一种启发式高频率方法以及一种贝叶斯后验损失函数方法，并将其应用于异常出生检测。我们发现，我们的算法在一个测试数据集上的结果比仅由医生诊断的结果具有更高的总体真阳性率和更低的假阳性率。我们还发现，来自农村地区的医生的诊断更容易被替代，这表明人工智能辅助决策制定更容易提高精确度。

    This paper proposes a statistical framework with which artificial intelligence can improve human decision making. The performance of each human decision maker is first benchmarked against machine predictions; we then replace the decisions made by a subset of the decision makers with the recommendation from the proposed artificial intelligence algorithm. Using a large nationwide dataset of pregnancy outcomes and doctor diagnoses from prepregnancy checkups of reproductive age couples, we experimented with both a heuristic frequentist approach and a Bayesian posterior loss function approach with an application to abnormal birth detection. We find that our algorithm on a test dataset results in a higher overall true positive rate and a lower false positive rate than the diagnoses made by doctors only. We also find that the diagnoses of doctors from rural areas are more frequently replaceable, suggesting that artificial intelligence assisted decision making tends to improve precision more i
    

