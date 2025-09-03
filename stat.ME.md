# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^2] | [Statistical Performance Guarantee for Selecting Those Predicted to Benefit Most from Treatment.](http://arxiv.org/abs/2310.07973) | 本研究针对选择最有可能从治疗中获益的人的问题，在估计治疗效果和确定截断值时面临多重测试问题，提出一种统一的置信带方法来评估这些个体的平均治疗效果。 |
| [^3] | [Causal Effects in Matching Mechanisms with Strategically Reported Preferences.](http://arxiv.org/abs/2307.14282) | 本文提供一种考虑了策略性误报的因果效应识别方法，可以对学校分配对未来结果产生的影响进行准确度量。我们的方法适用于各种机制，并能够得到对策略行为鲁棒的因果效应的尖锐边界。 |
| [^4] | [A Flexible Framework for Incorporating Patient Preferences Into Q-Learning.](http://arxiv.org/abs/2307.12022) | 这个论文提出了一种称为潜在效用Q学习的方法，能够将患者偏好纳入复合结果的动态治疗方案中，解决了传统方法对时间点和结果数量的限制，能够实现强大的性能。 |
| [^5] | [K-Tensors: Clustering Positive Semi-Definite Matrices.](http://arxiv.org/abs/2306.06534) | 本文介绍了一种针对正半定矩阵的自一致性聚类算法（K-张量），通过考虑其特征结构，能够有效地将正半定矩阵进行分区。 |

# 详细

[^1]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^2]: 对于选择最有可能从治疗中获益的人的统计性能保证

    Statistical Performance Guarantee for Selecting Those Predicted to Benefit Most from Treatment. (arXiv:2310.07973v1 [stat.ME])

    [http://arxiv.org/abs/2310.07973](http://arxiv.org/abs/2310.07973)

    本研究针对选择最有可能从治疗中获益的人的问题，在估计治疗效果和确定截断值时面临多重测试问题，提出一种统一的置信带方法来评估这些个体的平均治疗效果。

    

    在广泛的学科领域中，许多研究人员使用机器学习算法来识别一组被称为例外反应者的个体，他们最有可能从治疗中获益。一个常见的方法包括两个步骤。首先使用机器学习算法估计条件平均治疗效果或其代理。然后确定所得治疗优先顺序分数的截断值，以选择那些最有可能从治疗中获益的人。不幸的是，这些估计的治疗优先顺序分数往往存在偏差和噪声。此外，利用相同的数据既选择截断值又估计所选个体的平均治疗效果会遇到多重测试问题。为了解决这些挑战，我们开发了一个统一的置信带来实验性地评估那些治疗优先顺序分数至少与任何给定量化值相等的个体的排序平均治疗效果（GATES）。

    Across a wide array of disciplines, many researchers use machine learning (ML) algorithms to identify a subgroup of individuals, called exceptional responders, who are likely to be helped by a treatment the most. A common approach consists of two steps. One first estimates the conditional average treatment effect or its proxy using an ML algorithm. They then determine the cutoff of the resulting treatment prioritization score to select those predicted to benefit most from the treatment. Unfortunately, these estimated treatment prioritization scores are often biased and noisy. Furthermore, utilizing the same data to both choose a cutoff value and estimate the average treatment effect among the selected individuals suffer from a multiple testing problem. To address these challenges, we develop a uniform confidence band for experimentally evaluating the sorted average treatment effect (GATES) among the individuals whose treatment prioritization score is at least as high as any given quant
    
[^3]: 匹配机制中的因果效应与策略性报告偏好

    Causal Effects in Matching Mechanisms with Strategically Reported Preferences. (arXiv:2307.14282v1 [econ.EM])

    [http://arxiv.org/abs/2307.14282](http://arxiv.org/abs/2307.14282)

    本文提供一种考虑了策略性误报的因果效应识别方法，可以对学校分配对未来结果产生的影响进行准确度量。我们的方法适用于各种机制，并能够得到对策略行为鲁棒的因果效应的尖锐边界。

    

    越来越多的中央机构使用分配机制将学生分配到学校，以反映学生的偏好和学校的优先权。然而，大多数现实世界的机制会给学生提供一种策略性并误报他们的偏好的激励。在本文中，我们提供了一种识别因果效应的方法，该方法考虑了策略性的误报。误报可能使现有的点识别方法无效，我们推导出对策略行为鲁棒的因果效应的尖锐边界。我们的方法适用于任何机制，只要存在描述该机制分配规则的配对分数和截点。我们使用智利一个延迟接受机制的数据，该机制将学生分配到1000多个大学专业组合。学生出于策略考虑而行动，因为智利的机制限制了学生在偏好中提交的专业数量为八个。

    A growing number of central authorities use assignment mechanisms to allocate students to schools in a way that reflects student preferences and school priorities. However, most real-world mechanisms give students an incentive to be strategic and misreport their preferences. In this paper, we provide an identification approach for causal effects of school assignment on future outcomes that accounts for strategic misreporting. Misreporting may invalidate existing point-identification approaches, and we derive sharp bounds for causal effects that are robust to strategic behavior. Our approach applies to any mechanism as long as there exist placement scores and cutoffs that characterize that mechanism's allocation rule. We use data from a deferred acceptance mechanism that assigns students to more than 1,000 university-major combinations in Chile. Students behave strategically because the mechanism in Chile constrains the number of majors that students submit in their preferences to eight
    
[^4]: 将患者偏好纳入Q学习的灵活框架

    A Flexible Framework for Incorporating Patient Preferences Into Q-Learning. (arXiv:2307.12022v1 [cs.LG])

    [http://arxiv.org/abs/2307.12022](http://arxiv.org/abs/2307.12022)

    这个论文提出了一种称为潜在效用Q学习的方法，能够将患者偏好纳入复合结果的动态治疗方案中，解决了传统方法对时间点和结果数量的限制，能够实现强大的性能。

    

    在现实世界的医疗问题中，通常存在多个竞争性的关注点，如治疗疗效和副作用严重程度。然而，用于估计动态治疗方案 (DTRs) 的统计方法通常假设只有一个关注点，而处理复合结果的方法很少，存在重要限制，包括对单个时间点和两个结果的限制、无法纳入患者的自述偏好以及有限的理论保证。为此，我们提出了一个新的方法来解决这些限制，我们称之为潜在效用Q学习(LUQ-Learning)。LUQ-Learning采用潜在模型方法，自然地将Q学习扩展到复合结果设置，并为每个患者选择理想的结果权衡。与之前的方法不同，我们的框架允许任意数量的时间点和结果，纳入陈述的偏好，并实现强大的渐近性能。

    In real-world healthcare problems, there are often multiple competing outcomes of interest, such as treatment efficacy and side effect severity. However, statistical methods for estimating dynamic treatment regimes (DTRs) usually assume a single outcome of interest, and the few methods that deal with composite outcomes suffer from important limitations. This includes restrictions to a single time point and two outcomes, the inability to incorporate self-reported patient preferences and limited theoretical guarantees. To this end, we propose a new method to address these limitations, which we dub Latent Utility Q-Learning (LUQ-Learning). LUQ-Learning uses a latent model approach to naturally extend Q-learning to the composite outcome setting and adopt the ideal trade-off between outcomes to each patient. Unlike previous approaches, our framework allows for an arbitrary number of time points and outcomes, incorporates stated preferences and achieves strong asymptotic performance with rea
    
[^5]: K-Tensors：对正半定矩阵进行聚类

    K-Tensors: Clustering Positive Semi-Definite Matrices. (arXiv:2306.06534v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.06534](http://arxiv.org/abs/2306.06534)

    本文介绍了一种针对正半定矩阵的自一致性聚类算法（K-张量），通过考虑其特征结构，能够有效地将正半定矩阵进行分区。

    

    本文介绍了一种新颖的自一致性聚类算法（K-Tensors），用于基于它们的特征结构将正半定矩阵进行分区。由于正半定矩阵可以在 p≥2 的空间中表示为椭球体，因此保持它们的结构信息以进行有效的聚类至关重要。然而，传统的矩阵聚类算法常常涉及将矩阵向量化，导致关键结构信息的丢失。为了解决这个问题，我们提出了一种基于正半定矩阵结构信息的距离度量来进行聚类。这种距离度量使得聚类算法能够考虑正半定矩阵与它们在由一组正半定矩阵定义的正交向量张成的共同空间上的投影之间的差异。这是一种创新的聚类方法。

    This paper introduces a novel self-consistency clustering algorithm ($K$-Tensors) designed for {partitioning a distribution of} positive-semidefinite matrices based on their eigenstructures. As positive semi-definite matrices can be represented as ellipsoids in $\Re^p$, $p \ge 2$, it is critical to maintain their structural information to perform effective clustering. However, traditional clustering algorithms {applied to matrices} often {involve vectorization of} the matrices, resulting in a loss of essential structural information. To address this issue, we propose a distance metric {for clustering} that is specifically based on the structural information of positive semi-definite matrices. This distance metric enables the clustering algorithm to consider the differences between positive semi-definite matrices and their projections onto {a} common space spanned by \thadJulyTen{orthonormal vectors defined from a set of} positive semi-definite matrices. This innovative approach to clus
    

