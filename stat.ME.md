# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Consistent Validation for Predictive Methods in Spatial Settings](https://arxiv.org/abs/2402.03527) | 本论文研究了在空间环境中验证预测方法的一致性问题，提出了一种能够处理不匹配情况的方法。 |
| [^2] | [Sample-Efficient Clustering and Conquer Procedures for Parallel Large-Scale Ranking and Selection](https://arxiv.org/abs/2402.02196) | 我们提出了一种新颖的并行大规模排序和选择问题的聚类及征服方法，通过利用相关信息进行聚类以提高样本效率，在大规模AI应用中表现优异。 |
| [^3] | [Distributionally Robust Machine Learning with Multi-source Data.](http://arxiv.org/abs/2309.02211) | 本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。 |
| [^4] | [Causality-oriented robustness: exploiting general additive interventions.](http://arxiv.org/abs/2307.10299) | 本文提出了一种名为DRIG的方法，通过利用训练数据中的一般性可加干预，在预测模型中结合了内分布预测和因果性，从而实现了对未见干预的鲁棒预测。 |
| [^5] | [Invariant Causal Set Covering Machines.](http://arxiv.org/abs/2306.04777) | 本文提出了一种名为不变因果集覆盖机的算法，它避免了产生虚假关联，可以在多项式时间内识别感兴趣变量的因果父节点。 |
| [^6] | [The Local Approach to Causal Inference under Network Interference.](http://arxiv.org/abs/2105.03810) | 我们提出了一种新的非参数建模框架，用于网络干扰条件下的因果推断，通过对代理人之间的连接方式进行建模和学习政策或治疗分配的影响。我们还提出了一种有效的测试方法来检验政策无关性/治疗效应，并对平均或分布式政策效应/治疗反应的估计器给出了上界。 |

# 详细

[^1]: 在空间环境中一致验证预测方法

    Consistent Validation for Predictive Methods in Spatial Settings

    [https://arxiv.org/abs/2402.03527](https://arxiv.org/abs/2402.03527)

    本论文研究了在空间环境中验证预测方法的一致性问题，提出了一种能够处理不匹配情况的方法。

    

    空间预测任务对于天气预报、空气污染研究和其他科学工作至关重要。确定我们对统计或物理方法所作预测的可信度是科学结论的重要问题。不幸的是，传统的验证方法无法处理验证位置和我们希望进行预测的（测试）位置之间的不匹配。这种不匹配通常不是协变量偏移的一个实例（常常被形式化），因为验证和测试位置是固定的（例如，在网格上或选定的点上），而不是从两个分布中独立同分布地采样。在本文中，我们形式化了对验证方法的检查：随着验证数据的密度越来越大，它们能够变得任意精确。我们证明了传统方法和协变量偏移方法可能不满足这个检查。相反，我们提出了一种方法，它借鉴了协变量偏移文献中的现有思想，但对验证数据进行了调整。

    Spatial prediction tasks are key to weather forecasting, studying air pollution, and other scientific endeavors. Determining how much to trust predictions made by statistical or physical methods is essential for the credibility of scientific conclusions. Unfortunately, classical approaches for validation fail to handle mismatch between locations available for validation and (test) locations where we want to make predictions. This mismatch is often not an instance of covariate shift (as commonly formalized) because the validation and test locations are fixed (e.g., on a grid or at select points) rather than i.i.d. from two distributions. In the present work, we formalize a check on validation methods: that they become arbitrarily accurate as validation data becomes arbitrarily dense. We show that classical and covariate-shift methods can fail this check. We instead propose a method that builds from existing ideas in the covariate-shift literature, but adapts them to the validation data 
    
[^2]: 并行大规模排序和选择问题的样本高效聚类及征服方法

    Sample-Efficient Clustering and Conquer Procedures for Parallel Large-Scale Ranking and Selection

    [https://arxiv.org/abs/2402.02196](https://arxiv.org/abs/2402.02196)

    我们提出了一种新颖的并行大规模排序和选择问题的聚类及征服方法，通过利用相关信息进行聚类以提高样本效率，在大规模AI应用中表现优异。

    

    我们提出了一种新颖的"聚类和征服"方法，用于解决并行大规模排序和选择问题，通过利用相关信息进行聚类，以打破样本效率的瓶颈。在并行计算环境中，基于相关性的聚类可以实现O(p)的样本复杂度减少速度，这是理论上可达到的最佳减少速度。我们提出的框架是通用的，在固定预算和固定精度的范式下，可以无缝集成各种常见的排序和选择方法。它可以在无需高精确度相关估计和精确聚类的情况下实现改进。在大规模人工智能应用中，如神经结构搜索，我们的无筛选版本的方法惊人地超过了完全顺序化的基准，表现出更高的样本效率。这表明利用有价值的结构信息，如相关性，是绕过传统方法的一条可行路径。

    We propose novel "clustering and conquer" procedures for the parallel large-scale ranking and selection (R&S) problem, which leverage correlation information for clustering to break the bottleneck of sample efficiency. In parallel computing environments, correlation-based clustering can achieve an $\mathcal{O}(p)$ sample complexity reduction rate, which is the optimal reduction rate theoretically attainable. Our proposed framework is versatile, allowing for seamless integration of various prevalent R&S methods under both fixed-budget and fixed-precision paradigms. It can achieve improvements without the necessity of highly accurate correlation estimation and precise clustering. In large-scale AI applications such as neural architecture search, a screening-free version of our procedure surprisingly surpasses fully-sequential benchmarks in terms of sample efficiency. This suggests that leveraging valuable structural information, such as correlation, is a viable path to bypassing the trad
    
[^3]: 基于多源数据的分布鲁棒机器学习

    Distributionally Robust Machine Learning with Multi-source Data. (arXiv:2309.02211v1 [stat.ML])

    [http://arxiv.org/abs/2309.02211](http://arxiv.org/abs/2309.02211)

    本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。

    

    当目标分布与源数据集不同时，传统的机器学习方法可能导致较差的预测性能。本文利用多个数据源，并引入了一种基于组分布鲁棒预测模型来优化关于目标分布类的可解释方差的对抗性奖励。与传统的经验风险最小化相比，所提出的鲁棒预测模型改善了具有分布偏移的目标人群的预测准确性。我们证明了组分布鲁棒预测模型是源数据集条件结果模型的加权平均。我们利用这一关键鉴别结果来提高任意机器学习算法的鲁棒性，包括随机森林和神经网络等。我们设计了一种新的偏差校正估计器来估计通用机器学习算法的最优聚合权重，并展示了其在c方面的改进。

    Classical machine learning methods may lead to poor prediction performance when the target distribution differs from the source populations. This paper utilizes data from multiple sources and introduces a group distributionally robust prediction model defined to optimize an adversarial reward about explained variance with respect to a class of target distributions. Compared to classical empirical risk minimization, the proposed robust prediction model improves the prediction accuracy for target populations with distribution shifts. We show that our group distributionally robust prediction model is a weighted average of the source populations' conditional outcome models. We leverage this key identification result to robustify arbitrary machine learning algorithms, including, for example, random forests and neural networks. We devise a novel bias-corrected estimator to estimate the optimal aggregation weight for general machine-learning algorithms and demonstrate its improvement in the c
    
[^4]: 因果性导向的鲁棒性：利用一般性可加干预

    Causality-oriented robustness: exploiting general additive interventions. (arXiv:2307.10299v1 [stat.ME])

    [http://arxiv.org/abs/2307.10299](http://arxiv.org/abs/2307.10299)

    本文提出了一种名为DRIG的方法，通过利用训练数据中的一般性可加干预，在预测模型中结合了内分布预测和因果性，从而实现了对未见干预的鲁棒预测。

    

    由于在现实应用中经常发生分布变化，急需开发对这种变化具有鲁棒性的预测模型。现有的框架，如经验风险最小化或分布鲁棒优化，要么对未见分布缺乏通用性，要么依赖于假定的距离度量。相比之下，因果性提供了一种基于数据和结构的稳健预测方法。然而，进行因果推断所需的假设可能过于严格，这种因果模型提供的鲁棒性常常缺乏灵活性。在本文中，我们专注于因果性导向的鲁棒性，并提出了一种名为DRIG（Distributional Robustness via Invariant Gradients）的方法，该方法利用训练数据中的一般性可加干预，以实现对未见干预的鲁棒预测，并在内分布预测和因果性之间自然地进行插值。在线性设置中，我们证明了DRIG产生的预测是

    Since distribution shifts are common in real-world applications, there is a pressing need for developing prediction models that are robust against such shifts. Existing frameworks, such as empirical risk minimization or distributionally robust optimization, either lack generalizability for unseen distributions or rely on postulated distance measures. Alternatively, causality offers a data-driven and structural perspective to robust predictions. However, the assumptions necessary for causal inference can be overly stringent, and the robustness offered by such causal models often lacks flexibility. In this paper, we focus on causality-oriented robustness and propose Distributional Robustness via Invariant Gradients (DRIG), a method that exploits general additive interventions in training data for robust predictions against unseen interventions, and naturally interpolates between in-distribution prediction and causality. In a linear setting, we prove that DRIG yields predictions that are 
    
[^5]: 不变因果集覆盖机

    Invariant Causal Set Covering Machines. (arXiv:2306.04777v1 [cs.LG])

    [http://arxiv.org/abs/2306.04777](http://arxiv.org/abs/2306.04777)

    本文提出了一种名为不变因果集覆盖机的算法，它避免了产生虚假关联，可以在多项式时间内识别感兴趣变量的因果父节点。

    

    基于规则的模型，如决策树，因其可解释的特性受到从业者的欢迎。然而，产生这种模型的学习算法往往容易受到虚假关联的影响，因此不能保证提取的是具有因果关系的洞见。在这项工作中，我们借鉴了不变因果预测文献中的思想，提出了不变的因果集覆盖机，这是一种经典的集覆盖机算法的扩展，用于二值规则的合取/析取，可以证明它避免了虚假关联。我们理论上和实践上证明，我们的方法可以在多项式时间内识别感兴趣变量的因果父节点。

    Rule-based models, such as decision trees, appeal to practitioners due to their interpretable nature. However, the learning algorithms that produce such models are often vulnerable to spurious associations and thus, they are not guaranteed to extract causally-relevant insights. In this work, we build on ideas from the invariant causal prediction literature to propose Invariant Causal Set Covering Machines, an extension of the classical Set Covering Machine algorithm for conjunctions/disjunctions of binary-valued rules that provably avoids spurious associations. We demonstrate both theoretically and empirically that our method can identify the causal parents of a variable of interest in polynomial time.
    
[^6]: 网络干扰条件下因果推断的局部方法

    The Local Approach to Causal Inference under Network Interference. (arXiv:2105.03810v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2105.03810](http://arxiv.org/abs/2105.03810)

    我们提出了一种新的非参数建模框架，用于网络干扰条件下的因果推断，通过对代理人之间的连接方式进行建模和学习政策或治疗分配的影响。我们还提出了一种有效的测试方法来检验政策无关性/治疗效应，并对平均或分布式政策效应/治疗反应的估计器给出了上界。

    

    我们提出了一种新的非参数建模框架，用于处理社交或经济网络中代理人之间连接方式对结果产生影响的因果推断问题。这种网络干扰描述了关于治疗溢出、社交互动、社会学习、信息扩散、疾病和金融传染、社会资本形成等领域的大量文献。我们的方法首先通过测量路径距离来描述代理人在网络中的连接方式，然后通过汇集具有类似配置的代理人的结果数据来学习政策或治疗分配的影响。我们通过提出一个渐近有效的测试来演示该方法，该测试用于检验政策无关性/治疗效应的假设，并给出了针对平均或分布式政策效应/治疗反应的k最近邻估计器的均方误差的上界。

    We propose a new nonparametric modeling framework for causal inference when outcomes depend on how agents are linked in a social or economic network. Such network interference describes a large literature on treatment spillovers, social interactions, social learning, information diffusion, disease and financial contagion, social capital formation, and more. Our approach works by first characterizing how an agent is linked in the network using the configuration of other agents and connections nearby as measured by path distance. The impact of a policy or treatment assignment is then learned by pooling outcome data across similarly configured agents. We demonstrate the approach by proposing an asymptotically valid test for the hypothesis of policy irrelevance/no treatment effects and bounding the mean-squared error of a k-nearest-neighbor estimator for the average or distributional policy effect/treatment response.
    

