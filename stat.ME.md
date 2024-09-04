# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Neural Networks for Treatment Effect Prediction](https://arxiv.org/abs/2403.19289) | 提出了一种图神经网络来减少治疗效果预测所需的训练集大小，有效利用电子商务数据的图结构，为治疗效果预测带来新的可能性 |
| [^2] | [Adaptive Split Balancing for Optimal Random Forest](https://arxiv.org/abs/2402.11228) | 介绍了自适应分割平衡森林（ASBF），可在学习树表示的同时，在复杂情况下实现极小极优性，并提出了一个本地化版本，在H\"older类下达到最小极优性。 |
| [^3] | [Scalable Estimation of Multinomial Response Models with Uncertain Consideration Sets.](http://arxiv.org/abs/2308.12470) | 这篇论文提出了一种克服估计多项式响应模型中指数级支持问题的方法，通过使用基于列联表概率分布的考虑集概率模型。 |
| [^4] | [Long-term Causal Inference Under Persistent Confounding via Data Combination.](http://arxiv.org/abs/2202.07234) | 本研究通过数据组合解决了长期治疗效果识别和估计中的持续未测量混淆因素挑战，并提出了三种新的识别策略和估计器。 |

# 详细

[^1]: 用于治疗效果预测的图神经网络

    Graph Neural Networks for Treatment Effect Prediction

    [https://arxiv.org/abs/2403.19289](https://arxiv.org/abs/2403.19289)

    提出了一种图神经网络来减少治疗效果预测所需的训练集大小，有效利用电子商务数据的图结构，为治疗效果预测带来新的可能性

    

    在电子商务中估计因果效应往往涉及昂贵的治疗分配，这在大规模设置中可能是不切实际的。利用机器学习来预测这种治疗效果而无需实际干预是减少风险的一种标准做法。然而，现有的治疗效果预测方法往往依赖于大规模实验构建的训练集，因此从根本上存在风险。在这项工作中，我们提出了一种图神经网络，以减少所需的训练集大小，依赖于电子商务数据中常见的图。具体地，我们将问题视为具有有限数量标记实例的节点回归，开发了一个类似于先前因果效应估计器的双模型神经架构，并测试了不同的消息传递层进行编码。此外，作为额外步骤，我们将模型与获取函数相结合，以引导信息传递。

    arXiv:2403.19289v1 Announce Type: cross  Abstract: Estimating causal effects in e-commerce tends to involve costly treatment assignments which can be impractical in large-scale settings. Leveraging machine learning to predict such treatment effects without actual intervention is a standard practice to diminish the risk. However, existing methods for treatment effect prediction tend to rely on training sets of substantial size, which are built from real experiments and are thus inherently risky to create. In this work we propose a graph neural network to diminish the required training set size, relying on graphs that are common in e-commerce data. Specifically, we view the problem as node regression with a restricted number of labeled instances, develop a two-model neural architecture akin to previous causal effect estimators, and test varying message-passing layers for encoding. Furthermore, as an extra step, we combine the model with an acquisition function to guide the creation of th
    
[^2]: 自适应分割平衡优化随机森林

    Adaptive Split Balancing for Optimal Random Forest

    [https://arxiv.org/abs/2402.11228](https://arxiv.org/abs/2402.11228)

    介绍了自适应分割平衡森林（ASBF），可在学习树表示的同时，在复杂情况下实现极小极优性，并提出了一个本地化版本，在H\"older类下达到最小极优性。

    

    尽管随机森林通常用于回归问题，但现有方法在复杂情况下缺乏适应性，或在简单、平滑情景下失去最优性。在本研究中，我们介绍了自适应分割平衡森林（ASBF），能够从数据中学习树表示，同时在Lipschitz类下实现极小极优性。为了利用更高阶的平滑性水平，我们进一步提出了一个本地化版本，该版本在任意$q \in \mathbb{N}$和$\beta \in (0,1]$的Hölder类$\mathcal{H}^{q,\beta}$下达到最小极优性。与广泛使用的随机特征选择不同，我们考虑了对现有方法的平衡修改。我们的结果表明，过度依赖辅助随机性可能会损害树模型的逼近能力，导致次优结果。相反，一个更平衡、更少随机的方法表现出最佳性能。

    arXiv:2402.11228v1 Announce Type: cross  Abstract: While random forests are commonly used for regression problems, existing methods often lack adaptability in complex situations or lose optimality under simple, smooth scenarios. In this study, we introduce the adaptive split balancing forest (ASBF), capable of learning tree representations from data while simultaneously achieving minimax optimality under the Lipschitz class. To exploit higher-order smoothness levels, we further propose a localized version that attains the minimax rate under the H\"older class $\mathcal{H}^{q,\beta}$ for any $q\in\mathbb{N}$ and $\beta\in(0,1]$. Rather than relying on the widely-used random feature selection, we consider a balanced modification to existing approaches. Our results indicate that an over-reliance on auxiliary randomness may compromise the approximation power of tree models, leading to suboptimal results. Conversely, a less random, more balanced approach demonstrates optimality. Additionall
    
[^3]: 可伸缩估计具有不确定的选项集的多项式响应模型

    Scalable Estimation of Multinomial Response Models with Uncertain Consideration Sets. (arXiv:2308.12470v1 [stat.ME])

    [http://arxiv.org/abs/2308.12470](http://arxiv.org/abs/2308.12470)

    这篇论文提出了一种克服估计多项式响应模型中指数级支持问题的方法，通过使用基于列联表概率分布的考虑集概率模型。

    

    在交叉或纵向数据的无序多项式响应模型拟合中的一个标准假设是，响应来自于相同的J个类别集合。然而，当响应度量主体做出的选择时，更适合假设多项式响应的分布是在主体特定的考虑集条件下，其中这个考虑集是从{1,2, ..., J}的幂集中抽取的。由于这个幂集的基数在J中是指数级的，一般来说估计是无法实现的。在本文中，我们提供了一种克服这个问题的方法。这种方法中的一个关键步骤是基于在列联表上的概率分布的一般表示的考虑集的概率模型。尽管这个分布的支持是指数级大的，但给定参数的考虑集的后验分布通常是稀疏的。

    A standard assumption in the fitting of unordered multinomial response models for J mutually exclusive nominal categories, on cross-sectional or longitudinal data, is that the responses arise from the same set of J categories between subjects. However, when responses measure a choice made by the subject, it is more appropriate to assume that the distribution of multinomial responses is conditioned on a subject-specific consideration set, where this consideration set is drawn from the power set of {1,2,...,J}. Because the cardinality of this power set is exponential in J, estimation is infeasible in general. In this paper, we provide an approach to overcoming this problem. A key step in the approach is a probability model over consideration sets, based on a general representation of probability distributions on contingency tables. Although the support of this distribution is exponentially large, the posterior distribution over consideration sets given parameters is typically sparse, and
    
[^4]: 长期持续混淆情况下的因果推断与数据组合研究

    Long-term Causal Inference Under Persistent Confounding via Data Combination. (arXiv:2202.07234v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2202.07234](http://arxiv.org/abs/2202.07234)

    本研究通过数据组合解决了长期治疗效果识别和估计中的持续未测量混淆因素挑战，并提出了三种新的识别策略和估计器。

    

    我们研究了当实验数据和观察数据同时存在时，长期治疗效果的识别和估计问题。由于长期结果仅在长时间延迟后才观察到，在实验数据中无法测量，但在观察数据中有记录。然而，这两种类型的数据都包含对一些短期结果的观察。在本文中，我们独特地解决了持续未测量混淆因素的挑战，即一些未测量混淆因素可以同时影响治疗、短期结果和长期结果，而这会使得之前文献中的识别策略无效。为了解决这个挑战，我们利用多个短期结果的连续结构，为平均长期治疗效果提出了三种新的识别策略。我们进一步提出了三种对应的估计器，并证明了它们的渐近一致性和渐近正态性。最后，我们将我们的方法应用于估计长期治疗效果。

    We study the identification and estimation of long-term treatment effects when both experimental and observational data are available. Since the long-term outcome is observed only after a long delay, it is not measured in the experimental data, but only recorded in the observational data. However, both types of data include observations of some short-term outcomes. In this paper, we uniquely tackle the challenge of persistent unmeasured confounders, i.e., some unmeasured confounders that can simultaneously affect the treatment, short-term outcomes and the long-term outcome, noting that they invalidate identification strategies in previous literature. To address this challenge, we exploit the sequential structure of multiple short-term outcomes, and develop three novel identification strategies for the average long-term treatment effect. We further propose three corresponding estimators and prove their asymptotic consistency and asymptotic normality. We finally apply our methods to esti
    

