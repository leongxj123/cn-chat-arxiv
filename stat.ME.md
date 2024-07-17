# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rao-Blackwellising Bayesian Causal Inference](https://arxiv.org/abs/2402.14781) | 本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。 |
| [^2] | [Online Local False Discovery Rate Control: A Resource Allocation Approach](https://arxiv.org/abs/2402.11425) | 该研究提出了一种在线局部虚发现率控制的资源分配方法，实现了$O(\sqrt{T})$的后悔率，并指出这种后悔率在一般情况下是不可改进的。 |
| [^3] | [Designing Decision Support Systems Using Counterfactual Prediction Sets.](http://arxiv.org/abs/2306.03928) | 本文提出了一种基于反事实预测集的决策支持系统设计方法，不同于传统的单一标签预测，它使用符合预测器构建预测集，并引导人类专家从中选择标签值。 |

# 详细

[^1]: Rao-Blackwellising Bayesian Causal Inference

    Rao-Blackwellising Bayesian Causal Inference

    [https://arxiv.org/abs/2402.14781](https://arxiv.org/abs/2402.14781)

    本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。

    

    贝叶斯因果推断，即推断用于下游因果推理任务中的因果模型的后验概率，构成了一个在文献中鲜有探讨的难解的计算推断问题。本文将基于顺序的MCMC结构学习技术与最近梯度图学习的进展相结合，构建了一个有效的贝叶斯因果推断框架。具体而言，我们将推断因果结构的问题分解为(i)推断变量之间的拓扑顺序以及(ii)推断每个变量的父节点集合。当限制每个变量的父节点数量时，我们可以在多项式时间内完全边缘化父节点集合。我们进一步使用高斯过程来建模未知的因果机制，从而允许其精确边缘化。这引入了一个Rao-Blackwell化方案，其中除了因果顺序之外，模型中的所有组件都被消除。

    arXiv:2402.14781v1 Announce Type: cross  Abstract: Bayesian causal inference, i.e., inferring a posterior over causal models for the use in downstream causal reasoning tasks, poses a hard computational inference problem that is little explored in literature. In this work, we combine techniques from order-based MCMC structure learning with recent advances in gradient-based graph learning into an effective Bayesian causal inference framework. Specifically, we decompose the problem of inferring the causal structure into (i) inferring a topological order over variables and (ii) inferring the parent sets for each variable. When limiting the number of parents per variable, we can exactly marginalise over the parent sets in polynomial time. We further use Gaussian processes to model the unknown causal mechanisms, which also allows their exact marginalisation. This introduces a Rao-Blackwellization scheme, where all components are eliminated from the model, except for the causal order, for whi
    
[^2]: 在线局部虚发现率控制：一种资源分配方法

    Online Local False Discovery Rate Control: A Resource Allocation Approach

    [https://arxiv.org/abs/2402.11425](https://arxiv.org/abs/2402.11425)

    该研究提出了一种在线局部虚发现率控制的资源分配方法，实现了$O(\sqrt{T})$的后悔率，并指出这种后悔率在一般情况下是不可改进的。

    

    我们考虑在线局部虚发现率（FDR）控制问题，其中多个测试被顺序进行，目标是最大化总期望的发现次数。我们将问题形式化为一种在线资源分配问题，涉及接受/拒绝决策，从高层次来看，这可以被视为一个带有额外不确定性的在线背包问题，即随机预算补充。我们从一般的到达分布开始，并提出了一个简单的策略，实现了$O(\sqrt{T})$的后悔。我们通过展示这种后悔率在一般情况下是不可改进的来补充这一结果。然后我们将焦点转向离散到达分布。我们发现许多现有的在线资源分配文献中的重新解决启发式虽然在典型设置中实现了有界的损失，但可能会造成$\Omega(\sqrt{T})$甚至$\Omega(T)$的后悔。通过观察到典型策略往往太过

    arXiv:2402.11425v1 Announce Type: cross  Abstract: We consider the problem of online local false discovery rate (FDR) control where multiple tests are conducted sequentially, with the goal of maximizing the total expected number of discoveries. We formulate the problem as an online resource allocation problem with accept/reject decisions, which from a high level can be viewed as an online knapsack problem, with the additional uncertainty of random budget replenishment. We start with general arrival distributions and propose a simple policy that achieves a $O(\sqrt{T})$ regret. We complement the result by showing that such regret rate is in general not improvable. We then shift our focus to discrete arrival distributions. We find that many existing re-solving heuristics in the online resource allocation literature, albeit achieve bounded loss in canonical settings, may incur a $\Omega(\sqrt{T})$ or even a $\Omega(T)$ regret. With the observation that canonical policies tend to be too op
    
[^3]: 使用反事实预测集设计决策支持系统

    Designing Decision Support Systems Using Counterfactual Prediction Sets. (arXiv:2306.03928v1 [cs.LG])

    [http://arxiv.org/abs/2306.03928](http://arxiv.org/abs/2306.03928)

    本文提出了一种基于反事实预测集的决策支持系统设计方法，不同于传统的单一标签预测，它使用符合预测器构建预测集，并引导人类专家从中选择标签值。

    

    分类任务的决策支持系统通常被设计用于预测地面实况标签的值。然而，由于它们的预测并不完美，这些系统还需要让人类专家了解何时以及如何使用这些预测来更新自己的预测。不幸的是，这被证明是具有挑战性的。最近有人认为，另一种类型的决策支持系统可能会避开这个挑战。这些系统不是提供单个标签预测，而是使用符合预测器构建一组标签预测值，即预测集，并强制要求专家从预测集中预测一个标签值。然而，这些系统的设计和评估迄今仍依赖于样式化的专家模型，这引发了人们对它们的承诺的质疑。本文从在线学习的角度重新审视了这种系统的设计，并开发了一种不需要。

    Decision support systems for classification tasks are predominantly designed to predict the value of the ground truth labels. However, since their predictions are not perfect, these systems also need to make human experts understand when and how to use these predictions to update their own predictions. Unfortunately, this has been proven challenging. In this context, it has been recently argued that an alternative type of decision support systems may circumvent this challenge. Rather than providing a single label prediction, these systems provide a set of label prediction values constructed using a conformal predictor, namely a prediction set, and forcefully ask experts to predict a label value from the prediction set. However, the design and evaluation of these systems have so far relied on stylized expert models, questioning their promise. In this paper, we revisit the design of this type of systems from the perspective of online learning and develop a methodology that does not requi
    

