# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rao-Blackwellising Bayesian Causal Inference](https://arxiv.org/abs/2402.14781) | 本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。 |

# 详细

[^1]: Rao-Blackwellising Bayesian Causal Inference

    Rao-Blackwellising Bayesian Causal Inference

    [https://arxiv.org/abs/2402.14781](https://arxiv.org/abs/2402.14781)

    本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。

    

    贝叶斯因果推断，即推断用于下游因果推理任务中的因果模型的后验概率，构成了一个在文献中鲜有探讨的难解的计算推断问题。本文将基于顺序的MCMC结构学习技术与最近梯度图学习的进展相结合，构建了一个有效的贝叶斯因果推断框架。具体而言，我们将推断因果结构的问题分解为(i)推断变量之间的拓扑顺序以及(ii)推断每个变量的父节点集合。当限制每个变量的父节点数量时，我们可以在多项式时间内完全边缘化父节点集合。我们进一步使用高斯过程来建模未知的因果机制，从而允许其精确边缘化。这引入了一个Rao-Blackwell化方案，其中除了因果顺序之外，模型中的所有组件都被消除。

    arXiv:2402.14781v1 Announce Type: cross  Abstract: Bayesian causal inference, i.e., inferring a posterior over causal models for the use in downstream causal reasoning tasks, poses a hard computational inference problem that is little explored in literature. In this work, we combine techniques from order-based MCMC structure learning with recent advances in gradient-based graph learning into an effective Bayesian causal inference framework. Specifically, we decompose the problem of inferring the causal structure into (i) inferring a topological order over variables and (ii) inferring the parent sets for each variable. When limiting the number of parents per variable, we can exactly marginalise over the parent sets in polynomial time. We further use Gaussian processes to model the unknown causal mechanisms, which also allows their exact marginalisation. This introduces a Rao-Blackwellization scheme, where all components are eliminated from the model, except for the causal order, for whi
    

