# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Global Optimality in Cooperative MARL with the Transformation And Distillation Framework.](http://arxiv.org/abs/2207.11143) | 本文研究了采用分散策略的MARL算法在梯度下降优化器下的次最优性，并提出了转化与蒸馏框架，该框架可以将多智能体MDP转化为单智能体MDP以实现分散执行。 |

# 详细

[^1]: 《采用转化与蒸馏框架实现合作MARL全局最优性》

    Towards Global Optimality in Cooperative MARL with the Transformation And Distillation Framework. (arXiv:2207.11143v3 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2207.11143](http://arxiv.org/abs/2207.11143)

    本文研究了采用分散策略的MARL算法在梯度下降优化器下的次最优性，并提出了转化与蒸馏框架，该框架可以将多智能体MDP转化为单智能体MDP以实现分散执行。

    

    在合作多智能体强化学习中，分散执行是一项核心需求。目前，大多数流行的MARL算法采用分散策略来实现分散执行，并使用梯度下降作为优化器。然而，在考虑到优化方法的情况下，这些算法几乎没有任何理论分析，我们发现当梯度下降被选为优化方法时，各种流行的分散策略MARL算法在玩具任务中都是次最优的。本文在理论上分析了两种常见的采用分散策略的算法——多智能体策略梯度方法和值分解方法，证明了它们在使用梯度下降时的次最优性。此外，我们提出了转化与蒸馏（TAD）框架，它将多智能体MDP重新制定为一种具有连续结构的特殊单智能体MDP，并通过蒸馏实现分散执行。

    Decentralized execution is one core demand in cooperative multi-agent reinforcement learning (MARL). Recently, most popular MARL algorithms have adopted decentralized policies to enable decentralized execution and use gradient descent as their optimizer. However, there is hardly any theoretical analysis of these algorithms taking the optimization method into consideration, and we find that various popular MARL algorithms with decentralized policies are suboptimal in toy tasks when gradient descent is chosen as their optimization method. In this paper, we theoretically analyze two common classes of algorithms with decentralized policies -- multi-agent policy gradient methods and value-decomposition methods to prove their suboptimality when gradient descent is used. In addition, we propose the Transformation And Distillation (TAD) framework, which reformulates a multi-agent MDP as a special single-agent MDP with a sequential structure and enables decentralized execution by distilling the
    

