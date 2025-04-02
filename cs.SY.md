# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Reinforcement Learning for Radio Resource Allocation.](http://arxiv.org/abs/2203.03906) | 该论文介绍了一种利用图强化学习方法进行无线资源分配的方法，通过利用拓扑信息和排列特性，降低了深度强化学习的训练复杂性，并通过优化预测功率分配问题来验证方法的有效性。 |

# 详细

[^1]: 图强化学习用于无线资源分配

    Graph Reinforcement Learning for Radio Resource Allocation. (arXiv:2203.03906v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.03906](http://arxiv.org/abs/2203.03906)

    该论文介绍了一种利用图强化学习方法进行无线资源分配的方法，通过利用拓扑信息和排列特性，降低了深度强化学习的训练复杂性，并通过优化预测功率分配问题来验证方法的有效性。

    

    由于其处理无模型和端到端问题的能力，深度强化学习(DRL)在资源分配方面得到了广泛的研究。然而，DRL的高训练复杂性限制了它在动态无线系统中的实际应用。为了降低训练成本，我们采用图强化学习来利用无线通信中许多问题固有的两种关系先验：拓扑信息和排列特性。为了系统地设计图强化学习框架来利用这两个先验，我们首先构思了一种将状态矩阵转换为状态图的方法，然后提出了一种通用的图神经网络方法来满足理想的排列特性。为了展示如何应用所提出的方法，我们以深度确定性策略梯度(DDPG)为例，优化了两个代表性的资源分配问题。一个是预测功率分配，旨在最小化能耗。

    Deep reinforcement learning (DRL) for resource allocation has been investigated extensively owing to its ability of handling model-free and end-to-end problems. Yet the high training complexity of DRL hinders its practical use in dynamic wireless systems. To reduce the training cost, we resort to graph reinforcement learning for exploiting two kinds of relational priors inherent in many problems in wireless communications: topology information and permutation properties. To design graph reinforcement learning framework systematically for harnessing the two priors, we first conceive a method to transform state matrix into state graph, and then propose a general method for graph neural networks to satisfy desirable permutation properties. To demonstrate how to apply the proposed methods, we take deep deterministic policy gradient (DDPG) as an example for optimizing two representative resource allocation problems. One is predictive power allocation that minimizes the energy consumed for e
    

