# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provable Mutual Benefits from Federated Learning in Privacy-Sensitive Domains](https://arxiv.org/abs/2403.06672) | 本文研究了在隐私敏感领域中如何设计一种FL协议，既能保证隐私，又能提高模型准确性，并提供了设计出对所有参与者都有益处的协议。 |
| [^2] | [Stochastic contextual bandits with graph feedback: from independence number to MAS number](https://arxiv.org/abs/2402.18591) | 本文研究了具有图反馈的上下文赌博问题，提出了一个刻画学习极限的图论量 $\beta_M(G)$，并建立了对应的遗憾下限。 |

# 详细

[^1]: 在隐私敏感领域中从联邦学习中有可证明的互惠益处

    Provable Mutual Benefits from Federated Learning in Privacy-Sensitive Domains

    [https://arxiv.org/abs/2403.06672](https://arxiv.org/abs/2403.06672)

    本文研究了在隐私敏感领域中如何设计一种FL协议，既能保证隐私，又能提高模型准确性，并提供了设计出对所有参与者都有益处的协议。

    

    跨领域联邦学习（FL）允许数据所有者通过从彼此的私有数据集中获益来训练准确的机器学习模型。本文研究了在何时以及如何服务器可以设计一种对所有参与者都有利的FL协议的问题。我们提供了在均值估计和凸随机优化背景下存在相互有利协议的必要和充分条件。我们推导出了在对称隐私偏好下，最大化总客户效用的协议。最后，我们设计了最大化最终模型准确性的协议，并在合成实验中展示了它们的好处。

    arXiv:2403.06672v1 Announce Type: cross  Abstract: Cross-silo federated learning (FL) allows data owners to train accurate machine learning models by benefiting from each others private datasets. Unfortunately, the model accuracy benefits of collaboration are often undermined by privacy defenses. Therefore, to incentivize client participation in privacy-sensitive domains, a FL protocol should strike a delicate balance between privacy guarantees and end-model accuracy. In this paper, we study the question of when and how a server could design a FL protocol provably beneficial for all participants. First, we provide necessary and sufficient conditions for the existence of mutually beneficial protocols in the context of mean estimation and convex stochastic optimization. We also derive protocols that maximize the total clients' utility, given symmetric privacy preferences. Finally, we design protocols maximizing end-model accuracy and demonstrate their benefits in synthetic experiments.
    
[^2]: 具有图反馈的随机上下文赌博：从独立数到MAS数

    Stochastic contextual bandits with graph feedback: from independence number to MAS number

    [https://arxiv.org/abs/2402.18591](https://arxiv.org/abs/2402.18591)

    本文研究了具有图反馈的上下文赌博问题，提出了一个刻画学习极限的图论量 $\beta_M(G)$，并建立了对应的遗憾下限。

    

    我们考虑具有图反馈的上下文赌博，在这类互动学习问题中，具有比普通上下文赌博更丰富结构，其中采取一个行动将在所有情境下揭示所有相邻行动的奖励。与多臂赌博设置不同，多文献已经对图反馈的理解进行了全面探讨，但在上下文赌博对应部分仍有许多未被探讨的地方。在本文中，我们通过建立一个遗憾下限 $\Omega(\sqrt{\beta_M(G) T})$ 探究了这个问题，其中 $M$ 是情境数，$G$ 是反馈图，$\beta_M(G)$ 是我们提出的表征该问题类的基础学习限制的图论量。有趣的是，$\beta_M(G)$ 在 $\alpha(G)$ (图的独立数) 和 $\mathsf{m}(G)$ (图的最大无环子图（MAS）数) 之间插值。

    arXiv:2402.18591v1 Announce Type: new  Abstract: We consider contextual bandits with graph feedback, a class of interactive learning problems with richer structures than vanilla contextual bandits, where taking an action reveals the rewards for all neighboring actions in the feedback graph under all contexts. Unlike the multi-armed bandits setting where a growing literature has painted a near-complete understanding of graph feedback, much remains unexplored in the contextual bandits counterpart. In this paper, we make inroads into this inquiry by establishing a regret lower bound $\Omega(\sqrt{\beta_M(G) T})$, where $M$ is the number of contexts, $G$ is the feedback graph, and $\beta_M(G)$ is our proposed graph-theoretical quantity that characterizes the fundamental learning limit for this class of problems. Interestingly, $\beta_M(G)$ interpolates between $\alpha(G)$ (the independence number of the graph) and $\mathsf{m}(G)$ (the maximum acyclic subgraph (MAS) number of the graph) as 
    

