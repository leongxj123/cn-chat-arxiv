# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decentralized Sporadic Federated Learning: A Unified Methodology with Generalized Convergence Guarantees](https://arxiv.org/abs/2402.03448) | 本文提出了一种称为分散式间歇联邦学习（DSpodFL）的方法，它统一了分布式梯度下降（DGD）、随机闲话（RG）和分散式联邦平均（DFedAvg）等著名的分散优化方法。根据分析结果，DSpodFL能够在更一般的假设下达到几何收敛速率与最佳性差距的匹配。经过实验验证了该方法的有效性。 |

# 详细

[^1]: 分散式间歇联邦学习：具有广义收敛保证的统一方法

    Decentralized Sporadic Federated Learning: A Unified Methodology with Generalized Convergence Guarantees

    [https://arxiv.org/abs/2402.03448](https://arxiv.org/abs/2402.03448)

    本文提出了一种称为分散式间歇联邦学习（DSpodFL）的方法，它统一了分布式梯度下降（DGD）、随机闲话（RG）和分散式联邦平均（DFedAvg）等著名的分散优化方法。根据分析结果，DSpodFL能够在更一般的假设下达到几何收敛速率与最佳性差距的匹配。经过实验验证了该方法的有效性。

    

    分散式联邦学习（DFL）近来受到了重要的研究关注，涵盖了模型更新和模型聚合这两个关键联邦学习过程都由客户端进行的设置。在本文中，我们提出了分散式间歇联邦学习（DSpodFL），这是一种DFL方法，它在这两个过程中广义化了间歇性的概念，建模了在实际DFL设置中出现的不同形式的异质性的影响。DSpodFL将许多着名的分散优化方法，如分布式梯度下降（DGD），随机闲话（RG）和分散式联邦平均（DFedAvg），统一到一个建模框架下。我们对DSpodFL的收敛行为进行了分析，显示出可以在更一般的假设下，将几何收敛速率与有限的最佳性差距相匹配。通过实验证明：

    Decentralized Federated Learning (DFL) has received significant recent research attention, capturing settings where both model updates and model aggregations -- the two key FL processes -- are conducted by the clients. In this work, we propose Decentralized Sporadic Federated Learning ($\texttt{DSpodFL}$), a DFL methodology which generalizes the notion of sporadicity in both of these processes, modeling the impact of different forms of heterogeneity that manifest in realistic DFL settings. $\texttt{DSpodFL}$ unifies many of the prominent decentralized optimization methods, e.g., distributed gradient descent (DGD), randomized gossip (RG), and decentralized federated averaging (DFedAvg), under a single modeling framework. We analytically characterize the convergence behavior of $\texttt{DSpodFL}$, showing, among other insights, that we can match a geometric convergence rate to a finite optimality gap under more general assumptions than in existing works. Through experiments, we demonstra
    

