# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedADMM-InSa: An Inexact and Self-Adaptive ADMM for Federated Learning](https://arxiv.org/abs/2402.13989) | 提出了一种不精确和自适应的FedADMM算法，通过为客户端的本地更新设计一个不精确性标准，消除了调整本地训练准确度的需要，降低了计算成本并减轻了滞后效应。 |

# 详细

[^1]: FedADMM-InSa: 一种不精确和自适应的联邦学习ADMM

    FedADMM-InSa: An Inexact and Self-Adaptive ADMM for Federated Learning

    [https://arxiv.org/abs/2402.13989](https://arxiv.org/abs/2402.13989)

    提出了一种不精确和自适应的FedADMM算法，通过为客户端的本地更新设计一个不精确性标准，消除了调整本地训练准确度的需要，降低了计算成本并减轻了滞后效应。

    

    联邦学习(FL)是一个有希望的框架，可以从分布式数据中学习同时保持隐私。有效的FL算法的发展面临各种挑战，包括异构数据和系统、通信能力有限以及受限的本地计算资源。最近开发的FedADMM方法对数据和系统的异构性表现出很强的韧性。然而，如果超参数没有经过精心调整，它们仍然会遭受性能下降的问题。为了解决这个问题，我们提出了一种不精确和自适应的FedADMM算法，名为FedADMM-InSa。首先，我们为客户端的本地更新设计了一个不精确性标准，以消除必须根据经验设置本地训练准确性的需求。这种不精确性标准可以由每个客户端独立地根据其独特条件进行评估，从而降低本地计算成本并减轻不良的滞后效应。

    arXiv:2402.13989v1 Announce Type: new  Abstract: Federated learning (FL) is a promising framework for learning from distributed data while maintaining privacy. The development of efficient FL algorithms encounters various challenges, including heterogeneous data and systems, limited communication capacities, and constrained local computational resources. Recently developed FedADMM methods show great resilience to both data and system heterogeneity. However, they still suffer from performance deterioration if the hyperparameters are not carefully tuned. To address this issue, we propose an inexact and self-adaptive FedADMM algorithm, termed FedADMM-InSa. First, we design an inexactness criterion for the clients' local updates to eliminate the need for empirically setting the local training accuracy. This inexactness criterion can be assessed by each client independently based on its unique condition, thereby reducing the local computational cost and mitigating the undesirable straggle e
    

