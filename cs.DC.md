# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TurboSVM-FL: Boosting Federated Learning through SVM Aggregation for Lazy Clients.](http://arxiv.org/abs/2401.12012) | TurboSVM-FL是一种新颖的联邦聚合策略，通过SVM聚合为懒惰客户端增强联邦学习。这种策略在不增加客户端计算负担的情况下解决了联邦学习中的收敛速度慢的问题。 |

# 详细

[^1]: TurboSVM-FL: 通过SVM聚合为懒惰客户端增强联邦学习

    TurboSVM-FL: Boosting Federated Learning through SVM Aggregation for Lazy Clients. (arXiv:2401.12012v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.12012](http://arxiv.org/abs/2401.12012)

    TurboSVM-FL是一种新颖的联邦聚合策略，通过SVM聚合为懒惰客户端增强联邦学习。这种策略在不增加客户端计算负担的情况下解决了联邦学习中的收敛速度慢的问题。

    

    联邦学习是一种分布式协作机器学习范例，在近年来获得了强烈的推动力。在联邦学习中，中央服务器定期通过客户端协调模型，并聚合由客户端在本地训练的模型，而无需访问本地数据。尽管具有潜力，但联邦学习的实施仍然面临一些挑战，主要是由于数据异质性导致的收敛速度慢。收敛速度慢在跨设备联邦学习场景中尤为问题，其中客户端可能受到计算能力和存储空间的严重限制，因此对客户端产生额外计算或内存负担的方法，如辅助目标项和更大的训练迭代次数，可能不实际。在本文中，我们提出了一种新颖的联邦聚合策略TurboSVM-FL，它不会给客户端增加额外的计算负担

    Federated learning is a distributed collaborative machine learning paradigm that has gained strong momentum in recent years. In federated learning, a central server periodically coordinates models with clients and aggregates the models trained locally by clients without necessitating access to local data. Despite its potential, the implementation of federated learning continues to encounter several challenges, predominantly the slow convergence that is largely due to data heterogeneity. The slow convergence becomes particularly problematic in cross-device federated learning scenarios where clients may be strongly limited by computing power and storage space, and hence counteracting methods that induce additional computation or memory cost on the client side such as auxiliary objective terms and larger training iterations can be impractical. In this paper, we propose a novel federated aggregation strategy, TurboSVM-FL, that poses no additional computation burden on the client side and c
    

