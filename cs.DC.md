# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How to Collaborate: Towards Maximizing the Generalization Performance in Cross-Silo Federated Learning.](http://arxiv.org/abs/2401.13236) | 本文研究了异构数据联邦学习中的合作问题。通过推导出每个客户端的泛化界限，发现只有与拥有更多训练数据和相似数据分布的客户端合作，才能改善模型的泛化性能。根据这一分析，提出了基于层次聚类的合作训练方案。 |

# 详细

[^1]: 如何合作：朝着最大化异构数据联邦学习的泛化性能迈进

    How to Collaborate: Towards Maximizing the Generalization Performance in Cross-Silo Federated Learning. (arXiv:2401.13236v1 [cs.LG])

    [http://arxiv.org/abs/2401.13236](http://arxiv.org/abs/2401.13236)

    本文研究了异构数据联邦学习中的合作问题。通过推导出每个客户端的泛化界限，发现只有与拥有更多训练数据和相似数据分布的客户端合作，才能改善模型的泛化性能。根据这一分析，提出了基于层次聚类的合作训练方案。

    

    联邦学习（FL）作为一种保护隐私的分布式学习框架，吸引了广泛的关注。本文关注交叉数据源的FL，其中客户端在训练后成为模型所有者，并且只关心模型在本地数据上的泛化性能。由于数据异质性问题，要求所有客户端参加单一的FL训练过程可能会导致模型性能下降。为了调查合作的有效性，我们首先推导了每个客户端在与其他客户端合作或独立训练时的泛化界限。我们展示了仅通过与具有更多训练数据和相似数据分布的其他客户端合作，可以改善客户端的泛化性能。我们的分析使我们能够通过将客户端分成多个合作组来制定客户端效用最大化问题。然后提出了一种基于层次聚类的合作训练（HCCT）方案。

    Federated learning (FL) has attracted vivid attention as a privacy-preserving distributed learning framework. In this work, we focus on cross-silo FL, where clients become the model owners after training and are only concerned about the model's generalization performance on their local data. Due to the data heterogeneity issue, asking all the clients to join a single FL training process may result in model performance degradation. To investigate the effectiveness of collaboration, we first derive a generalization bound for each client when collaborating with others or when training independently. We show that the generalization performance of a client can be improved only by collaborating with other clients that have more training data and similar data distribution. Our analysis allows us to formulate a client utility maximization problem by partitioning clients into multiple collaborating groups. A hierarchical clustering-based collaborative training (HCCT) scheme is then proposed, wh
    

