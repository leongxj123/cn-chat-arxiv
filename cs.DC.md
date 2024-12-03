# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decoupled Vertical Federated Learning for Practical Training on Vertically Partitioned Data](https://arxiv.org/abs/2403.03871) | 提出了Decoupled VFL（DVFL），一种面向VFL的分段学习方法，实现了分散聚合和隔离，从而提高了容错性。 |
| [^2] | [Topology-Based Reconstruction Prevention for Decentralised Learning](https://arxiv.org/abs/2312.05248) | 通过研究发现，在去中心化学习中，被动的好奇敌手可以在几次保护隐私的求和操作后推断出其他用户的私人数据。 |

# 详细

[^1]: 面向垂直分区数据的解耦式垂直联邦学习，用于实际训练

    Decoupled Vertical Federated Learning for Practical Training on Vertically Partitioned Data

    [https://arxiv.org/abs/2403.03871](https://arxiv.org/abs/2403.03871)

    提出了Decoupled VFL（DVFL），一种面向VFL的分段学习方法，实现了分散聚合和隔离，从而提高了容错性。

    

    垂直联邦学习（VFL）是一种新兴的分布式机器学习范式，其中共同实体的不同特征所有者合作学习全局模型而无需共享数据。在VFL中，主机客户端拥有每个实体的数据标签，并基于所有客户端的中间本地表示学习最终表示。因此，主机是一个单点故障，标签反馈可以被恶意客户端用来推断私有特征。要求所有参与者在整个训练过程中保持活跃和值得信赖通常是不切实际的，在受控环境之外完全不可行。我们提出了一种面向VFL的分段学习方法Decoupled VFL（DVFL）。通过在各自的目标上训练每个模型，DVFL允许特征学习和标签监督之间的分散聚合和隔离。具有这些属性，DVFL具有容错性。

    arXiv:2403.03871v1 Announce Type: new  Abstract: Vertical Federated Learning (VFL) is an emergent distributed machine learning paradigm wherein owners of disjoint features of a common set of entities collaborate to learn a global model without sharing data. In VFL, a host client owns data labels for each entity and learns a final representation based on intermediate local representations from all guest clients. Therefore, the host is a single point of failure and label feedback can be used by malicious guest clients to infer private features. Requiring all participants to remain active and trustworthy throughout the entire training process is generally impractical and altogether infeasible outside of controlled environments. We propose Decoupled VFL (DVFL), a blockwise learning approach to VFL. By training each model on its own objective, DVFL allows for decentralized aggregation and isolation between feature learning and label supervision. With these properties, DVFL is fault tolerant
    
[^2]: 基于拓扑的去重建防护在去中心化学习中的应用

    Topology-Based Reconstruction Prevention for Decentralised Learning

    [https://arxiv.org/abs/2312.05248](https://arxiv.org/abs/2312.05248)

    通过研究发现，在去中心化学习中，被动的好奇敌手可以在几次保护隐私的求和操作后推断出其他用户的私人数据。

    

    最近，去中心化学习作为一种替代联邦学习的方式，获得了人们的关注，其中数据和协调都分布在用户之间。为了保护数据的机密性，去中心化学习依赖于差分隐私、多方计算，或者二者的结合。然而，连续运行多个保护隐私的求和操作可能会使对手进行重建攻击。不幸的是，当前的重建对策要么无法简单地适应分布式环境，要么会添加过多的噪音。在这项工作中，我们首先表明，被动的好奇敌手可以在几次保护隐私的求和之后推断出其他用户的私人数据。例如，在拓扑中有18个用户的子图中，我们发现只有三个被动的好奇敌手成功重建私人数据的概率为11.0%，平均每个对手需要8.8次求和。

    arXiv:2312.05248v2 Announce Type: replace-cross  Abstract: Decentralised learning has recently gained traction as an alternative to federated learning in which both data and coordination are distributed over its users. To preserve data confidentiality, decentralised learning relies on differential privacy, multi-party computation, or a combination thereof. However, running multiple privacy-preserving summations in sequence may allow adversaries to perform reconstruction attacks. Unfortunately, current reconstruction countermeasures either cannot trivially be adapted to the distributed setting, or add excessive amounts of noise.   In this work, we first show that passive honest-but-curious adversaries can infer other users' private data after several privacy-preserving summations. For example, in subgraphs with 18 users, we show that only three passive honest-but-curious adversaries succeed at reconstructing private data 11.0% of the time, requiring an average of 8.8 summations per adve
    

