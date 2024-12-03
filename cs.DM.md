# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Topology-Based Reconstruction Prevention for Decentralised Learning](https://arxiv.org/abs/2312.05248) | 通过研究发现，在去中心化学习中，被动的好奇敌手可以在几次保护隐私的求和操作后推断出其他用户的私人数据。 |

# 详细

[^1]: 基于拓扑的去重建防护在去中心化学习中的应用

    Topology-Based Reconstruction Prevention for Decentralised Learning

    [https://arxiv.org/abs/2312.05248](https://arxiv.org/abs/2312.05248)

    通过研究发现，在去中心化学习中，被动的好奇敌手可以在几次保护隐私的求和操作后推断出其他用户的私人数据。

    

    最近，去中心化学习作为一种替代联邦学习的方式，获得了人们的关注，其中数据和协调都分布在用户之间。为了保护数据的机密性，去中心化学习依赖于差分隐私、多方计算，或者二者的结合。然而，连续运行多个保护隐私的求和操作可能会使对手进行重建攻击。不幸的是，当前的重建对策要么无法简单地适应分布式环境，要么会添加过多的噪音。在这项工作中，我们首先表明，被动的好奇敌手可以在几次保护隐私的求和之后推断出其他用户的私人数据。例如，在拓扑中有18个用户的子图中，我们发现只有三个被动的好奇敌手成功重建私人数据的概率为11.0%，平均每个对手需要8.8次求和。

    arXiv:2312.05248v2 Announce Type: replace-cross  Abstract: Decentralised learning has recently gained traction as an alternative to federated learning in which both data and coordination are distributed over its users. To preserve data confidentiality, decentralised learning relies on differential privacy, multi-party computation, or a combination thereof. However, running multiple privacy-preserving summations in sequence may allow adversaries to perform reconstruction attacks. Unfortunately, current reconstruction countermeasures either cannot trivially be adapted to the distributed setting, or add excessive amounts of noise.   In this work, we first show that passive honest-but-curious adversaries can infer other users' private data after several privacy-preserving summations. For example, in subgraphs with 18 users, we show that only three passive honest-but-curious adversaries succeed at reconstructing private data 11.0% of the time, requiring an average of 8.8 summations per adve
    

