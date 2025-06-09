# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment.](http://arxiv.org/abs/2302.09913) | 本文提出了一种基于编码计算和向量承诺的拜占庭抵抗安全聚合方案，用于联邦学习。该方案通过RAM秘密共享将本地更新分割成较小子向量，并使用双重RAMP共享技术实现成对距离的安全计算。 |

# 详细

[^1]: 基于编码计算和向量承诺的拜占庭抵抗安全聚合方案，用于联邦学习 (arXiv:2302.09913v2 [cs.CR] UPDATED)

    ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment. (arXiv:2302.09913v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2302.09913](http://arxiv.org/abs/2302.09913)

    本文提出了一种基于编码计算和向量承诺的拜占庭抵抗安全聚合方案，用于联邦学习。该方案通过RAM秘密共享将本地更新分割成较小子向量，并使用双重RAMP共享技术实现成对距离的安全计算。

    

    本文提出了一种高效的联邦学习保护方案，可以抵御拜占庭攻击和隐私泄露。这种方案通过处理单个更新来管理对抗行为，并在抵御串通节点的同时保护数据隐私。然而，用于对更新向量进行安全秘密共享的通信负载可能非常高。为了解决这个问题，本文提出了一种将本地更新分割成较小子向量并使用RAM秘密共享的方案。但是，这种共享方法无法进行双线性计算，例如需要异常检测算法的成对距离计算。为了克服这个问题，每个用户都会运行另一轮RAMP共享，该共享具有不同的数据嵌入其中。这种受编码计算思想启发的技术实现了成对距离的安全计算。

    In this paper, we propose an efficient secure aggregation scheme for federated learning that is protected against Byzantine attacks and privacy leakages. Processing individual updates to manage adversarial behavior, while preserving privacy of data against colluding nodes, requires some sort of secure secret sharing. However, communication load for secret sharing of long vectors of updates can be very high. To resolve this issue, in the proposed scheme, local updates are partitioned into smaller sub-vectors and shared using ramp secret sharing. However, this sharing method does not admit bi-linear computations, such as pairwise distance calculations, needed by outlier-detection algorithms. To overcome this issue, each user runs another round of ramp sharing, with different embedding of data in the sharing polynomial. This technique, motivated by ideas from coded computing, enables secure computation of pairwise distance. In addition, to maintain the integrity and privacy of the local u
    

