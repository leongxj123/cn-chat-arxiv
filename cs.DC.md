# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Decentralized Alternating Gradient Method for Communication-Efficient Bilevel Programming.](http://arxiv.org/abs/2211.04088) | 本文提出了一种通信高效的分散式交替梯度法解决双层规划问题，相较于其他方法，该算法具有更低的通信成本和更高的隐私性。 |

# 详细

[^1]: 一种通信高效的分散式交替梯度法用于双层规划

    A Decentralized Alternating Gradient Method for Communication-Efficient Bilevel Programming. (arXiv:2211.04088v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.04088](http://arxiv.org/abs/2211.04088)

    本文提出了一种通信高效的分散式交替梯度法解决双层规划问题，相较于其他方法，该算法具有更低的通信成本和更高的隐私性。

    

    双层规划近期引起了学术界的广泛关注，因为它有许多应用，包括强化学习和超参数优化。然而，现有的解决方案通常采用单机或联邦学习的方式，并存在通信成本高和隐私泄露风险等问题。本文提出了一种基于惩罚函数的分散式算法来解决这类优化问题，改进了现有的方法并且在理论上得到了保证。

    Bilevel programming has recently received attention in the literature, due to a wide range of applications, including reinforcement learning and hyper-parameter optimization. However, it is widely assumed that the underlying bilevel optimization problem is solved either by a single machine or in the case of multiple machines connected in a star-shaped network, i.e., federated learning setting. The latter approach suffers from a high communication cost on the central node (e.g., parameter server) and exhibits privacy vulnerabilities. Hence, it is of interest to develop methods that solve bilevel optimization problems in a communication-efficient decentralized manner. To that end, this paper introduces a penalty function based decentralized algorithm with theoretical guarantees for this class of optimization problems. Specifically, a distributed alternating gradient-type algorithm for solving consensus bilevel programming over a decentralized network is developed. A key feature of the pr
    

