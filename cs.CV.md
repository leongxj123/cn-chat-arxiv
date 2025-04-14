# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Class-Incremental Learning with Prompting.](http://arxiv.org/abs/2310.08948) | 本文提出了一种名为FCILPT的方法，用于解决联邦增量学习中的灾难性遗忘问题，该方法能够处理非独立和同分布数据分布情况，并保护数据隐私。 |

# 详细

[^1]: 具有提示的联邦增量学习

    Federated Class-Incremental Learning with Prompting. (arXiv:2310.08948v1 [cs.CV])

    [http://arxiv.org/abs/2310.08948](http://arxiv.org/abs/2310.08948)

    本文提出了一种名为FCILPT的方法，用于解决联邦增量学习中的灾难性遗忘问题，该方法能够处理非独立和同分布数据分布情况，并保护数据隐私。

    

    随着Web技术的发展，使用存储在不同客户端上的数据变得越来越常见。同时，由于在让模型从分布在各个客户端上的数据中学习时能够保护数据隐私，联邦学习引起了广泛关注。然而，大多数现有的工作都假设客户端的数据是固定的。在现实场景中，这种假设很可能不成立，因为数据可能不断生成，新的类别也可能出现。因此，我们专注于实际且具有挑战性的联邦增量学习（FCIL）问题。对于FCIL，由于新类别的出现和客户端数据分布的非独立和同分布性质（non-iid），局部和全局模型可能会对旧类别发生灾难性遗忘。在本文中，我们提出了一种新颖的方法，称为具有提示的联邦增量学习（FCILPT）。

    As Web technology continues to develop, it has become increasingly common to use data stored on different clients. At the same time, federated learning has received widespread attention due to its ability to protect data privacy when let models learn from data which is distributed across various clients. However, most existing works assume that the client's data are fixed. In real-world scenarios, such an assumption is most likely not true as data may be continuously generated and new classes may also appear. To this end, we focus on the practical and challenging federated class-incremental learning (FCIL) problem. For FCIL, the local and global models may suffer from catastrophic forgetting on old classes caused by the arrival of new classes and the data distributions of clients are non-independent and identically distributed (non-iid).  In this paper, we propose a novel method called Federated Class-Incremental Learning with PrompTing (FCILPT). Given the privacy and limited memory, F
    

