# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VFedMH: Vertical Federated Learning for Training Multi-party Heterogeneous Models.](http://arxiv.org/abs/2310.13367) | VFedMH是一种垂直联合学习方法，通过在前向传播过程中聚合参与者的嵌入来处理参与者之间的异构模型，解决了现有VFL方法面临的挑战。 |

# 详细

[^1]: VFedMH: 垂直联合学习用于训练多参与方异构模型

    VFedMH: Vertical Federated Learning for Training Multi-party Heterogeneous Models. (arXiv:2310.13367v1 [cs.LG])

    [http://arxiv.org/abs/2310.13367](http://arxiv.org/abs/2310.13367)

    VFedMH是一种垂直联合学习方法，通过在前向传播过程中聚合参与者的嵌入来处理参与者之间的异构模型，解决了现有VFL方法面临的挑战。

    

    垂直联合学习（VFL）作为一种集成样本对齐和特征合并的新型训练范式，已经引起了越来越多的关注。然而，现有的VFL方法在处理参与者之间存在异构本地模型时面临挑战，这影响了优化收敛性和泛化能力。为了解决这个问题，本文提出了一种名为VFedMH的新方法，用于训练多方异构模型。VFedMH的重点是在前向传播期间聚合每个参与者知识的嵌入，而不是中间结果。主动方，拥有样本的标签和特征，在VFedMH中安全地聚合本地嵌入以获得全局知识嵌入，并将其发送给被动方。被动方仅拥有样本的特征，然后利用全局嵌入在其本地异构网络上进行前向传播。然而，被动方不拥有标签。

    Vertical Federated Learning (VFL) has gained increasing attention as a novel training paradigm that integrates sample alignment and feature union. However, existing VFL methods face challenges when dealing with heterogeneous local models among participants, which affects optimization convergence and generalization. To address this issue, this paper proposes a novel approach called Vertical Federated learning for training Multi-parties Heterogeneous models (VFedMH). VFedMH focuses on aggregating the embeddings of each participant's knowledge instead of intermediate results during forward propagation. The active party, who possesses labels and features of the sample, in VFedMH securely aggregates local embeddings to obtain global knowledge embeddings, and sends them to passive parties. The passive parties, who own only features of the sample, then utilize the global embeddings to propagate forward on their local heterogeneous networks. However, the passive party does not own the labels, 
    

