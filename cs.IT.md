# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Generalization via Information-Theoretic Distribution Diversification.](http://arxiv.org/abs/2310.07171) | 该论文研究了联邦学习中泛化能力的挑战，特别关注训练分布和测试分布的不匹配。提出了一种信息论的泛化方法来解决这个问题。 |

# 详细

[^1]: 通过信息论分布多样化实现联邦泛化能力

    Federated Generalization via Information-Theoretic Distribution Diversification. (arXiv:2310.07171v1 [cs.LG])

    [http://arxiv.org/abs/2310.07171](http://arxiv.org/abs/2310.07171)

    该论文研究了联邦学习中泛化能力的挑战，特别关注训练分布和测试分布的不匹配。提出了一种信息论的泛化方法来解决这个问题。

    

    联邦学习（FL）因其在无需直接数据共享的情况下进行协同模型训练的能力而日益突出。然而，客户端之间本地数据分布的巨大差异，通常被称为非独立同分布（non-IID）挑战，对FL的泛化能力构成了重大障碍。当并非所有客户端都参与训练过程时，情况变得更加复杂，这是由于不稳定的网络连接或有限的计算能力而常见。这可能极大地复杂化了对训练模型的泛化能力的评估。尽管最近的大量研究集中在涉及具有不同分布的参与客户端的未见数据的泛化差距问题上，但参与客户端的训练分布和非参与客户端的测试分布之间的差异却被大部分忽视了。为此，我们的论文揭示了一种基于信息论的泛化方法。

    Federated Learning (FL) has surged in prominence due to its capability of collaborative model training without direct data sharing. However, the vast disparity in local data distributions among clients, often termed the non-Independent Identically Distributed (non-IID) challenge, poses a significant hurdle to FL's generalization efficacy. The scenario becomes even more complex when not all clients participate in the training process, a common occurrence due to unstable network connections or limited computational capacities. This can greatly complicate the assessment of the trained models' generalization abilities. While a plethora of recent studies has centered on the generalization gap pertaining to unseen data from participating clients with diverse distributions, the divergence between the training distributions of participating clients and the testing distributions of non-participating ones has been largely overlooked. In response, our paper unveils an information-theoretic genera
    

