# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Semi-Supervised Federated Learning for Heterogeneous Participants.](http://arxiv.org/abs/2307.15870) | 本论文提出了一种高效的半监督异构参与者联邦学习系统，通过引入聚类正则化来改进模型在数据非独立同分布情况下的性能，并对模型收敛性进行了理论和实验研究。 |

# 详细

[^1]: 高效的半监督异构参与者联邦学习

    Efficient Semi-Supervised Federated Learning for Heterogeneous Participants. (arXiv:2307.15870v1 [cs.LG])

    [http://arxiv.org/abs/2307.15870](http://arxiv.org/abs/2307.15870)

    本论文提出了一种高效的半监督异构参与者联邦学习系统，通过引入聚类正则化来改进模型在数据非独立同分布情况下的性能，并对模型收敛性进行了理论和实验研究。

    

    联邦学习（FL）允许多个客户端在私有数据上协同训练机器学习模型，但在资源有限的环境中训练和部署大型模型用于广泛应用是具有挑战性的。幸运的是，分离式联邦学习（SFL）通过减轻客户端的计算和通信负担提供了优秀的解决方案。SFL通常假设客户端具有标记的数据进行本地训练，然而在实践中并非总是如此。以前的研究采用半监督技术来利用FL中的无标记数据，但数据的非独立同分布性提出了确保训练效率的另一个挑战。在这里，我们提出了一种新颖的系统Pseudo-Clustering Semi-SFL，用于在标记数据位于服务器上的情境下训练模型。通过引入聚类正则化，可以提高数据非独立同分布情况下的模型性能。此外，我们对模型收敛性进行了理论和实验研究，发现了...

    Federated Learning (FL) has emerged to allow multiple clients to collaboratively train machine learning models on their private data. However, training and deploying large models for broader applications is challenging in resource-constrained environments. Fortunately, Split Federated Learning (SFL) offers an excellent solution by alleviating the computation and communication burden on the clients SFL often assumes labeled data for local training on clients, however, it is not the case in practice.Prior works have adopted semi-supervised techniques for leveraging unlabeled data in FL, but data non-IIDness poses another challenge to ensure training efficiency. Herein, we propose Pseudo-Clustering Semi-SFL, a novel system for training models in scenarios where labeled data reside on the server. By introducing Clustering Regularization, model performance under data non-IIDness can be improved. Besides, our theoretical and experimental investigations into model convergence reveal that the 
    

