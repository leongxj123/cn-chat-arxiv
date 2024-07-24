# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rendering Wireless Environments Useful for Gradient Estimators: A Zero-Order Stochastic Federated Learning Method](https://arxiv.org/abs/2401.17460) | 提出了一种新颖的零阶随机联邦学习方法，通过利用无线通信通道的特性，在学习算法中考虑了无线通道，避免了资源的浪费和分析难度。 |
| [^2] | [FedPop: Federated Population-based Hyperparameter Tuning.](http://arxiv.org/abs/2308.08634) | FedPop是一种用于解决联邦学习中超参数调优问题的新算法，它采用基于人口的进化算法来优化客户端和服务器上的超参数。 |
| [^3] | [Efficient Semi-Supervised Federated Learning for Heterogeneous Participants.](http://arxiv.org/abs/2307.15870) | 本论文提出了一种高效的半监督异构参与者联邦学习系统，通过引入聚类正则化来改进模型在数据非独立同分布情况下的性能，并对模型收敛性进行了理论和实验研究。 |

# 详细

[^1]: 使无线环境对梯度估计器有用：一种零阶随机联邦学习方法

    Rendering Wireless Environments Useful for Gradient Estimators: A Zero-Order Stochastic Federated Learning Method

    [https://arxiv.org/abs/2401.17460](https://arxiv.org/abs/2401.17460)

    提出了一种新颖的零阶随机联邦学习方法，通过利用无线通信通道的特性，在学习算法中考虑了无线通道，避免了资源的浪费和分析难度。

    

    联邦学习（FL）是一种新颖的机器学习方法，允许多个边缘设备协同训练模型，而无需公开原始数据。然而，当设备和服务器通过无线信道通信时，该方法面临着通信和计算瓶颈。通过利用一个通信高效的框架，我们提出了一种新颖的零阶（ZO）方法，采用一点梯度估计器，利用无线通信通道的特性，而无需知道通道状态系数。这是第一种将无线通道包含在学习算法本身中的方法，而不是浪费资源来分析和消除其影响。这项工作的两个主要困难是，在FL中，目标函数通常不是凸的，这使得将FL扩展到ZO方法具有挑战性，以及包括影响的难度。

    Federated learning (FL) is a novel approach to machine learning that allows multiple edge devices to collaboratively train a model without disclosing their raw data. However, several challenges hinder the practical implementation of this approach, especially when devices and the server communicate over wireless channels, as it suffers from communication and computation bottlenecks in this case. By utilizing a communication-efficient framework, we propose a novel zero-order (ZO) method with a one-point gradient estimator that harnesses the nature of the wireless communication channel without requiring the knowledge of the channel state coefficient. It is the first method that includes the wireless channel in the learning algorithm itself instead of wasting resources to analyze it and remove its impact. The two main difficulties of this work are that in FL, the objective function is usually not convex, which makes the extension of FL to ZO methods challenging, and that including the impa
    
[^2]: FedPop: 联邦式基于人口的超参数调优

    FedPop: Federated Population-based Hyperparameter Tuning. (arXiv:2308.08634v1 [cs.LG])

    [http://arxiv.org/abs/2308.08634](http://arxiv.org/abs/2308.08634)

    FedPop是一种用于解决联邦学习中超参数调优问题的新算法，它采用基于人口的进化算法来优化客户端和服务器上的超参数。

    

    联邦学习（FL）是一种分布式机器学习（ML）范式，多个客户端在不集中本地数据的情况下共同训练ML模型。与传统的ML流程类似，FL中的客户端本地优化和服务器聚合过程对超参数（HP）的选择非常敏感。尽管在集中式ML中对调优HP进行了广泛研究，但将这些方法应用于FL时会产生次优结果。这主要是因为它们的“调优后训练”框架对于计算能力有限的FL不合适。虽然一些方法已经提出用于FL中的HP调优，但这些方法仅限于客户端本地更新的HP。在这项工作中，我们提出了一种名为联邦式基于人口的超参数调优（FedPop）的新型HP调优算法，以解决这个重要但具有挑战性的问题。FedPop采用基于人口的进化算法来优化HP，此算法适用于客户端和服务器上的各种HP类型。

    Federated Learning (FL) is a distributed machine learning (ML) paradigm, in which multiple clients collaboratively train ML models without centralizing their local data. Similar to conventional ML pipelines, the client local optimization and server aggregation procedure in FL are sensitive to the hyperparameter (HP) selection. Despite extensive research on tuning HPs for centralized ML, these methods yield suboptimal results when employed in FL. This is mainly because their "training-after-tuning" framework is unsuitable for FL with limited client computation power. While some approaches have been proposed for HP-Tuning in FL, they are limited to the HPs for client local updates. In this work, we propose a novel HP-tuning algorithm, called Federated Population-based Hyperparameter Tuning (FedPop), to address this vital yet challenging problem. FedPop employs population-based evolutionary algorithms to optimize the HPs, which accommodates various HP types at both client and server sides
    
[^3]: 高效的半监督异构参与者联邦学习

    Efficient Semi-Supervised Federated Learning for Heterogeneous Participants. (arXiv:2307.15870v1 [cs.LG])

    [http://arxiv.org/abs/2307.15870](http://arxiv.org/abs/2307.15870)

    本论文提出了一种高效的半监督异构参与者联邦学习系统，通过引入聚类正则化来改进模型在数据非独立同分布情况下的性能，并对模型收敛性进行了理论和实验研究。

    

    联邦学习（FL）允许多个客户端在私有数据上协同训练机器学习模型，但在资源有限的环境中训练和部署大型模型用于广泛应用是具有挑战性的。幸运的是，分离式联邦学习（SFL）通过减轻客户端的计算和通信负担提供了优秀的解决方案。SFL通常假设客户端具有标记的数据进行本地训练，然而在实践中并非总是如此。以前的研究采用半监督技术来利用FL中的无标记数据，但数据的非独立同分布性提出了确保训练效率的另一个挑战。在这里，我们提出了一种新颖的系统Pseudo-Clustering Semi-SFL，用于在标记数据位于服务器上的情境下训练模型。通过引入聚类正则化，可以提高数据非独立同分布情况下的模型性能。此外，我们对模型收敛性进行了理论和实验研究，发现了...

    Federated Learning (FL) has emerged to allow multiple clients to collaboratively train machine learning models on their private data. However, training and deploying large models for broader applications is challenging in resource-constrained environments. Fortunately, Split Federated Learning (SFL) offers an excellent solution by alleviating the computation and communication burden on the clients SFL often assumes labeled data for local training on clients, however, it is not the case in practice.Prior works have adopted semi-supervised techniques for leveraging unlabeled data in FL, but data non-IIDness poses another challenge to ensure training efficiency. Herein, we propose Pseudo-Clustering Semi-SFL, a novel system for training models in scenarios where labeled data reside on the server. By introducing Clustering Regularization, model performance under data non-IIDness can be improved. Besides, our theoretical and experimental investigations into model convergence reveal that the 
    

