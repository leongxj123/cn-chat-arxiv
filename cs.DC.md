# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Semi-Supervised Federated Learning for Heterogeneous Participants.](http://arxiv.org/abs/2307.15870) | 本论文提出了一种高效的半监督异构参与者联邦学习系统，通过引入聚类正则化来改进模型在数据非独立同分布情况下的性能，并对模型收敛性进行了理论和实验研究。 |
| [^2] | [Distributed Neural Representation for Reactive in situ Visualization.](http://arxiv.org/abs/2304.10516) | 本研究开发了一种分布式体积数据的隐式神经表示技术，结合到反应式编程系统中构建了一个原位时间缓存系统，并在Ascent基础架构中使用实际模拟评估了其100倍容量的性能。 |
| [^3] | [Does Federated Learning Really Need Backpropagation?.](http://arxiv.org/abs/2301.12195) | 本文提出一种不需要反向传播的联邦学习框架BAFFLE，该框架使用多个正向过程估计梯度，具有高内存效率，容易适应上传带宽，与硬件优化和模型量化/修剪兼容，适用于受信任的执行环境。 |

# 详细

[^1]: 高效的半监督异构参与者联邦学习

    Efficient Semi-Supervised Federated Learning for Heterogeneous Participants. (arXiv:2307.15870v1 [cs.LG])

    [http://arxiv.org/abs/2307.15870](http://arxiv.org/abs/2307.15870)

    本论文提出了一种高效的半监督异构参与者联邦学习系统，通过引入聚类正则化来改进模型在数据非独立同分布情况下的性能，并对模型收敛性进行了理论和实验研究。

    

    联邦学习（FL）允许多个客户端在私有数据上协同训练机器学习模型，但在资源有限的环境中训练和部署大型模型用于广泛应用是具有挑战性的。幸运的是，分离式联邦学习（SFL）通过减轻客户端的计算和通信负担提供了优秀的解决方案。SFL通常假设客户端具有标记的数据进行本地训练，然而在实践中并非总是如此。以前的研究采用半监督技术来利用FL中的无标记数据，但数据的非独立同分布性提出了确保训练效率的另一个挑战。在这里，我们提出了一种新颖的系统Pseudo-Clustering Semi-SFL，用于在标记数据位于服务器上的情境下训练模型。通过引入聚类正则化，可以提高数据非独立同分布情况下的模型性能。此外，我们对模型收敛性进行了理论和实验研究，发现了...

    Federated Learning (FL) has emerged to allow multiple clients to collaboratively train machine learning models on their private data. However, training and deploying large models for broader applications is challenging in resource-constrained environments. Fortunately, Split Federated Learning (SFL) offers an excellent solution by alleviating the computation and communication burden on the clients SFL often assumes labeled data for local training on clients, however, it is not the case in practice.Prior works have adopted semi-supervised techniques for leveraging unlabeled data in FL, but data non-IIDness poses another challenge to ensure training efficiency. Herein, we propose Pseudo-Clustering Semi-SFL, a novel system for training models in scenarios where labeled data reside on the server. By introducing Clustering Regularization, model performance under data non-IIDness can be improved. Besides, our theoretical and experimental investigations into model convergence reveal that the 
    
[^2]: 分布式神经表示技术用于反应式原位可视化

    Distributed Neural Representation for Reactive in situ Visualization. (arXiv:2304.10516v1 [cs.DC])

    [http://arxiv.org/abs/2304.10516](http://arxiv.org/abs/2304.10516)

    本研究开发了一种分布式体积数据的隐式神经表示技术，结合到反应式编程系统中构建了一个原位时间缓存系统，并在Ascent基础架构中使用实际模拟评估了其100倍容量的性能。

    

    利用反应式编程实现计算模型的原位可视化和控制十分高效，它利用时间抽象和数据缓存机制来创建动态工作流。然而，对于大规模模拟，实现时间缓存可能存在挑战。隐式神经网络已被证明在压缩大型数据方面是有效的。然而，它们在分布式数据上的应用还没有被充分探索。在本研究中，我们开发了一种分布式体积数据的隐式神经表示，并将其结合到DIVA反应式编程系统中。这种实现使我们能够构建一个原位时间缓存系统，其容量比以前的容量大100倍。我们将这种方法集成到Ascent基础架构中，并使用实际模拟来评估其性能。

    In situ visualization and steering of computational modeling can be effectively achieved using reactive programming, which leverages temporal abstraction and data caching mechanisms to create dynamic workflows. However, implementing a temporal cache for large-scale simulations can be challenging. Implicit neural networks have proven effective in compressing large volume data. However, their application to distributed data has yet to be fully explored. In this work, we develop an implicit neural representation for distributed volume data and incorporate it into the DIVA reactive programming system. This implementation enables us to build an in situ temporal caching system with a capacity 100 times larger than previously achieved. We integrate our implementation into the Ascent infrastructure and evaluate its performance using real-world simulations.
    
[^3]: 《联邦学习是否真正需要反向传播？》

    Does Federated Learning Really Need Backpropagation?. (arXiv:2301.12195v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12195](http://arxiv.org/abs/2301.12195)

    本文提出一种不需要反向传播的联邦学习框架BAFFLE，该框架使用多个正向过程估计梯度，具有高内存效率，容易适应上传带宽，与硬件优化和模型量化/修剪兼容，适用于受信任的执行环境。

    

    联邦学习（FL）是一种去中心化地让客户端共同训练一个服务器模型的一般性原则，而无需共享本地数据。FL是一个具有实际应用的有前途的框架，但其标准训练范式要求客户端通过模型进行反向传播以计算梯度。由于这些客户端通常是边缘设备而不是完全受信任的，因此在它们上执行反向传播会产生计算和存储开销以及白盒漏洞。因此，我们开发了一种不需要反向传播的联邦学习，称为BAFFLE，其中反向传播替换为多个正向过程以估计梯度。BAFFLE具有以下优点：1）内存效率高并且容易适应上传带宽；2）与仅推理硬件优化以及模型量化或修剪兼容；3）非常适合受信任的执行环境，因为BAFFLE中的客户端仅执行正向传播并返回一组标量到服务器。我们通过实验使用了BAFFLE的优越性能。

    Federated learning (FL) is a general principle for decentralized clients to train a server model collectively without sharing local data. FL is a promising framework with practical applications, but its standard training paradigm requires the clients to backpropagate through the model to compute gradients. Since these clients are typically edge devices and not fully trusted, executing backpropagation on them incurs computational and storage overhead as well as white-box vulnerability. In light of this, we develop backpropagation-free federated learning, dubbed BAFFLE, in which backpropagation is replaced by multiple forward processes to estimate gradients. BAFFLE is 1) memory-efficient and easily fits uploading bandwidth; 2) compatible with inference-only hardware optimization and model quantization or pruning; and 3) well-suited to trusted execution environments, because the clients in BAFFLE only execute forward propagation and return a set of scalars to the server. Empirically we us
    

