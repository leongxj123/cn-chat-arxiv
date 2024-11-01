# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GreedyML: A Parallel Algorithm for Maximizing Submodular Functions](https://arxiv.org/abs/2403.10332) | 提出了一种用于在分布式存储多处理器上最大化子模函数的并行近似算法，以解决实际应用领域中海量数据集上的子模优化问题。 |
| [^2] | [Solution Simplex Clustering for Heterogeneous Federated Learning](https://arxiv.org/abs/2403.03333) | 提出了Solution Simplex Clustered Federated Learning（SosicFL），通过学习解决方案单纯形的思想，为每个客户端分配单一区域，从而同时实现了学习本地和全局模型的目标。 |

# 详细

[^1]: GreedyML：一种用于最大化子模函数的并行算法

    GreedyML: A Parallel Algorithm for Maximizing Submodular Functions

    [https://arxiv.org/abs/2403.10332](https://arxiv.org/abs/2403.10332)

    提出了一种用于在分布式存储多处理器上最大化子模函数的并行近似算法，以解决实际应用领域中海量数据集上的子模优化问题。

    

    我们描述了一种用于在分布式存储多处理器上最大化单调子模函数的并行近似算法。我们的工作受到在海量数据集上解决子模优化问题的需求的启发，用于实际应用领域，如数据摘要，机器学习和图稀疏化。我们的工作基于Barbosa、Ene、Nguyen和Ward（2015）提出的随机分布式RandGreedI算法。该算法通过将数据随机分区到所有处理器中，然后使用单个累积步骤计算分布式解决方案，其中所有处理器将它们的部分解决方案发送给一个处理器。然而，对于大问题，累积步骤可能超过处理器上可用的内存，并且执行累积的处理器可能成为计算瓶颈。

    arXiv:2403.10332v1 Announce Type: cross  Abstract: We describe a parallel approximation algorithm for maximizing monotone submodular functions subject to hereditary constraints on distributed memory multiprocessors. Our work is motivated by the need to solve submodular optimization problems on massive data sets, for practical applications in areas such as data summarization, machine learning, and graph sparsification. Our work builds on the randomized distributed RandGreedI algorithm, proposed by Barbosa, Ene, Nguyen, and Ward (2015). This algorithm computes a distributed solution by randomly partitioning the data among all the processors and then employing a single accumulation step in which all processors send their partial solutions to one processor. However, for large problems, the accumulation step could exceed the memory available on a processor, and the processor which performs the accumulation could become a computational bottleneck.   Here, we propose a generalization of the R
    
[^2]: Solution Simplex Clustering for Heterogeneous Federated Learning

    Solution Simplex Clustering for Heterogeneous Federated Learning

    [https://arxiv.org/abs/2403.03333](https://arxiv.org/abs/2403.03333)

    提出了Solution Simplex Clustered Federated Learning（SosicFL），通过学习解决方案单纯形的思想，为每个客户端分配单一区域，从而同时实现了学习本地和全局模型的目标。

    

    我们针对联邦学习（FL）中的一个主要挑战提出了解决方案，即在高度异构的客户分布下实现良好的性能。这种困难部分源于两个看似矛盾的目标：通过聚合来自客户端的信息来学习一个通用模型，以及学习应适应每个本地分布的本地个性化模型。在这项工作中，我们提出了Solution Simplex Clustered Federated Learning（SosicFL）来消除这种矛盾。基于学习解决方案单纯形的最新思想，SosicFL为每个客户端分配一个单纯形中的子区域，并执行FL来学习一个通用解决方案单纯形。这使得客户端模型在解决方案单纯形的自由度范围内具有其特征，同时实现了学习一个全局通用模型的目标。我们的实验证明，SosicFL改善了性能，并加速了全局和训练过程。

    arXiv:2403.03333v1 Announce Type: new  Abstract: We tackle a major challenge in federated learning (FL) -- achieving good performance under highly heterogeneous client distributions. The difficulty partially arises from two seemingly contradictory goals: learning a common model by aggregating the information from clients, and learning local personalized models that should be adapted to each local distribution. In this work, we propose Solution Simplex Clustered Federated Learning (SosicFL) for dissolving such contradiction. Based on the recent ideas of learning solution simplices, SosicFL assigns a subregion in a simplex to each client, and performs FL to learn a common solution simplex. This allows the client models to possess their characteristics within the degrees of freedom in the solution simplex, and at the same time achieves the goal of learning a global common model. Our experiments show that SosicFL improves the performance and accelerates the training process for global and 
    

