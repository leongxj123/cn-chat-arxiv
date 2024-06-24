# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Solution Simplex Clustering for Heterogeneous Federated Learning](https://arxiv.org/abs/2403.03333) | 提出了Solution Simplex Clustered Federated Learning（SosicFL），通过学习解决方案单纯形的思想，为每个客户端分配单一区域，从而同时实现了学习本地和全局模型的目标。 |
| [^2] | [SRL: Scaling Distributed Reinforcement Learning to Over Ten Thousand Cores.](http://arxiv.org/abs/2306.16688) | SRL是一个可扩展，高效，可扩展的分布式强化学习系统，通过一种新的抽象框架统一了各种实际强化学习训练，并实现了精细优化。 |

# 详细

[^1]: Solution Simplex Clustering for Heterogeneous Federated Learning

    Solution Simplex Clustering for Heterogeneous Federated Learning

    [https://arxiv.org/abs/2403.03333](https://arxiv.org/abs/2403.03333)

    提出了Solution Simplex Clustered Federated Learning（SosicFL），通过学习解决方案单纯形的思想，为每个客户端分配单一区域，从而同时实现了学习本地和全局模型的目标。

    

    我们针对联邦学习（FL）中的一个主要挑战提出了解决方案，即在高度异构的客户分布下实现良好的性能。这种困难部分源于两个看似矛盾的目标：通过聚合来自客户端的信息来学习一个通用模型，以及学习应适应每个本地分布的本地个性化模型。在这项工作中，我们提出了Solution Simplex Clustered Federated Learning（SosicFL）来消除这种矛盾。基于学习解决方案单纯形的最新思想，SosicFL为每个客户端分配一个单纯形中的子区域，并执行FL来学习一个通用解决方案单纯形。这使得客户端模型在解决方案单纯形的自由度范围内具有其特征，同时实现了学习一个全局通用模型的目标。我们的实验证明，SosicFL改善了性能，并加速了全局和训练过程。

    arXiv:2403.03333v1 Announce Type: new  Abstract: We tackle a major challenge in federated learning (FL) -- achieving good performance under highly heterogeneous client distributions. The difficulty partially arises from two seemingly contradictory goals: learning a common model by aggregating the information from clients, and learning local personalized models that should be adapted to each local distribution. In this work, we propose Solution Simplex Clustered Federated Learning (SosicFL) for dissolving such contradiction. Based on the recent ideas of learning solution simplices, SosicFL assigns a subregion in a simplex to each client, and performs FL to learn a common solution simplex. This allows the client models to possess their characteristics within the degrees of freedom in the solution simplex, and at the same time achieves the goal of learning a global common model. Our experiments show that SosicFL improves the performance and accelerates the training process for global and 
    
[^2]: SRL: 将分布式强化学习扩展到一万多个核心

    SRL: Scaling Distributed Reinforcement Learning to Over Ten Thousand Cores. (arXiv:2306.16688v1 [cs.DC])

    [http://arxiv.org/abs/2306.16688](http://arxiv.org/abs/2306.16688)

    SRL是一个可扩展，高效，可扩展的分布式强化学习系统，通过一种新的抽象框架统一了各种实际强化学习训练，并实现了精细优化。

    

    强化学习（RL）任务的不断复杂化要求分布式RL系统可以高效地生成和处理大量数据以训练智能Agent。然而，现有的开源库存在各种限制，阻碍了它们在需要大规模训练的挑战性场景中的实际应用。虽然OpenAI和DeepMind的工业系统已经成功实现了大规模RL训练，但是它们的系统架构和实现细节对社区来说仍然不公开。在本文中，我们提出了RL训练数据流的新抽象，将各种应用中的实际RL训练统一成一个通用框架，并实现了精细优化。根据这个抽象，我们开发了一个可扩展、高效、可扩展的分布式RL系统，名为"ReaLly Scalable RL（SRL）"。

    The ever-growing complexity of reinforcement learning (RL) tasks demands a distributed RL system to efficiently generate and process a massive amount of data to train intelligent agents. However, existing open-source libraries suffer from various limitations, which impede their practical use in challenging scenarios where large-scale training is necessary. While industrial systems from OpenAI and DeepMind have achieved successful large-scale RL training, their system architecture and implementation details remain undisclosed to the community. In this paper, we present a novel abstraction on the dataflows of RL training, which unifies practical RL training across diverse applications into a general framework and enables fine-grained optimizations. Following this abstraction, we develop a scalable, efficient, and extensible distributed RL system called ReaLly Scalable RL (SRL). The system architecture of SRL separates major RL computation components and allows massively parallelized trai
    

