# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Strict Partitioning for Sporadic Rigid Gang Tasks](https://arxiv.org/abs/2403.10726) | 提出了一种新的严格分区调度策略，用于零星刚性流式任务，通过创建不相交的任务和处理器分区，并尝试将相似容量的任务分配给同一分区，以减少干扰。 |
| [^2] | [Sentinel: An Aggregation Function to Secure Decentralized Federated Learning.](http://arxiv.org/abs/2310.08097) | Sentinel是一种用于保护分散式联邦学习的防御策略，通过利用本地数据并定义一个三步聚合协议来对抗污染攻击。评估结果表明Sentinel在不同数据集和评估指标下表现良好。 |

# 详细

[^1]: 针对零星刚性流式任务的严格分区方法

    Strict Partitioning for Sporadic Rigid Gang Tasks

    [https://arxiv.org/abs/2403.10726](https://arxiv.org/abs/2403.10726)

    提出了一种新的严格分区调度策略，用于零星刚性流式任务，通过创建不相交的任务和处理器分区，并尝试将相似容量的任务分配给同一分区，以减少干扰。

    

    刚性流式任务模型基于在固定数量的处理器上同时执行多个线程以提高效率和性能的思想。虽然全局刚性流式调度有大量文献，但分区方法具有几个实际优势（例如任务隔离和减少调度开销）。本文提出了一种新的用于刚性流式任务的分区调度策略，称为严格分区。该方法创建任务和处理器的不相交分区，以避免分区间干扰。此外，它尝试将具有相似容量（即并行性）的任务分配给同一分区，以减少分区内干扰。在每个分区内，任务可以使用任何类型的调度器进行调度，这允许使用不那么悲观的可调度测试。大量的合成实验证明和基于Edge TPU基准的案例研究显示

    arXiv:2403.10726v1 Announce Type: cross  Abstract: The rigid gang task model is based on the idea of executing multiple threads simultaneously on a fixed number of processors to increase efficiency and performance. Although there is extensive literature on global rigid gang scheduling, partitioned approaches have several practical advantages (e.g., task isolation and reduced scheduling overheads). In this paper, we propose a new partitioned scheduling strategy for rigid gang tasks, named strict partitioning. The method creates disjoint partitions of tasks and processors to avoid inter-partition interference. Moreover, it tries to assign tasks with similar volumes (i.e., parallelisms) to the same partition so that the intra-partition interference can be reduced. Within each partition, the tasks can be scheduled using any type of scheduler, which allows the use of a less pessimistic schedulability test. Extensive synthetic experiments and a case study based on Edge TPU benchmarks show th
    
[^2]: Sentinel: 一种用于保护分散式联邦学习的聚合函数

    Sentinel: An Aggregation Function to Secure Decentralized Federated Learning. (arXiv:2310.08097v1 [cs.DC])

    [http://arxiv.org/abs/2310.08097](http://arxiv.org/abs/2310.08097)

    Sentinel是一种用于保护分散式联邦学习的防御策略，通过利用本地数据并定义一个三步聚合协议来对抗污染攻击。评估结果表明Sentinel在不同数据集和评估指标下表现良好。

    

    将联邦学习（FL）快速整合到网络中涵盖了网络管理、服务质量和网络安全等各个方面，同时保护数据隐私。在这种情况下，分散式联邦学习（DFL）作为一种创新范式，用于训练协作模型，解决了单点失效的限制。然而，FL和DFL的安全性和可信性受到污染攻击的影响，从而对其性能产生负面影响。现有的防御机制针对集中式FL进行设计，并未充分利用DFL的特点。因此，本文引入了Sentinel，一种在DFL中对抗污染攻击的防御策略。Sentinel利用本地数据的可访问性，定义了一个三步聚合协议，包括相似性过滤、引导验证和标准化，以防止恶意模型更新。通过使用不同数据集和不同的评估指标对Sentinel进行了评估。

    The rapid integration of Federated Learning (FL) into networking encompasses various aspects such as network management, quality of service, and cybersecurity while preserving data privacy. In this context, Decentralized Federated Learning (DFL) emerges as an innovative paradigm to train collaborative models, addressing the single point of failure limitation. However, the security and trustworthiness of FL and DFL are compromised by poisoning attacks, negatively impacting its performance. Existing defense mechanisms have been designed for centralized FL and they do not adequately exploit the particularities of DFL. Thus, this work introduces Sentinel, a defense strategy to counteract poisoning attacks in DFL. Sentinel leverages the accessibility of local data and defines a three-step aggregation protocol consisting of similarity filtering, bootstrap validation, and normalization to safeguard against malicious model updates. Sentinel has been evaluated with diverse datasets and various 
    

