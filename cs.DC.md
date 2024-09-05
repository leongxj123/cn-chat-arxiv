# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Navigating the Maize: Cyclic and conditional computational graphs for molecular simulation](https://arxiv.org/abs/2402.10064) | 该论文介绍了一种用于分子模拟的循环和条件计算图的工作流管理器，通过并行化和通信实现任意图结构的执行，具有很高的实用性和效果。 |
| [^2] | [Sentinel: An Aggregation Function to Secure Decentralized Federated Learning.](http://arxiv.org/abs/2310.08097) | Sentinel是一种用于保护分散式联邦学习的防御策略，通过利用本地数据并定义一个三步聚合协议来对抗污染攻击。评估结果表明Sentinel在不同数据集和评估指标下表现良好。 |

# 详细

[^1]: 导航玉米：分子模拟的循环和条件计算图

    Navigating the Maize: Cyclic and conditional computational graphs for molecular simulation

    [https://arxiv.org/abs/2402.10064](https://arxiv.org/abs/2402.10064)

    该论文介绍了一种用于分子模拟的循环和条件计算图的工作流管理器，通过并行化和通信实现任意图结构的执行，具有很高的实用性和效果。

    

    许多计算化学和分子模拟工作流程可以表示为计算图。这种抽象有助于模块化和潜在地重用现有组件，并提供并行化和易于复制。现有工具将计算表示为有向无环图(DAG)，从而通过并行化并发分支来实现高效执行。然而，这些系统通常无法表示循环和条件工作流程。因此，我们开发了Maize，一种基于流程编程原理的、用于循环和条件图的工作流管理器。通过在单独的进程中同时运行图中的每个节点，并在任何时间通过专用的节点间通道进行通信，可以执行任意的图结构。我们通过在计算药物设计中进行动态主动学习任务来展示工具的有效性，其中涉及使用小分子 gen

    arXiv:2402.10064v1 Announce Type: cross  Abstract: Many computational chemistry and molecular simulation workflows can be expressed as graphs. This abstraction is useful to modularize and potentially reuse existing components, as well as provide parallelization and ease reproducibility. Existing tools represent the computation as a directed acyclic graph (DAG), thus allowing efficient execution by parallelization of concurrent branches. These systems can, however, generally not express cyclic and conditional workflows. We therefore developed Maize, a workflow manager for cyclic and conditional graphs based on the principles of flow-based programming. By running each node of the graph concurrently in separate processes and allowing communication at any time through dedicated inter-node channels, arbitrary graph structures can be executed. We demonstrate the effectiveness of the tool on a dynamic active learning task in computational drug design, involving the use of a small molecule gen
    
[^2]: Sentinel: 一种用于保护分散式联邦学习的聚合函数

    Sentinel: An Aggregation Function to Secure Decentralized Federated Learning. (arXiv:2310.08097v1 [cs.DC])

    [http://arxiv.org/abs/2310.08097](http://arxiv.org/abs/2310.08097)

    Sentinel是一种用于保护分散式联邦学习的防御策略，通过利用本地数据并定义一个三步聚合协议来对抗污染攻击。评估结果表明Sentinel在不同数据集和评估指标下表现良好。

    

    将联邦学习（FL）快速整合到网络中涵盖了网络管理、服务质量和网络安全等各个方面，同时保护数据隐私。在这种情况下，分散式联邦学习（DFL）作为一种创新范式，用于训练协作模型，解决了单点失效的限制。然而，FL和DFL的安全性和可信性受到污染攻击的影响，从而对其性能产生负面影响。现有的防御机制针对集中式FL进行设计，并未充分利用DFL的特点。因此，本文引入了Sentinel，一种在DFL中对抗污染攻击的防御策略。Sentinel利用本地数据的可访问性，定义了一个三步聚合协议，包括相似性过滤、引导验证和标准化，以防止恶意模型更新。通过使用不同数据集和不同的评估指标对Sentinel进行了评估。

    The rapid integration of Federated Learning (FL) into networking encompasses various aspects such as network management, quality of service, and cybersecurity while preserving data privacy. In this context, Decentralized Federated Learning (DFL) emerges as an innovative paradigm to train collaborative models, addressing the single point of failure limitation. However, the security and trustworthiness of FL and DFL are compromised by poisoning attacks, negatively impacting its performance. Existing defense mechanisms have been designed for centralized FL and they do not adequately exploit the particularities of DFL. Thus, this work introduces Sentinel, a defense strategy to counteract poisoning attacks in DFL. Sentinel leverages the accessibility of local data and defines a three-step aggregation protocol consisting of similarity filtering, bootstrap validation, and normalization to safeguard against malicious model updates. Sentinel has been evaluated with diverse datasets and various 
    

