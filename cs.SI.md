# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via Self-Supervised Learning](https://arxiv.org/abs/2402.11933) | SLADE通过自监督学习在边缘流中迅速检测动态异常，无需依赖标签，主要通过观察节点交互模式的偏差来检测节点状态转变。 |
| [^2] | [Heterophily-Aware Fair Recommendation using Graph Convolutional Networks](https://arxiv.org/abs/2402.03365) | 本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。 |
| [^3] | [Towards Learning from Graphs with Heterophily: Progress and Future.](http://arxiv.org/abs/2401.09769) | 本调查综合概述了关于从带有异质性的图中学习的现有研究，并根据学习策略、模型架构和实际应用等方面对方法进行了分类。同时讨论了现有研究的主要挑战，并提出了未来研究的潜在方向。 |
| [^4] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |

# 详细

[^1]: SLADE：通过自监督学习在边缘流中检测动态异常

    SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via Self-Supervised Learning

    [https://arxiv.org/abs/2402.11933](https://arxiv.org/abs/2402.11933)

    SLADE通过自监督学习在边缘流中迅速检测动态异常，无需依赖标签，主要通过观察节点交互模式的偏差来检测节点状态转变。

    

    为了检测真实世界图中的异常，如社交、电子邮件和金融网络，已经开发了各种方法。在大多数真实世界图随时间增长，自然地表示为边缘流的情况下，我们的目标是：(a)在异常发生时即时检测异常，(b)适应动态变化的状态，(c)处理动态异常标签的稀缺性。在本文中，我们提出了SLADE（边缘流异常检测的自监督学习），用于在边缘流中快速检测动态异常，而不依赖于标签。SLADE通过观察节点在时间上相互作用模式的偏差来检测节点进入异常状态的转变。为此，它训练一个深度神经网络执行两个自监督任务：(a)最小化节点表示中的漂移，(b)从短期生成长期交互模式。

    arXiv:2402.11933v1 Announce Type: new  Abstract: To detect anomalies in real-world graphs, such as social, email, and financial networks, various approaches have been developed. While they typically assume static input graphs, most real-world graphs grow over time, naturally represented as edge streams. In this context, we aim to achieve three goals: (a) instantly detecting anomalies as they occur, (b) adapting to dynamically changing states, and (c) handling the scarcity of dynamic anomaly labels. In this paper, we propose SLADE (Self-supervised Learning for Anomaly Detection in Edge Streams) for rapid detection of dynamic anomalies in edge streams, without relying on labels. SLADE detects the shifts of nodes into abnormal states by observing deviations in their interaction patterns over time. To this end, it trains a deep neural network to perform two self-supervised tasks: (a) minimizing drift in node representations and (b) generating long-term interaction patterns from short-term 
    
[^2]: 利用图卷积网络的異质友善推荐方法

    Heterophily-Aware Fair Recommendation using Graph Convolutional Networks

    [https://arxiv.org/abs/2402.03365](https://arxiv.org/abs/2402.03365)

    本文提出了一种利用图卷积网络的公平推荐系统，名为HetroFair，旨在提高项目侧的公平性。HetroFair使用公平注意力和异质性特征加权两个组件来生成具有公平性意识的嵌入。

    

    近年来，图神经网络（GNNs）已成为提高推荐系统准确性和性能的流行工具。现代推荐系统不仅设计为为最终用户服务，还要让其他参与者（如项目和项目供应商）从中受益。这些参与者可能具有不同或冲突的目标和利益，这引发了对公平性和流行度偏差考虑的需求。基于GNN的推荐方法也面临不公平性和流行度偏差的挑战，其归一化和聚合过程受到这些挑战的影响。在本文中，我们提出了一种公平的基于GNN的推荐系统，称为HetroFair，旨在提高项目侧的公平性。HetroFair使用两个独立的组件生成具有公平性意识的嵌入：i）公平注意力，它在GNN的归一化过程中结合了点积，以减少节点度数的影响；ii）异质性特征加权，为不同的特征分配不同的权重。

    In recent years, graph neural networks (GNNs) have become a popular tool to improve the accuracy and performance of recommender systems. Modern recommender systems are not only designed to serve the end users, but also to benefit other participants, such as items and items providers. These participants may have different or conflicting goals and interests, which raise the need for fairness and popularity bias considerations. GNN-based recommendation methods also face the challenges of unfairness and popularity bias and their normalization and aggregation processes suffer from these challenges. In this paper, we propose a fair GNN-based recommender system, called HetroFair, to improve items' side fairness. HetroFair uses two separate components to generate fairness-aware embeddings: i) fairness-aware attention which incorporates dot product in the normalization process of GNNs, to decrease the effect of nodes' degrees, and ii) heterophily feature weighting to assign distinct weights to 
    
[^3]: 走向异质图学习：进展与未来

    Towards Learning from Graphs with Heterophily: Progress and Future. (arXiv:2401.09769v1 [cs.SI])

    [http://arxiv.org/abs/2401.09769](http://arxiv.org/abs/2401.09769)

    本调查综合概述了关于从带有异质性的图中学习的现有研究，并根据学习策略、模型架构和实际应用等方面对方法进行了分类。同时讨论了现有研究的主要挑战，并提出了未来研究的潜在方向。

    

    图是用来模拟现实世界实体之间复杂关系的结构化数据。最近，异质图，其中连接的节点往往具有不同的标签或不同的特征，引起了广泛关注并找到了许多应用。与此同时，人们也在不断努力推进从异质图中学习。虽然有关该主题的调查存在，但它们只关注于异质图神经网络（GNNs），而忽略了异质图学习的其他子主题。在本调查中，我们全面回顾了关于从带有异质性的图中学习的现有研究。首先，我们收集了180多篇论文，介绍了该领域的发展。然后，我们根据层次分类法对现有方法进行了系统分类，包括学习策略、模型架构和实际应用。最后，我们讨论了现有研究的主要挑战，并突出了未来研究的潜在方向。

    Graphs are structured data that models complex relations between real-world entities. Heterophilous graphs, where linked nodes are prone to be with different labels or dissimilar features, have recently attracted significant attention and found many applications. Meanwhile, increasing efforts have been made to advance learning from heterophilous graphs. Although there exist surveys on the relevant topic, they focus on heterophilous GNNs, which are only sub-topics of heterophilous graph learning. In this survey, we comprehensively overview existing works on learning from graphs with heterophily.First, we collect over 180 publications and introduce the development of this field. Then, we systematically categorize existing methods based on a hierarchical taxonomy including learning strategies, model architectures and practical applications. Finally, we discuss the primary challenges of existing studies and highlight promising avenues for future research.More publication details and corres
    
[^4]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    

