# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Locally Differentially Private Graph Embedding.](http://arxiv.org/abs/2310.11060) | 该论文提出了一种局部差分隐私图嵌入框架（LDP-GE），该框架采用LDP机制来保护节点数据的隐私，并使用个性化PageRank作为近似度度量来学习节点表示。大量实验证明，LDP-GE在隐私和效用方面取得了有利的折衷效果，并且明显优于现有的方法。 |
| [^2] | [A Survey on Knowledge Graphs for Healthcare: Resources, Applications, and Promises.](http://arxiv.org/abs/2306.04802) | 本论文综述了医疗知识图谱(HKGs)的构建流程、关键技术和利用方法以及现有资源，并深入探讨了HKG在各种医疗领域的变革性影响。 |
| [^3] | [SIMGA: A Simple and Effective Heterophilous Graph Neural Network with Efficient Global Aggregation.](http://arxiv.org/abs/2305.09958) | 本文章提出了一种简单有效的异质图神经网络模型SIMGA，它通过SimRank全局聚合来解决异质性节点聚合的问题，具有接近于线性的传播效率，同时具有良好的有效性和可扩展性。 |

# 详细

[^1]: 局部差分隐私图嵌入

    Locally Differentially Private Graph Embedding. (arXiv:2310.11060v1 [cs.CR])

    [http://arxiv.org/abs/2310.11060](http://arxiv.org/abs/2310.11060)

    该论文提出了一种局部差分隐私图嵌入框架（LDP-GE），该框架采用LDP机制来保护节点数据的隐私，并使用个性化PageRank作为近似度度量来学习节点表示。大量实验证明，LDP-GE在隐私和效用方面取得了有利的折衷效果，并且明显优于现有的方法。

    

    图嵌入被证明是学习图中节点潜在表示的强大工具。然而，尽管在各种基于图的机器学习任务中表现出卓越性能，但在涉及敏感信息的图数据上进行学习可能引发重大的隐私问题。为了解决这个问题，本文研究了开发能满足局部差分隐私（LDP）的图嵌入算法的问题。我们提出了一种新颖的隐私保护图嵌入框架LDP-GE，用于保护节点数据的隐私。具体而言，我们提出了一种LDP机制来混淆节点数据，并采用个性化PageRank作为近似度度量来学习节点表示。然后，我们从理论上分析了LDP-GE框架的隐私保证和效用。在几个真实世界的图数据集上进行的大量实验表明，LDP-GE在隐私-效用权衡方面取得了有利的效果，并且明显优于现有的方法。

    Graph embedding has been demonstrated to be a powerful tool for learning latent representations for nodes in a graph. However, despite its superior performance in various graph-based machine learning tasks, learning over graphs can raise significant privacy concerns when graph data involves sensitive information. To address this, in this paper, we investigate the problem of developing graph embedding algorithms that satisfy local differential privacy (LDP). We propose LDP-GE, a novel privacy-preserving graph embedding framework, to protect the privacy of node data. Specifically, we propose an LDP mechanism to obfuscate node data and adopt personalized PageRank as the proximity measure to learn node representations. Then, we theoretically analyze the privacy guarantees and utility of the LDP-GE framework. Extensive experiments conducted over several real-world graph datasets demonstrate that LDP-GE achieves favorable privacy-utility trade-offs and significantly outperforms existing appr
    
[^2]: 医疗知识图谱综述：资源、应用和前景

    A Survey on Knowledge Graphs for Healthcare: Resources, Applications, and Promises. (arXiv:2306.04802v1 [cs.AI])

    [http://arxiv.org/abs/2306.04802](http://arxiv.org/abs/2306.04802)

    本论文综述了医疗知识图谱(HKGs)的构建流程、关键技术和利用方法以及现有资源，并深入探讨了HKG在各种医疗领域的变革性影响。

    

    医疗知识图谱(HKGs)已成为组织医学知识的有结构且可解释的有为工具，提供了医学概念及其关系的全面视图。然而，数据异质性和覆盖范围有限等挑战仍然存在，强调了在HKG领域需要进一步研究的必要性。本综述是HKG的第一份综合概述。我们总结了HKG构建的流程和关键技术（即从头开始和通过集成），以及常见的利用方法（即基于模型和非基于模型）。为了为研究人员提供有价值的资源，我们根据它们捕获的数据类型和应用领域（该资源存储于https://github.com/lujiaying/Awesome-HealthCare-KnowledgeBase）组织了现有的HKG，并提供了相关的统计信息。在应用部分，我们深入探讨了HKG在各种医疗领域的变革性影响。

    Healthcare knowledge graphs (HKGs) have emerged as a promising tool for organizing medical knowledge in a structured and interpretable way, which provides a comprehensive view of medical concepts and their relationships. However, challenges such as data heterogeneity and limited coverage remain, emphasizing the need for further research in the field of HKGs. This survey paper serves as the first comprehensive overview of HKGs. We summarize the pipeline and key techniques for HKG construction (i.e., from scratch and through integration), as well as the common utilization approaches (i.e., model-free and model-based). To provide researchers with valuable resources, we organize existing HKGs (The resource is available at https://github.com/lujiaying/Awesome-HealthCare-KnowledgeBase) based on the data types they capture and application domains, supplemented with pertinent statistical information. In the application section, we delve into the transformative impact of HKGs across various hea
    
[^3]: SIMGA：一种简单有效的异质图神经网络结构与高效的全局聚合

    SIMGA: A Simple and Effective Heterophilous Graph Neural Network with Efficient Global Aggregation. (arXiv:2305.09958v1 [cs.LG])

    [http://arxiv.org/abs/2305.09958](http://arxiv.org/abs/2305.09958)

    本文章提出了一种简单有效的异质图神经网络模型SIMGA，它通过SimRank全局聚合来解决异质性节点聚合的问题，具有接近于线性的传播效率，同时具有良好的有效性和可扩展性。

    

    图神经网络在图学习领域取得了巨大成功，但遇到异质性时会出现性能下降，即因为局部和统一聚合而导致的相邻节点不相似。现有的异质性图神经网络中，试图整合全局聚合的尝试通常需要迭代地维护和更新全图信息，对于一个具有 $n$ 个节点的图，这需要 $\mathcal{O}(n^2)$ 的计算效率，从而导致对大型图的扩展性较差。在本文中，我们提出了 SIMGA，一种将 SimRank 结构相似度测量作为全局聚合的 GNN 结构。 SIMGA 的设计简单，且在效率和有效性方面都有着有 promising 的结果。SIMGA 的简单性使其成为第一个可以实现接近于线性的 $n$ 传播效率的异质性 GNN 模型。我们从理论上证明了它的有效性，将 SimRank 视为 GNN 的一种新解释，并证明了汇聚节点表示的有效性。

    Graph neural networks (GNNs) realize great success in graph learning but suffer from performance loss when meeting heterophily, i.e. neighboring nodes are dissimilar, due to their local and uniform aggregation. Existing attempts in incoorporating global aggregation for heterophilous GNNs usually require iteratively maintaining and updating full-graph information, which entails $\mathcal{O}(n^2)$ computation efficiency for a graph with $n$ nodes, leading to weak scalability to large graphs. In this paper, we propose SIMGA, a GNN structure integrating SimRank structural similarity measurement as global aggregation. The design of SIMGA is simple, yet it leads to promising results in both efficiency and effectiveness. The simplicity of SIMGA makes it the first heterophilous GNN model that can achieve a propagation efficiency near-linear to $n$. We theoretically demonstrate its effectiveness by treating SimRank as a new interpretation of GNN and prove that the aggregated node representation
    

