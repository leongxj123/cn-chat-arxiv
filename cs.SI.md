# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Self-Supervised Pre-Training of Graph Foundation Models: A Knowledge-Based Perspective](https://arxiv.org/abs/2403.16137) | 该论文从基于知识的角度全面调查和分析了图基础模型的自监督预训练任务，涉及微观和宏观知识，包括9个知识类别、25个预训练任务以及各种下游任务适应策略。 |
| [^2] | [A Unified Framework for Exploratory Learning-Aided Community Detection in Networks with Unknown Topology.](http://arxiv.org/abs/2304.04497) | META-CODE是一个统一的框架，通过探索学习和易于收集的节点元数据，在未知拓扑网络中检测重叠社区。实验结果证明了META-CODE的有效性和可扩展性。 |

# 详细

[^1]: 自监督预训练图基础模型的调查：基于知识的视角

    A Survey on Self-Supervised Pre-Training of Graph Foundation Models: A Knowledge-Based Perspective

    [https://arxiv.org/abs/2403.16137](https://arxiv.org/abs/2403.16137)

    该论文从基于知识的角度全面调查和分析了图基础模型的自监督预训练任务，涉及微观和宏观知识，包括9个知识类别、25个预训练任务以及各种下游任务适应策略。

    

    图自监督学习现在是预训练图基础模型的首选方法，包括图神经网络、图变换器，以及更近期的基于大型语言模型（LLM）的图模型。文章全面调查和分析了基于知识的视角下的图基础模型的预训练任务，包括微观（节点、链接等）和宏观知识（簇、全局结构等）。涵盖了共计9个知识类别和25个预训练任务，以及各种下游任务适应策略。

    arXiv:2403.16137v1 Announce Type: new  Abstract: Graph self-supervised learning is now a go-to method for pre-training graph foundation models, including graph neural networks, graph transformers, and more recent large language model (LLM)-based graph models. There is a wide variety of knowledge patterns embedded in the structure and properties of graphs which may be used for pre-training, but we lack a systematic overview of self-supervised pre-training tasks from the perspective of graph knowledge. In this paper, we comprehensively survey and analyze the pre-training tasks of graph foundation models from a knowledge-based perspective, consisting of microscopic (nodes, links, etc) and macroscopic knowledge (clusters, global structure, etc). It covers a total of 9 knowledge categories and 25 pre-training tasks, as well as various downstream task adaptation strategies. Furthermore, an extensive list of the related papers with detailed metadata is provided at https://github.com/Newiz430/
    
[^2]: 未知拓扑网络中的探索学习辅助社区检测的统一框架

    A Unified Framework for Exploratory Learning-Aided Community Detection in Networks with Unknown Topology. (arXiv:2304.04497v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2304.04497](http://arxiv.org/abs/2304.04497)

    META-CODE是一个统一的框架，通过探索学习和易于收集的节点元数据，在未知拓扑网络中检测重叠社区。实验结果证明了META-CODE的有效性和可扩展性。

    

    在社交网络中，发现社区结构作为各种网络分析任务中的一个基本问题受到了广泛关注。然而，由于隐私问题或访问限制，网络结构通常是未知的，这使得现有的社区检测方法在没有昂贵的网络拓扑获取的情况下无效。为了解决这个挑战，我们提出了 META-CODE，这是一个统一的框架，通过探索学习辅助易于收集的节点元数据，在未知拓扑网络中检测重叠社区。具体而言，META-CODE 除了初始的网络推理步骤外，还包括三个迭代步骤：1) 基于图神经网络（GNNs）的节点级社区归属嵌入，通过我们的新重构损失进行训练，2) 基于社区归属的节点查询进行网络探索，3) 使用探索网络中的基于边连接的连体神经网络模型进行网络推理。通过实验结果证明了 META-CODE 的有效性和可扩展性。

    In social networks, the discovery of community structures has received considerable attention as a fundamental problem in various network analysis tasks. However, due to privacy concerns or access restrictions, the network structure is often unknown, thereby rendering established community detection approaches ineffective without costly network topology acquisition. To tackle this challenge, we present META-CODE, a unified framework for detecting overlapping communities in networks with unknown topology via exploratory learning aided by easy-to-collect node metadata. Specifically, META-CODE consists of three iterative steps in addition to the initial network inference step: 1) node-level community-affiliation embeddings based on graph neural networks (GNNs) trained by our new reconstruction loss, 2) network exploration via community-affiliation-based node queries, and 3) network inference using an edge connectivity-based Siamese neural network model from the explored network. Through e
    

