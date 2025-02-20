# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sampling-based Distributed Training with Message Passing Neural Network](https://arxiv.org/abs/2402.15106) | 该论文介绍了一种基于采样和分布式训练的消息传递神经网络（MPNN），能够有效解决边缘图神经网络在节点数量增加时的扩展挑战。 |
| [^2] | [Scalable Decentralized Algorithms for Online Personalized Mean Estimation](https://arxiv.org/abs/2402.12812) | 本研究提出了一种可扩展的分散算法框架，使代理能够自组织成图，并提出了两种协同均值估计算法，解决了每个代理在学习模型的同时识别具有相似分布客户的挑战。 |

# 详细

[^1]: 基于采样的消息传递神经网络分布式训练

    Sampling-based Distributed Training with Message Passing Neural Network

    [https://arxiv.org/abs/2402.15106](https://arxiv.org/abs/2402.15106)

    该论文介绍了一种基于采样和分布式训练的消息传递神经网络（MPNN），能够有效解决边缘图神经网络在节点数量增加时的扩展挑战。

    

    在这项研究中，我们介绍了一种基于域分解的消息传递神经网络（MPNN）分布式训练和推断方法。我们的目标是解决随着节点数量增加而扩展边缘图神经网络的挑战。通过我们的分布式训练方法，结合Nystrom-近似采样技术，我们提出了一种可扩展的图神经网络，称为DS-MPNN（其中D和S分别代表分布式和采样），能够扩展到$O(10^5)$个节点。我们在两个案例上验证了我们的采样和分布式训练方法：（a）Darcy流数据集和（b）2-D机翼的稳态RANS模拟，提供了与单GPU实现和基于节点的图卷积网络（GCNs）的比较。DS-MPNN模型表现出与单GPU实现相当的准确性，能够容纳比单个GPU实现更多数量的节点。

    arXiv:2402.15106v1 Announce Type: new  Abstract: In this study, we introduce a domain-decomposition-based distributed training and inference approach for message-passing neural networks (MPNN). Our objective is to address the challenge of scaling edge-based graph neural networks as the number of nodes increases. Through our distributed training approach, coupled with Nystr\"om-approximation sampling techniques, we present a scalable graph neural network, referred to as DS-MPNN (D and S standing for distributed and sampled, respectively), capable of scaling up to $O(10^5)$ nodes. We validate our sampling and distributed training approach on two cases: (a) a Darcy flow dataset and (b) steady RANS simulations of 2-D airfoils, providing comparisons with both single-GPU implementation and node-based graph convolution networks (GCNs). The DS-MPNN model demonstrates comparable accuracy to single-GPU implementation, can accommodate a significantly larger number of nodes compared to the single-
    
[^2]: 可扩展的分散算法用于在线个性化均值估计

    Scalable Decentralized Algorithms for Online Personalized Mean Estimation

    [https://arxiv.org/abs/2402.12812](https://arxiv.org/abs/2402.12812)

    本研究提出了一种可扩展的分散算法框架，使代理能够自组织成图，并提出了两种协同均值估计算法，解决了每个代理在学习模型的同时识别具有相似分布客户的挑战。

    

    在许多情况下，代理缺乏足够的数据直接学习模型。与其他代理合作可能有所帮助，但当本地数据分布不同时，会引入偏差-方差权衡。一个关键挑战是每个代理在学习模型的同时识别具有相似分布的客户，这个问题主要仍未解决。本研究着眼于一个简化版本的普遍问题，即每个代理随时间从实值分布中收集样本来估计其均值。现有算法面临着不切实际的空间和时间复杂度（与代理数量A的平方成正比）。为了解决可扩展性挑战，我们提出了一个框架，代理自组织成一个图，使得每个代理只能与选定数量的对等体r进行通信。我们介绍了两种协作均值估计算法：一种灵感来源于信念传播，另一种采用基于共识的方法。

    arXiv:2402.12812v1 Announce Type: new  Abstract: In numerous settings, agents lack sufficient data to directly learn a model. Collaborating with other agents may help, but it introduces a bias-variance trade-off, when local data distributions differ. A key challenge is for each agent to identify clients with similar distributions while learning the model, a problem that remains largely unresolved. This study focuses on a simplified version of the overarching problem, where each agent collects samples from a real-valued distribution over time to estimate its mean. Existing algorithms face impractical space and time complexities (quadratic in the number of agents A). To address scalability challenges, we propose a framework where agents self-organize into a graph, allowing each agent to communicate with only a selected number of peers r. We introduce two collaborative mean estimation algorithms: one draws inspiration from belief propagation, while the other employs a consensus-based appr
    

