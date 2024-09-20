# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Explanation-Guided Fair Federated Learning for Transparent 6G RAN Slicing.](http://arxiv.org/abs/2307.09494) | 这篇论文提出了一个解释引导的联邦学习方案，通过利用可解释的人工智能策略产生透明和无偏的深度神经网络，从而确保可靠的预测。 |
| [^2] | [Node Feature Augmentation Vitaminizes Network Alignment.](http://arxiv.org/abs/2304.12751) | 本研究提出了Grad-Align+方法，通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法，解决了NA方法缺乏额外信息的问题。 |

# 详细

[^1]: 透明的6G RAN切片中基于解释的公平联邦学习

    Explanation-Guided Fair Federated Learning for Transparent 6G RAN Slicing. (arXiv:2307.09494v1 [cs.NI])

    [http://arxiv.org/abs/2307.09494](http://arxiv.org/abs/2307.09494)

    这篇论文提出了一个解释引导的联邦学习方案，通过利用可解释的人工智能策略产生透明和无偏的深度神经网络，从而确保可靠的预测。

    

    未来的零触摸人工智能驱动的6G网络自动化需要通过可解释的人工智能建立对AI黑盒子的信任，预计AI的可信度将与通信关键性能指标一起作为可量化的服务级别协议指标。这需要利用可解释人工智能输出来生成透明和无偏的深度神经网络。我们设计了一个基于解释的联邦学习方案(EGFL)来确保在训练运行时通过Jensen-Shannon (JS)散度利用XAI策略的模型解释以确保可靠的预测。具体而言，我们通过将回忆度指标作为优化任务的约束条件，预测每个切片RAN的丢包概率来说明所提出的概念。

    Future zero-touch artificial intelligence (AI)-driven 6G network automation requires building trust in the AI black boxes via explainable artificial intelligence (XAI), where it is expected that AI faithfulness would be a quantifiable service-level agreement (SLA) metric along with telecommunications key performance indicators (KPIs). This entails exploiting the XAI outputs to generate transparent and unbiased deep neural networks (DNNs). Motivated by closed-loop (CL) automation and explanation-guided learning (EGL), we design an explanation-guided federated learning (EGFL) scheme to ensure trustworthy predictions by exploiting the model explanation emanating from XAI strategies during the training run time via Jensen-Shannon (JS) divergence. Specifically, we predict per-slice RAN dropped traffic probability to exemplify the proposed concept while respecting fairness goals formulated in terms of the recall metric which is included as a constraint in the optimization task. Finally, the 
    
[^2]: 节点特征增强改进网络对齐

    Node Feature Augmentation Vitaminizes Network Alignment. (arXiv:2304.12751v1 [cs.SI])

    [http://arxiv.org/abs/2304.12751](http://arxiv.org/abs/2304.12751)

    本研究提出了Grad-Align+方法，通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法，解决了NA方法缺乏额外信息的问题。

    

    网络对齐（NA）是通过给定网络的拓扑和/或特征信息来发现多个网络之间的节点对应关系的任务。虽然NA方法在各种场景下取得了显著的成功，但其有效性并不总是有额外信息，如先前的锚点链接和/或节点特征。为了解决这个实际的挑战，我们提出了Grad-Align+，这是一种新颖的NA方法，建立在最近一种最先进的NA方法Grad-Align之上，Grad-Align+仅逐步发现部分节点对，直到找到所有节点对。在设计Grad-Align+时，我们考虑如何通过增强节点特征来执行NA任务，并最大限度地利用增强的节点特征来设计NA方法。为了实现这个目标，我们开发了由三个关键组成部分组成的Grad-Align+：基于中心性的节点特征增强（CNFA）、图切片生成和优化节点嵌入特征（ONIFE）。

    Network alignment (NA) is the task of discovering node correspondences across multiple networks using topological and/or feature information of given networks. Although NA methods have achieved remarkable success in a myriad of scenarios, their effectiveness is not without additional information such as prior anchor links and/or node features, which may not always be available due to privacy concerns or access restrictions. To tackle this practical challenge, we propose Grad-Align+, a novel NA method built upon a recent state-of-the-art NA method, the so-called Grad-Align, that gradually discovers only a part of node pairs until all node pairs are found. In designing Grad-Align+, we account for how to augment node features in the sense of performing the NA task and how to design our NA method by maximally exploiting the augmented node features. To achieve this goal, we develop Grad-Align+ consisting of three key components: 1) centrality-based node feature augmentation (CNFA), 2) graph
    

