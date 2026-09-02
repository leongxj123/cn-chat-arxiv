# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedReview: A Review Mechanism for Rejecting Poisoned Updates in Federated Learning](https://arxiv.org/abs/2402.16934) | 提出了FedReview机制，通过随机分配评审员客户端来识别和拒绝联邦学习中的潜在毒化更新，并采用多数表决机制来整合排名并移除这些更新。 |
| [^2] | [Building Expressive and Tractable Probabilistic Generative Models: A Review](https://arxiv.org/abs/2402.00759) | 本文综述了富有表现力和可处理的概率生成建模领域的进展和技术，并重点关注了概率电路。文章提供了关于表达能力和可处理性之间权衡的统一视角，并说明了设计原则和算法扩展，成功地构建了富有表现力和高效的概率电路。此外，文章还讨论了最新的深度和混合概率电路研究，并概述了未来研究的挑战和开放性问题。 |

# 详细

[^1]: FedReview: 一种用于拒绝毒化更新的联邦学习审查机制

    FedReview: A Review Mechanism for Rejecting Poisoned Updates in Federated Learning

    [https://arxiv.org/abs/2402.16934](https://arxiv.org/abs/2402.16934)

    提出了FedReview机制，通过随机分配评审员客户端来识别和拒绝联邦学习中的潜在毒化更新，并采用多数表决机制来整合排名并移除这些更新。

    

    Federated learning最近已经被提出作为一种去中心化的方法，在不访问用户数据的情况下学习一个高性能模型。尽管其有效性，但联邦学习给恶意用户提供了机会通过向服务器上传毒化模型更新来操纵模型。在本文中，我们提出了一种名为FedReview的审查机制，用于识别和拒绝联邦学习中潜在的毒化更新。在我们的机制下，服务器每轮随机分配子集客户端作为评审员，在其训练数据集上评估模型更新。评审员根据评价结果对模型更新进行排名，统计相对低质量的更新数量作为估计的毒化更新数量。基于审查报告，服务器采用多数表决机制整合排名并在模型聚合过程中去除潜在的毒化更新。

    arXiv:2402.16934v1 Announce Type: cross  Abstract: Federated learning has recently emerged as a decentralized approach to learn a high-performance model without access to user data. Despite its effectiveness, federated learning gives malicious users opportunities to manipulate the model by uploading poisoned model updates to the server. In this paper, we propose a review mechanism called FedReview to identify and decline the potential poisoned updates in federated learning. Under our mechanism, the server randomly assigns a subset of clients as reviewers to evaluate the model updates on their training datasets in each round. The reviewers rank the model updates based on the evaluation results and count the number of the updates with relatively low quality as the estimated number of poisoned updates. Based on review reports, the server employs a majority voting mechanism to integrate the rankings and remove the potential poisoned updates in the model aggregation process. Extensive evalu
    
[^2]: 构建富有表现力和可处理的概率生成模型：一项综述

    Building Expressive and Tractable Probabilistic Generative Models: A Review

    [https://arxiv.org/abs/2402.00759](https://arxiv.org/abs/2402.00759)

    本文综述了富有表现力和可处理的概率生成建模领域的进展和技术，并重点关注了概率电路。文章提供了关于表达能力和可处理性之间权衡的统一视角，并说明了设计原则和算法扩展，成功地构建了富有表现力和高效的概率电路。此外，文章还讨论了最新的深度和混合概率电路研究，并概述了未来研究的挑战和开放性问题。

    

    我们对可处理的概率生成建模领域中的进展和技术进行了全面的调查，重点关注概率电路（PCs）。我们提供了关于表达能力和可处理性之间固有权衡的统一视角，突出了使PCs富有表现力和高效的设计原则和算法扩展，并提供了该领域的分类法。我们还讨论了最近通过融合深度神经模型概念来构建深度和混合PCs的努力，并概述了指导未来研究的挑战和开放性问题。

    We present a comprehensive survey of the advancements and techniques in the field of tractable probabilistic generative modeling, primarily focusing on Probabilistic Circuits (PCs). We provide a unified perspective on the inherent trade-offs between expressivity and the tractability, highlighting the design principles and algorithmic extensions that have enabled building expressive and efficient PCs, and provide a taxonomy of the field. We also discuss recent efforts to build deep and hybrid PCs by fusing notions from deep neural models, and outline the challenges and open questions that can guide future research in this evolving field.
    

