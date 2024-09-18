# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Backdoor Attacks in Peer-to-Peer Federated Learning.](http://arxiv.org/abs/2301.09732) | 本文提出了一种基于点对点联邦学习（P2PFL）的新型后门攻击，利用结构图属性选择恶意节点，实现高攻击成功率，同时保持隐蔽性。同时还评估了这些攻击在多种现实条件下的鲁棒性，并设计了新的防御措施。 |

# 详细

[^1]: 点对点联邦学习中的后门攻击

    Backdoor Attacks in Peer-to-Peer Federated Learning. (arXiv:2301.09732v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.09732](http://arxiv.org/abs/2301.09732)

    本文提出了一种基于点对点联邦学习（P2PFL）的新型后门攻击，利用结构图属性选择恶意节点，实现高攻击成功率，同时保持隐蔽性。同时还评估了这些攻击在多种现实条件下的鲁棒性，并设计了新的防御措施。

    

    大多数机器学习应用程序依赖于集中式学习过程，这开放了曝光其训练数据集的风险。尽管联邦学习（FL）在某种程度上缓解了这些隐私风险，但它仍依赖于可信的聚合服务器来训练共享全局模型。最近，基于点对点联邦学习（P2PFL）的新分布式学习架构在隐私和可靠性方面都提供了优势。然而，在训练期间对毒化攻击的鲁棒性尚未得到研究。在本文中，我们提出了一种新的P2PFL后门攻击，利用结构图属性选择恶意节点，实现高攻击成功率，同时保持隐蔽性。我们在各种实际条件下评估我们的攻击，包括多个图形拓扑、网络中有限的敌对能见度以及具有非独立同分布数据的客户端。最后，我们展示了从FL中适应的现有防御措施的局限性，并设计了一种新的防御措施。

    Most machine learning applications rely on centralized learning processes, opening up the risk of exposure of their training datasets. While federated learning (FL) mitigates to some extent these privacy risks, it relies on a trusted aggregation server for training a shared global model. Recently, new distributed learning architectures based on Peer-to-Peer Federated Learning (P2PFL) offer advantages in terms of both privacy and reliability. Still, their resilience to poisoning attacks during training has not been investigated. In this paper, we propose new backdoor attacks for P2PFL that leverage structural graph properties to select the malicious nodes, and achieve high attack success, while remaining stealthy. We evaluate our attacks under various realistic conditions, including multiple graph topologies, limited adversarial visibility of the network, and clients with non-IID data. Finally, we show the limitations of existing defenses adapted from FL and design a new defense that su
    

