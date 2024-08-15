# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decentralized Federated Unlearning on Blockchain](https://arxiv.org/abs/2402.16294) | 提出了基于区块链的联邦遗忘（BlockFUL），使用Chameleon Hash（CH）技术重新设计区块链结构，减少模型更新的复杂性和成本。 |
| [^2] | [LearnDefend: Learning to Defend against Targeted Model-Poisoning Attacks on Federated Learning.](http://arxiv.org/abs/2305.02022) | LearnDefend是一种学习防御策略，能够有效地对抗联邦学习系统中的有针对性模型中毒攻击。它使用一个较小的防御数据集，估计客户端更新被污染的概率，通过学习毒数据检测器模型并使用耦合的优化方法估计毒数据检测器和客户端重要性模型。 |

# 详细

[^1]: 区块链上的去中心化联邦遗忘

    Decentralized Federated Unlearning on Blockchain

    [https://arxiv.org/abs/2402.16294](https://arxiv.org/abs/2402.16294)

    提出了基于区块链的联邦遗忘（BlockFUL），使用Chameleon Hash（CH）技术重新设计区块链结构，减少模型更新的复杂性和成本。

    

    区块链联邦学习（FL）在确保FL过程的完整性和可追溯性方面越来越受到关注。区块链FL涉及参与者在本地训练模型并随后将模型发布到区块链上，形成表示模型关系的类似有向无环图（DAG）的继承结构。然而，这种基于DAG的结构在使用敏感数据更新模型时存在挑战，因为涉及的复杂性和开销较大。为了解决这个问题，我们提出了基于区块链的联邦遗忘（BlockFUL），这是一个通用框架，使用变色龙哈希（CH）技术重新设计区块链结构，以减轻模型更新的复杂性，从而降低遗忘任务的计算和共识成本。此外，BlockFUL支持各种联邦遗忘方法，确保模型更新的完整性和可追溯性。

    arXiv:2402.16294v1 Announce Type: cross  Abstract: Blockchained Federated Learning (FL) has been gaining traction for ensuring the integrity and traceability of FL processes. Blockchained FL involves participants training models locally with their data and subsequently publishing the models on the blockchain, forming a Directed Acyclic Graph (DAG)-like inheritance structure that represents the model relationship. However, this particular DAG-based structure presents challenges in updating models with sensitive data, due to the complexity and overhead involved. To address this, we propose Blockchained Federated Unlearning (BlockFUL), a generic framework that redesigns the blockchain structure using Chameleon Hash (CH) technology to mitigate the complexity of model updating, thereby reducing the computational and consensus costs of unlearning tasks.Furthermore, BlockFUL supports various federated unlearning methods, ensuring the integrity and traceability of model updates, whether conduc
    
[^2]: LearnDefend：学习对抗联邦学习中的有针对性的模型中毒攻击

    LearnDefend: Learning to Defend against Targeted Model-Poisoning Attacks on Federated Learning. (arXiv:2305.02022v1 [cs.LG])

    [http://arxiv.org/abs/2305.02022](http://arxiv.org/abs/2305.02022)

    LearnDefend是一种学习防御策略，能够有效地对抗联邦学习系统中的有针对性模型中毒攻击。它使用一个较小的防御数据集，估计客户端更新被污染的概率，通过学习毒数据检测器模型并使用耦合的优化方法估计毒数据检测器和客户端重要性模型。

    

    面向联邦学习系统的有针对性模型中毒攻击构成了巨大的威胁。最近的研究显示，目标边缘案例型攻击（对输入空间的一小部分进行针对性攻击）几乎无法通过现有的防御策略进行反击。本文旨在通过使用较小的防御数据集设计一种学习防御策略来应对此类攻击。防御数据集可以由联邦学习任务的中央管理机构收集，其中应包含一些被污染的和没有被污染的示例。所提出的框架LearnDefend会估计客户端更新具有恶意的概率。防御数据集中的示例不需要事先标记为被污染或未被污染。我们还学习了一个可用于标记防御数据集中每个示例为干净或污染的毒数据检测器模型。我们使用耦合的优化方法来估计毒数据检测器和客户端重要性模型。我们的实验表明，LearnDefend能够成功应对有针对性模型中毒攻击。

    Targeted model poisoning attacks pose a significant threat to federated learning systems. Recent studies show that edge-case targeted attacks, which target a small fraction of the input space are nearly impossible to counter using existing fixed defense strategies. In this paper, we strive to design a learned-defense strategy against such attacks, using a small defense dataset. The defense dataset can be collected by the central authority of the federated learning task, and should contain a mix of poisoned and clean examples. The proposed framework, LearnDefend, estimates the probability of a client update being malicious. The examples in defense dataset need not be pre-marked as poisoned or clean. We also learn a poisoned data detector model which can be used to mark each example in the defense dataset as clean or poisoned. We estimate the poisoned data detector and the client importance models in a coupled optimization approach. Our experiments demonstrate that LearnDefend is capable
    

