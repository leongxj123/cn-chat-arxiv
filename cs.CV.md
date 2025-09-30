# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decentralized Federated Unlearning on Blockchain](https://arxiv.org/abs/2402.16294) | 提出了基于区块链的联邦遗忘（BlockFUL），使用Chameleon Hash（CH）技术重新设计区块链结构，减少模型更新的复杂性和成本。 |
| [^2] | [Vision Langauge Pre-training by Contrastive Learning with Cross-Modal Similarity Regulation.](http://arxiv.org/abs/2305.04474) | 本文提出了一种跨模态相似性逐步细化的对比学习策略，在视觉语言预训练中优化图像/文本锚点与其负样本文本/图像之间的互信息，有效应对了（部分）误反样本的挑战。 |

# 详细

[^1]: 区块链上的去中心化联邦遗忘

    Decentralized Federated Unlearning on Blockchain

    [https://arxiv.org/abs/2402.16294](https://arxiv.org/abs/2402.16294)

    提出了基于区块链的联邦遗忘（BlockFUL），使用Chameleon Hash（CH）技术重新设计区块链结构，减少模型更新的复杂性和成本。

    

    区块链联邦学习（FL）在确保FL过程的完整性和可追溯性方面越来越受到关注。区块链FL涉及参与者在本地训练模型并随后将模型发布到区块链上，形成表示模型关系的类似有向无环图（DAG）的继承结构。然而，这种基于DAG的结构在使用敏感数据更新模型时存在挑战，因为涉及的复杂性和开销较大。为了解决这个问题，我们提出了基于区块链的联邦遗忘（BlockFUL），这是一个通用框架，使用变色龙哈希（CH）技术重新设计区块链结构，以减轻模型更新的复杂性，从而降低遗忘任务的计算和共识成本。此外，BlockFUL支持各种联邦遗忘方法，确保模型更新的完整性和可追溯性。

    arXiv:2402.16294v1 Announce Type: cross  Abstract: Blockchained Federated Learning (FL) has been gaining traction for ensuring the integrity and traceability of FL processes. Blockchained FL involves participants training models locally with their data and subsequently publishing the models on the blockchain, forming a Directed Acyclic Graph (DAG)-like inheritance structure that represents the model relationship. However, this particular DAG-based structure presents challenges in updating models with sensitive data, due to the complexity and overhead involved. To address this, we propose Blockchained Federated Unlearning (BlockFUL), a generic framework that redesigns the blockchain structure using Chameleon Hash (CH) technology to mitigate the complexity of model updating, thereby reducing the computational and consensus costs of unlearning tasks.Furthermore, BlockFUL supports various federated unlearning methods, ensuring the integrity and traceability of model updates, whether conduc
    
[^2]: 跨模态相似性调节的对比学习在视觉语言预训练中的应用

    Vision Langauge Pre-training by Contrastive Learning with Cross-Modal Similarity Regulation. (arXiv:2305.04474v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.04474](http://arxiv.org/abs/2305.04474)

    本文提出了一种跨模态相似性逐步细化的对比学习策略，在视觉语言预训练中优化图像/文本锚点与其负样本文本/图像之间的互信息，有效应对了（部分）误反样本的挑战。

    

    在视觉语言预训练中，跨模态对比学习面临着（部分）误反样本的挑战。本文从 mutual information 优化的角度研究了这个问题。我们理论上证明了在存在噪声的情况下，涉及到负样本的互信息也很重要。我们提出了一种跨模态相似性逐步细化的对比学习策略，以更加精确地优化图像/文本锚点与其负样本文本/图像之间的互信息。我们的方法在四个下游跨模态任务上表现出竞争力，并在理论指导下系统地平衡了（部分）误反样本的有益影响和有害影响。

    Cross-modal contrastive learning in vision language pretraining (VLP) faces the challenge of (partial) false negatives. In this paper, we study this problem from the perspective of Mutual Information (MI) optimization. It is common sense that InfoNCE loss used in contrastive learning will maximize the lower bound of MI between anchors and their positives, while we theoretically prove that MI involving negatives also matters when noises commonly exist. Guided by a more general lower bound form for optimization, we propose a contrastive learning strategy regulated by progressively refined cross-modal similarity, to more accurately optimize MI between an image/text anchor and its negative texts/images instead of improperly minimizing it. Our method performs competitively on four downstream cross-modal tasks and systematically balances the beneficial and harmful effects of (partial) false negative samples under theoretical guidance.
    

