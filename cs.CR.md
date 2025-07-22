# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Defending Against Unforeseen Failure Modes with Latent Adversarial Training](https://arxiv.org/abs/2403.05030) | 本研究利用潜在对抗训练（LAT）来防御AI系统中未预见的故障模式，通过利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示，有效清除了恶意软件和对抗性攻击。 |
| [^2] | [zkFL: Zero-Knowledge Proof-based Gradient Aggregation for Federated Learning.](http://arxiv.org/abs/2310.02554) | zkFL是一种基于零知识证明的联邦学习梯度聚合方法，通过提供每轮的证明来解决协调者恶意行为的问题。 |

# 详细

[^1]: 利用潜在对抗训练防御未预见的故障模式

    Defending Against Unforeseen Failure Modes with Latent Adversarial Training

    [https://arxiv.org/abs/2403.05030](https://arxiv.org/abs/2403.05030)

    本研究利用潜在对抗训练（LAT）来防御AI系统中未预见的故障模式，通过利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示，有效清除了恶意软件和对抗性攻击。

    

    人工智能系统有时在部署后会展示出有害的意外行为。尽管开发人员进行了大量诊断和调试，这种情况经常发生。由于攻击面非常广泛，从模型中减少风险具有挑战性。耗尽地搜索可能导致模型失败的输入是不可行的。红队和对抗训练（AT）通常用于使人工智能系统更加健壮。然而，它们并不足以避免许多与对抗训练不同的真实世界故障模式。在这项工作中，我们利用潜在对抗训练（LAT）来防御漏洞，而无需生成引发这些漏洞的输入。LAT利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示。我们使用LAT来清除恶意软件并防御针对保留类别的对抗性攻击。我们展示在图像分类、文本分类

    arXiv:2403.05030v1 Announce Type: cross  Abstract: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classifi
    
[^2]: zkFL: 基于零知识证明的联邦学习梯度聚合

    zkFL: Zero-Knowledge Proof-based Gradient Aggregation for Federated Learning. (arXiv:2310.02554v1 [cs.AI])

    [http://arxiv.org/abs/2310.02554](http://arxiv.org/abs/2310.02554)

    zkFL是一种基于零知识证明的联邦学习梯度聚合方法，通过提供每轮的证明来解决协调者恶意行为的问题。

    

    联邦学习是一种机器学习范式，使多个分散的客户端在中央协调者的组织下共同训练一个模型。传统的联邦学习解决方案依赖于对中央协调者的信任，它以公平诚实的方式形成客户端的群体。然而，在现实中，恶意的协调者可能会放弃并替换客户端的训练模型，或者发动虚假客户端的肆意攻击。这种恶意行为让协调者在联邦学习环境中拥有更多控制客户端和决定最终训练结果的权力。本文介绍了zkFL，它利用零知识证明(ZKPs)来解决训练模型聚合过程中的恶意协调者问题。为了保证正确的聚合结果，协调者需要每轮提供一个证明。这个证明可以向客户端证明协调者忠实执行预期行为。为了进一步保护客户端隐私和数据安全，我们还引入了差分隐私机制，并对zkFL进行了实验评估。

    Federated Learning (FL) is a machine learning paradigm, which enables multiple and decentralized clients to collaboratively train a model under the orchestration of a central aggregator. Traditional FL solutions rely on the trust assumption of the centralized aggregator, which forms cohorts of clients in a fair and honest manner. However, a malicious aggregator, in reality, could abandon and replace the client's training models, or launch Sybil attacks to insert fake clients. Such malicious behaviors give the aggregator more power to control clients in the FL setting and determine the final training results. In this work, we introduce zkFL, which leverages zero-knowledge proofs (ZKPs) to tackle the issue of a malicious aggregator during the training model aggregation process. To guarantee the correct aggregation results, the aggregator needs to provide a proof per round. The proof can demonstrate to the clients that the aggregator executes the intended behavior faithfully. To further r
    

