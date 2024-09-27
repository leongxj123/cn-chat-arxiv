# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs](https://arxiv.org/abs/2403.15676) | 该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。 |
| [^2] | [Threats, Attacks, and Defenses in Machine Unlearning: A Survey](https://arxiv.org/abs/2403.13682) | 机器遗忘（MU）通过知识去除过程来解决训练数据相关的人工智能治理问题，提高了AI系统的安全和负责任使用。 |
| [^3] | [Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning.](http://arxiv.org/abs/2304.10783) | 本文提出了一种灵活的联邦学习模型毒化攻击策略，既可以实现拒绝服务(Dos)目标，也可以精确控制全局准确性，具有高效和隐形的特点。 |

# 详细

[^1]: AC4：用于ZKP中电路约束的代数计算检查器

    AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs

    [https://arxiv.org/abs/2403.15676](https://arxiv.org/abs/2403.15676)

    该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。

    

    ZKP系统已经引起了人们的关注，在当代密码学中发挥着基础性作用。 Zk-SNARK协议主导了ZKP的使用，通常通过算术电路编程范式实现。然而，欠约束或过约束的电路可能导致错误。 欠约束的电路指的是缺乏必要约束的电路，导致电路中出现意外解决方案，并导致验证者接受错误见证。 过约束的电路是指约束过度的电路，导致电路缺乏必要的解决方案，并导致验证者接受没有见证，使电路毫无意义。 本文介绍了一种新方法，用于找出ZKP电路中两种不同类型的错误。 该方法涉及将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统。

    arXiv:2403.15676v1 Announce Type: cross  Abstract: ZKP systems have surged attention and held a fundamental role in contemporary cryptography. Zk-SNARK protocols dominate the ZKP usage, often implemented through arithmetic circuit programming paradigm. However, underconstrained or overconstrained circuits may lead to bugs. Underconstrained circuits refer to circuits that lack the necessary constraints, resulting in unexpected solutions in the circuit and causing the verifier to accept a bogus witness. Overconstrained circuits refer to circuits that are constrained excessively, resulting in the circuit lacking necessary solutions and causing the verifier to accept no witness, rendering the circuit meaningless. This paper introduces a novel approach for pinpointing two distinct types of bugs in ZKP circuits. The method involves encoding the arithmetic circuit constraints to polynomial equation systems and solving polynomial equation systems over a finite field by algebraic computation. T
    
[^2]: 机器学习中的威胁、攻击和防御：一项调查

    Threats, Attacks, and Defenses in Machine Unlearning: A Survey

    [https://arxiv.org/abs/2403.13682](https://arxiv.org/abs/2403.13682)

    机器遗忘（MU）通过知识去除过程来解决训练数据相关的人工智能治理问题，提高了AI系统的安全和负责任使用。

    

    机器遗忘（MU）最近引起了相当大的关注，因为它有潜力通过从训练的机器学习模型中消除特定数据的影响来实现安全人工智能。这个被称为知识去除的过程解决了与训练数据相关的人工智能治理问题，如数据质量、敏感性、版权限制和过时性。这种能力对于确保遵守诸如被遗忘权等隐私法规也至关重要。此外，有效的知识去除有助于减轻有害结果的风险，防范偏见、误导和未经授权的数据利用，从而增强了AI系统的安全和负责任使用。已经开展了设计高效的遗忘方法的工作，通过研究MU服务以与现有的机器学习作为服务集成，使用户能够提交请求从训练语料库中删除特定数据。

    arXiv:2403.13682v2 Announce Type: replace-cross  Abstract: Machine Unlearning (MU) has gained considerable attention recently for its potential to achieve Safe AI by removing the influence of specific data from trained machine learning models. This process, known as knowledge removal, addresses AI governance concerns of training data such as quality, sensitivity, copyright restrictions, and obsolescence. This capability is also crucial for ensuring compliance with privacy regulations such as the Right To Be Forgotten. Furthermore, effective knowledge removal mitigates the risk of harmful outcomes, safeguarding against biases, misinformation, and unauthorized data exploitation, thereby enhancing the safe and responsible use of AI systems. Efforts have been made to design efficient unlearning approaches, with MU services being examined for integration with existing machine learning as a service, allowing users to submit requests to remove specific data from the training corpus. However, 
    
[^3]: 拒绝服务或细粒度控制：面向联邦学习的灵活模型毒化攻击

    Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning. (arXiv:2304.10783v1 [cs.LG])

    [http://arxiv.org/abs/2304.10783](http://arxiv.org/abs/2304.10783)

    本文提出了一种灵活的联邦学习模型毒化攻击策略，既可以实现拒绝服务(Dos)目标，也可以精确控制全局准确性，具有高效和隐形的特点。

    

    联邦学习容易受到毒化攻击，敌对方会破坏全局聚合结果并造成拒绝服务。本文提出了一种灵活模型毒化攻击(FMPA)，旨在实现多功能攻击目标。本文考虑如下实际情景：敌对方没有关于FL系统的额外信息（例如，聚合规则或良性设备上的更新）。FMPA利用全局历史信息构建估计器，将下一轮全局模型预测为良性参考模型，并微调参考模型以获得所需的精度低和扰动小的毒化模型。FMPA不仅可以达到DoS的目标，还可以自然地扩展到启动细粒度可控攻击，从而精确降低全局准确性。本文进一步探索了FMPA在几种FL场景下的攻击性能，包括二元分类和图像分类，在不同的攻击目标和攻击知识水平下。实验结果表明，FMPA可以有效而高效地实现所需的攻击目标，同时保持隐形和不可感知。

    Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a Flexible Model Poisoning Attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed wi
    

