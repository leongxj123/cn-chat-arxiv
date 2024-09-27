# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning.](http://arxiv.org/abs/2304.10783) | 本文提出了一种灵活的联邦学习模型毒化攻击策略，既可以实现拒绝服务(Dos)目标，也可以精确控制全局准确性，具有高效和隐形的特点。 |

# 详细

[^1]: 拒绝服务或细粒度控制：面向联邦学习的灵活模型毒化攻击

    Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning. (arXiv:2304.10783v1 [cs.LG])

    [http://arxiv.org/abs/2304.10783](http://arxiv.org/abs/2304.10783)

    本文提出了一种灵活的联邦学习模型毒化攻击策略，既可以实现拒绝服务(Dos)目标，也可以精确控制全局准确性，具有高效和隐形的特点。

    

    联邦学习容易受到毒化攻击，敌对方会破坏全局聚合结果并造成拒绝服务。本文提出了一种灵活模型毒化攻击(FMPA)，旨在实现多功能攻击目标。本文考虑如下实际情景：敌对方没有关于FL系统的额外信息（例如，聚合规则或良性设备上的更新）。FMPA利用全局历史信息构建估计器，将下一轮全局模型预测为良性参考模型，并微调参考模型以获得所需的精度低和扰动小的毒化模型。FMPA不仅可以达到DoS的目标，还可以自然地扩展到启动细粒度可控攻击，从而精确降低全局准确性。本文进一步探索了FMPA在几种FL场景下的攻击性能，包括二元分类和图像分类，在不同的攻击目标和攻击知识水平下。实验结果表明，FMPA可以有效而高效地实现所需的攻击目标，同时保持隐形和不可感知。

    Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a Flexible Model Poisoning Attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed wi
    

