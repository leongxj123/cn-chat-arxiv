# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target](https://arxiv.org/abs/2403.12116) | 本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。 |
| [^2] | [A Design Toolbox for the Development of Collaborative Distributed Machine Learning Systems.](http://arxiv.org/abs/2309.16584) | 我们开发了一个CDML设计工具箱，可以指导开发者设计满足用例要求的协作分布式机器学习系统。 |

# 详细

[^1]: 基于自定义生物启发目标的无监督端到端训练

    Unsupervised End-to-End Training with a Self-Defined Bio-Inspired Target

    [https://arxiv.org/abs/2403.12116](https://arxiv.org/abs/2403.12116)

    本研究提出了一种使用Winner-Take-All（WTA）选择性和生物启发的稳态机制相结合的“自定义目标”方法，旨在解决无监督学习方法在边缘AI硬件上的计算资源稀缺性问题。

    

    当前的无监督学习方法依赖于通过深度学习技术（如自监督学习）进行端到端训练，具有较高的计算需求，或者采用通过类似Hebbian学习的生物启发方法逐层训练，使用与监督学习不兼容的局部学习规则。为了解决这一挑战，在这项工作中，我们引入了一种使用网络最终层的胜者通吃（WTA）选择性的“自定义目标”，并通过生物启发的稳态机制进行正则化。

    arXiv:2403.12116v1 Announce Type: cross  Abstract: Current unsupervised learning methods depend on end-to-end training via deep learning techniques such as self-supervised learning, with high computational requirements, or employ layer-by-layer training using bio-inspired approaches like Hebbian learning, using local learning rules incompatible with supervised learning. Both approaches are problematic for edge AI hardware that relies on sparse computational resources and would strongly benefit from alternating between unsupervised and supervised learning phases - thus leveraging widely available unlabeled data from the environment as well as labeled training datasets. To solve this challenge, in this work, we introduce a 'self-defined target' that uses Winner-Take-All (WTA) selectivity at the network's final layer, complemented by regularization through biologically inspired homeostasis mechanism. This approach, framework-agnostic and compatible with both global (Backpropagation) and l
    
[^2]: 用于开发协作分布式机器学习系统的设计工具箱

    A Design Toolbox for the Development of Collaborative Distributed Machine Learning Systems. (arXiv:2309.16584v1 [cs.MA])

    [http://arxiv.org/abs/2309.16584](http://arxiv.org/abs/2309.16584)

    我们开发了一个CDML设计工具箱，可以指导开发者设计满足用例要求的协作分布式机器学习系统。

    

    为了在保护机器学习模型的机密性的同时利用来自多方的训练数据对模型进行充分训练，研究人员开发了各种协作分布式机器学习（CDML）系统设计，例如辅助学习、联邦学习和分裂学习。CDML系统设计展示了不同的特征，例如高度的代理人自治性、机器学习模型的机密性和容错性。面对不同特征的各种CDML系统设计，开发者很难有针对性地设计满足用例要求的CDML系统。然而，不合适的CDML系统设计可能导致CDML系统无法实现其预期目的。我们开发了一个CDML设计工具箱，可以指导CDML系统的开发。基于CDML设计工具箱，我们提出了具有不同关键特征的CDML系统典型，可以支持设计满足用例要求的CDML系统。

    To leverage training data for the sufficient training of ML models from multiple parties in a confidentiality-preserving way, various collaborative distributed machine learning (CDML) system designs have been developed, for example, to perform assisted learning, federated learning, and split learning. CDML system designs show different traits, for example, high agent autonomy, machine learning (ML) model confidentiality, and fault tolerance. Facing a wide variety of CDML system designs with different traits, it is difficult for developers to design CDML systems with traits that match use case requirements in a targeted way. However, inappropriate CDML system designs may result in CDML systems failing their envisioned purposes. We developed a CDML design toolbox that can guide the development of CDML systems. Based on the CDML design toolbox, we present CDML system archetypes with distinct key traits that can support the design of CDML systems to meet use case requirements.
    

