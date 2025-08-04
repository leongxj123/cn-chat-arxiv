# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Model Stock: All we need is just a few fine-tuned models](https://arxiv.org/abs/2403.19522) | 本文提出了一种高效的微调方法，只使用少量模型就能获得优越的性能，通过权重空间和层次加权平均技术超越了现有的模型方法。 |
| [^2] | [Gradient Leakage Defense with Key-Lock Module for Federated Learning.](http://arxiv.org/abs/2305.04095) | 本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。 |

# 详细

[^1]: 模型库：我们只需要几个经过良好调整的模型

    Model Stock: All we need is just a few fine-tuned models

    [https://arxiv.org/abs/2403.19522](https://arxiv.org/abs/2403.19522)

    本文提出了一种高效的微调方法，只使用少量模型就能获得优越的性能，通过权重空间和层次加权平均技术超越了现有的模型方法。

    

    本文介绍了一种高效的大型预训练模型微调方法，提供强大的内分布（ID）和外分布（OOD）性能。与需要大量微调模型进行平均的传统做法不同，我们的方法使用更少的模型来获得最终权重，同时产生更高的准确性。从微调权重的权重空间中汲取关键见解，我们揭示了性能和接近权重空间中心的强连接。基于此，我们引入一种方法，通过仅使用两个微调模型来近似中心接近的权重，可在训练期间或之后应用。我们的创新的逐层权重平均技术超越了Model Soup等最先进的模型方法，仅利用两个微调模型。这种策略可以被称为模型库，突出了它依赖于选择少量模型来进行综合的特点。

    arXiv:2403.19522v1 Announce Type: new  Abstract: This paper introduces an efficient fine-tuning method for large pre-trained models, offering strong in-distribution (ID) and out-of-distribution (OOD) performance. Breaking away from traditional practices that need a multitude of fine-tuned models for averaging, our approach employs significantly fewer models to achieve final weights yet yield superior accuracy. Drawing from key insights in the weight space of fine-tuned weights, we uncover a strong link between the performance and proximity to the center of weight space. Based on this, we introduce a method that approximates a center-close weight using only two fine-tuned models, applicable during or after training. Our innovative layer-wise weight averaging technique surpasses state-of-the-art model methods such as Model Soup, utilizing only two fine-tuned models. This strategy can be aptly coined Model Stock, highlighting its reliance on selecting a minimal number of models to draw a 
    
[^2]: 基于密钥锁模块的联邦学习梯度泄露防御

    Gradient Leakage Defense with Key-Lock Module for Federated Learning. (arXiv:2305.04095v1 [cs.LG])

    [http://arxiv.org/abs/2305.04095](http://arxiv.org/abs/2305.04095)

    本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。

    

    联邦学习是一种广泛采用的隐私保护机器学习方法，其中私有数据保持本地，允许安全计算和本地模型梯度与第三方参数服务器之间的交换。然而，最近的研究发现，通过共享的梯度可能会危及隐私并恢复敏感信息。本研究提供了详细的分析和对梯度泄漏问题的新视角。这些理论工作导致了一种新的梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构。只有锁定的梯度被传输到参数服务器进行全局模型聚合。我们提出的学习方法对梯度泄露攻击具有抵抗力，并且所设计和训练的密钥锁模块可以确保，没有密钥锁模块的私有信息：a) 无法从共享的梯度中重建私有训练数据。

    Federated Learning (FL) is a widely adopted privacy-preserving machine learning approach where private data remains local, enabling secure computations and the exchange of local model gradients between local clients and third-party parameter servers. However, recent findings reveal that privacy may be compromised and sensitive information potentially recovered from shared gradients. In this study, we offer detailed analysis and a novel perspective on understanding the gradient leakage problem. These theoretical works lead to a new gradient leakage defense technique that secures arbitrary model architectures using a private key-lock module. Only the locked gradient is transmitted to the parameter server for global model aggregation. Our proposed learning method is resistant to gradient leakage attacks, and the key-lock module is designed and trained to ensure that, without the private information of the key-lock module: a) reconstructing private training data from the shared gradient is
    

