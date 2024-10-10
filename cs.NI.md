# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [netFound: Foundation Model for Network Security.](http://arxiv.org/abs/2310.17025) | netFound是一个基于自我监督算法的基础模型，用于网络安全领域。该模型通过预训练捕捉网络流量的层次化和多模态属性，并能够在质量低、有限和嘈杂的数据情况下进行微调。 |

# 详细

[^1]: netFound: 网络安全的基础模型

    netFound: Foundation Model for Network Security. (arXiv:2310.17025v1 [cs.NI])

    [http://arxiv.org/abs/2310.17025](http://arxiv.org/abs/2310.17025)

    netFound是一个基于自我监督算法的基础模型，用于网络安全领域。该模型通过预训练捕捉网络流量的层次化和多模态属性，并能够在质量低、有限和嘈杂的数据情况下进行微调。

    

    在网络安全的机器学习领域，传统工作流依赖于高质量标记数据和手动特征工程，但有限的数据集和人类专业知识阻碍了特征选择，导致模型难以捕捉关键关系和有效泛化。受到GPT-4和Vision Transformers等机器学习应用领域的最新进展的启发，我们开发了netFound，一个网络安全的基础模型。该模型利用自我监督算法对现有的未标记网络数据包进行预训练。netFound的设计融合了网络流量的层次化和多模态属性，有效捕捉了隐藏的网络上下文，包括应用逻辑、通信协议和网络条件。有了这个预训练基础，即使处理质量低、有限和嘈杂的标记数据，我们也可以对netFound进行微调，适用于各种下游任务。我们的实验证明了netFound的效果。

    In ML for network security, traditional workflows rely on high-quality labeled data and manual feature engineering, but limited datasets and human expertise hinder feature selection, leading to models struggling to capture crucial relationships and generalize effectively. Inspired by recent advancements in ML application domains like GPT-4 and Vision Transformers, we have developed netFound, a foundational model for network security. This model undergoes pre-training using self-supervised algorithms applied to readily available unlabeled network packet traces. netFound's design incorporates hierarchical and multi-modal attributes of network traffic, effectively capturing hidden networking contexts, including application logic, communication protocols, and network conditions.  With this pre-trained foundation in place, we can fine-tune netFound for a wide array of downstream tasks, even when dealing with low-quality, limited, and noisy labeled data. Our experiments demonstrate netFound'
    

