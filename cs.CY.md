# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Low-Resource Knowledge Tracing Tasks by Supervised Pre-training and Importance Mechanism Fine-tuning](https://arxiv.org/abs/2403.06725) | 本文提出了名为LoReKT的低资源知识追踪框架，通过监督预训练和微调重要性机制，旨在从丰富资源的KT数据集中学习可转移的参数和表示来改进低资源知识追踪任务。 |

# 详细

[^1]: 通过监督预训练和重要性机制微调改进低资源知识追踪任务

    Improving Low-Resource Knowledge Tracing Tasks by Supervised Pre-training and Importance Mechanism Fine-tuning

    [https://arxiv.org/abs/2403.06725](https://arxiv.org/abs/2403.06725)

    本文提出了名为LoReKT的低资源知识追踪框架，通过监督预训练和微调重要性机制，旨在从丰富资源的KT数据集中学习可转移的参数和表示来改进低资源知识追踪任务。

    

    知识追踪（KT）旨在基于学生的历史互动来估计他们的知识掌握程度。最近，基于深度学习的KT（DLKT）方法在KT任务中取得了令人印象深刻的表现。然而，由于各种原因，如预算限制和隐私问题，许多实际场景中观察到的互动非常有限，即低资源KT数据集。直接在低资源KT数据集上训练DLKT模型可能会导致过拟合，并且很难选择适当的深度神经架构。因此，在本文中，我们提出了一个名为LoReKT的低资源KT框架来应对上述挑战。受盛行的“预训练和微调”范式的启发，我们旨在在预训练阶段从丰富资源的KT数据集中学习可转移的参数和表示。

    arXiv:2403.06725v1 Announce Type: cross  Abstract: Knowledge tracing (KT) aims to estimate student's knowledge mastery based on their historical interactions. Recently, the deep learning based KT (DLKT) approaches have achieved impressive performance in the KT task. These DLKT models heavily rely on the large number of available student interactions. However, due to various reasons such as budget constraints and privacy concerns, observed interactions are very limited in many real-world scenarios, a.k.a, low-resource KT datasets. Directly training a DLKT model on a low-resource KT dataset may lead to overfitting and it is difficult to choose the appropriate deep neural architecture. Therefore, in this paper, we propose a low-resource KT framework called LoReKT to address above challenges. Inspired by the prevalent "pre-training and fine-tuning" paradigm, we aim to learn transferable parameters and representations from rich-resource KT datasets during the pre-training stage and subseque
    

