# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Importance Guided Data Augmentation for Neural-Based Code Understanding](https://arxiv.org/abs/2402.15769) | 引入了一个通用数据增强框架GenCode，通过重要性指标选择生成的代码作为训练数据，以增强代码理解模型的训练。 |

# 详细

[^1]: 重点引导的数据增强用于基于神经网络的代码理解

    Importance Guided Data Augmentation for Neural-Based Code Understanding

    [https://arxiv.org/abs/2402.15769](https://arxiv.org/abs/2402.15769)

    引入了一个通用数据增强框架GenCode，通过重要性指标选择生成的代码作为训练数据，以增强代码理解模型的训练。

    

    arXiv:2402.15769v1 类型：交叉 摘要：预训练的代码模型开启了代码智能时代。最近许多模型都表现出色。然而，在代码学习领域，一个重要问题是自动进行代码数据增强，以帮助开发者准备训练数据，这方面的研究尚不足。本文介绍了一个通用的数据增强框架GenCode，用于增强代码理解模型的训练。GenCode遵循一种生成和选择的范式来准备有用的训练代码。具体来说，它使用代码转换技术首先生成新的代码候选，然后通过重要性指标选择重要的代码作为训练数据。为了评估GenCode与通用重要性指标（损失值）的有效性，我们在四个代码理解任务（如代码克隆检测）和三个预训练代码模型（如CodeT5）上进行实验。与最先进的代码增强技术相比，

    arXiv:2402.15769v1 Announce Type: cross  Abstract: Pre-trained code models lead the era of code intelligence. Many models have been designed with impressive performance recently. However, one important problem, data augmentation for code data that automatically helps developers prepare training data lacks study in the field of code learning. In this paper, we introduce a general data augmentation framework, GenCode, to enhance the training of code understanding models. GenCode follows a generation-and-selection paradigm to prepare useful training codes. Specifically, it uses code transformation techniques to generate new code candidates first and then selects important ones as the training data by importance metrics. To evaluate the effectiveness of GenCode with a general importance metric -- loss value, we conduct experiments on four code understanding tasks (e.g., code clone detection) and three pre-trained code models (e.g., CodeT5). Compared to the state-of-the-art (SOTA) code augm
    

