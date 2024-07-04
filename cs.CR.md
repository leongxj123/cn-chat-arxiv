# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Information Flow Control in Machine Learning through Modular Model Architecture.](http://arxiv.org/abs/2306.03235) | 本文提出了机器学习信息流控制的概念，并通过MoE架构实现了训练数据对模型输出的控制，从而提高了模型准确性。通过在推理时仅基于访问策略启用子集的专家，实现了对安全访问控制的支持。 |

# 详细

[^1]: 模块化模型架构中的机器学习信息流控制

    Information Flow Control in Machine Learning through Modular Model Architecture. (arXiv:2306.03235v1 [cs.LG])

    [http://arxiv.org/abs/2306.03235](http://arxiv.org/abs/2306.03235)

    本文提出了机器学习信息流控制的概念，并通过MoE架构实现了训练数据对模型输出的控制，从而提高了模型准确性。通过在推理时仅基于访问策略启用子集的专家，实现了对安全访问控制的支持。

    

    在当今的机器学习模型中，训练数据的任何部分都可以影响其输出。当访问控制只允许个人用户访问数据子集时，从训练数据到模型输出的信息流控制不足成为训练敏感数据模型的主要障碍。为了实现访问控制数据的安全机器学习，我们提出了机器学习信息流控制的概念，并基于混合专家（MoE）架构开发了一个安全Transformer型语言模型。通过限制来自每个安全领域的训练数据对单个专家模块的影响，并仅基于访问控制策略在推理时启用专家的子集，安全MoE架构控制了信息流。使用大型文本数据语料库进行的评估表明，所提出的MoE架构具有最小的性能开销（1.9%），并且可以显著提高模型准确性（最高可达37%），从而实现训练准确和安全分类器。

    In today's machine learning (ML) models, any part of the training data can affect its output. This lack of control for information flow from training data to model output is a major obstacle in training models on sensitive data when access control only allows individual users to access a subset of data. To enable secure machine learning for access controlled data, we propose the notion of information flow control for machine learning, and develop a secure Transformer-based language model based on the Mixture-of-Experts (MoE) architecture. The secure MoE architecture controls information flow by limiting the influence of training data from each security domain to a single expert module, and only enabling a subset of experts at inference time based on an access control policy. The evaluation using a large corpus of text data shows that the proposed MoE architecture has minimal (1.9%) performance overhead and can significantly improve model accuracy (up to 37%) by enabling training on acc
    

