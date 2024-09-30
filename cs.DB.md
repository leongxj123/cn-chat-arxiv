# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Learning-based Declarative Privacy-Preserving Framework for Federated Data Management.](http://arxiv.org/abs/2401.12393) | 本论文提出了一个基于学习的声明性隐私保护框架，通过使用Differentially-Private Stochastic Gradient Descent（DP-SGD）算法训练的深度学习模型替代部分实际数据来回答查询，并允许用户指定要保护的私人信息。此框架还可以自动选择转换计划和超参数，并允许人工专家审核和调整隐私保护机制。 |

# 详细

[^1]: 基于学习的声明性隐私保护数据联邦管理框架

    A Learning-based Declarative Privacy-Preserving Framework for Federated Data Management. (arXiv:2401.12393v1 [cs.DB])

    [http://arxiv.org/abs/2401.12393](http://arxiv.org/abs/2401.12393)

    本论文提出了一个基于学习的声明性隐私保护框架，通过使用Differentially-Private Stochastic Gradient Descent（DP-SGD）算法训练的深度学习模型替代部分实际数据来回答查询，并允许用户指定要保护的私人信息。此框架还可以自动选择转换计划和超参数，并允许人工专家审核和调整隐私保护机制。

    

    在多个私有数据孤岛上进行联邦查询处理时，平衡隐私和准确性是一项具有挑战性的任务。在这项工作中，我们将演示一种自动化新兴隐私保护技术的端到端工作流，该技术使用使用差分隐私随机梯度下降（DP-SGD）算法训练的深度学习模型替换实际数据的部分来回答查询。我们提出的新颖声明性隐私保护工作流允许用户指定“要保护的私人信息”而不是“如何保护”。在底层，系统自动选择查询-模型转换计划以及超参数。同时，所提出的工作流还允许人工专家审核和调整选择的隐私保护机制，用于审计/合规和优化目的。

    It is challenging to balance the privacy and accuracy for federated query processing over multiple private data silos. In this work, we will demonstrate an end-to-end workflow for automating an emerging privacy-preserving technique that uses a deep learning model trained using the Differentially-Private Stochastic Gradient Descent (DP-SGD) algorithm to replace portions of actual data to answer a query. Our proposed novel declarative privacy-preserving workflow allows users to specify "what private information to protect" rather than "how to protect". Under the hood, the system automatically chooses query-model transformation plans as well as hyper-parameters. At the same time, the proposed workflow also allows human experts to review and tune the selected privacy-preserving mechanism for audit/compliance, and optimization purposes.
    

