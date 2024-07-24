# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An exploratory study on automatic identification of assumptions in the development of deep learning frameworks.](http://arxiv.org/abs/2401.03653) | 本研究以构建一个新的最大假设数据集为基础，针对深度学习框架开发中手动识别假设的问题进行了探索性研究。在该研究中，我们发现手动识别假设的成本高，因此探讨了使用传统机器学习模型和流行的深度学习模型来识别假设的性能。 |

# 详细

[^1]: 关于深度学习框架开发中自动识别假设的探索性研究

    An exploratory study on automatic identification of assumptions in the development of deep learning frameworks. (arXiv:2401.03653v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2401.03653](http://arxiv.org/abs/2401.03653)

    本研究以构建一个新的最大假设数据集为基础，针对深度学习框架开发中手动识别假设的问题进行了探索性研究。在该研究中，我们发现手动识别假设的成本高，因此探讨了使用传统机器学习模型和流行的深度学习模型来识别假设的性能。

    

    利益相关方在深度学习框架开发中经常做出假设。这些假设涉及各种软件构件（例如需求、设计决策和技术债务），可能会被证明无效，从而导致系统故障。现有的假设管理方法和工具通常依赖于手动识别假设。然而，假设分散在深度学习框架开发的各种源头（例如代码注释、提交、拉取请求和问题）中，手动识别假设成本较高（例如时间和资源消耗）。为了解决深度学习框架开发中手动识别假设的问题，我们构建了一个新的并且最大的假设数据集（称为AssuEval），该数据集收集自GitHub上的TensorFlow和Keras代码库；我们探讨了七个传统的机器学习模型（例如支持向量机、分类回归树）和一个流行的深度学习模型的性能。

    Stakeholders constantly make assumptions in the development of deep learning (DL) frameworks. These assumptions are related to various types of software artifacts (e.g., requirements, design decisions, and technical debt) and can turn out to be invalid, leading to system failures. Existing approaches and tools for assumption management usually depend on manual identification of assumptions. However, assumptions are scattered in various sources (e.g., code comments, commits, pull requests, and issues) of DL framework development, and manually identifying assumptions has high costs (e.g., time and resources). To overcome the issues of manually identifying assumptions in DL framework development, we constructed a new and largest dataset (i.e., AssuEval) of assumptions collected from the TensorFlow and Keras repositories on GitHub; explored the performance of seven traditional machine learning models (e.g., Support Vector Machine, Classification and Regression Trees), a popular DL model (i
    

