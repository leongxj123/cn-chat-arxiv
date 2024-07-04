# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluating Large Language Models with Runtime Behavior of Program Execution](https://arxiv.org/abs/2403.16437) | 本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。 |
| [^2] | [Deep Configuration Performance Learning: A Systematic Survey and Taxonomy](https://arxiv.org/abs/2403.03322) | 性能是可配置软件系统行为的关键属性，本文针对深度学习在可配置软件性能学习方面进行了全面的调查与分类研究。 |

# 详细

[^1]: 使用程序执行运行时行为评估大型语言模型

    Evaluating Large Language Models with Runtime Behavior of Program Execution

    [https://arxiv.org/abs/2403.16437](https://arxiv.org/abs/2403.16437)

    本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。

    

    大型代码语言模型（即代码LLMs）展示了强大的代码理解和生成能力。为了评估代码LLMs在各个方面的能力，已经提出了许多基准（如HumanEval和ClassEval）。代码推理是代码LLMs最重要的能力之一，但现有的代码推理基准不足。通常，它们重点预测程序的输入和输出，忽略了程序执行过程中的中间行为评估，以及逻辑一致性（例如，如果执行路径预测错误，则模型不应该给出正确的输出）在执行推理时。为了解决这些问题，本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。我们利用现有的代码基准，并将它们适应到我们的框架中的新基准中。

    arXiv:2403.16437v1 Announce Type: cross  Abstract: Large language models for code (i.e., code LLMs) have shown strong code understanding and generation capabilities. To evaluate the capabilities of code LLMs in various aspects, many benchmarks have been proposed (e.g., HumanEval and ClassEval). Code reasoning is one of the most essential abilities of code LLMs, but existing benchmarks for code reasoning are not sufficient. Typically, they focus on predicting the input and output of a program, ignoring the evaluation of the intermediate behavior during program execution, as well as the logical consistency (e.g., the model should not give the correct output if the prediction of execution path is wrong) when performing the reasoning. To address these problems, in this paper, we propose a framework, namely REval, for evaluating code reasoning abilities and consistency of code LLMs with program execution. We utilize existing code benchmarks and adapt them to new benchmarks within our framew
    
[^2]: 深度配置性能学习：一项系统性调查与分类

    Deep Configuration Performance Learning: A Systematic Survey and Taxonomy

    [https://arxiv.org/abs/2403.03322](https://arxiv.org/abs/2403.03322)

    性能是可配置软件系统行为的关键属性，本文针对深度学习在可配置软件性能学习方面进行了全面的调查与分类研究。

    

    性能可以说是反映可配置软件系统行为的最关键属性。然而，随着现代软件规模和复杂性不断增加，对各种配置如何影响性能进行建模和预测成为软件维护中的主要挑战之一。因此，性能通常是在没有对软件系统有透彻了解的情况下建模的，主要依赖数据，这正好符合深度学习的目的。在这篇论文中，我们专注于深度学习在可配置软件性能学习方面进行了全面的回顾，涵盖了948篇来自六个索引服务的论文，基于此提取并分析了85篇主要论文。我们的结果总结了配置数据如何准备，深度配置性能学习模型如何构建，以及该模型如何进行评估等关键主题和统计信息。

    arXiv:2403.03322v1 Announce Type: cross  Abstract: Performance is arguably the most crucial attribute that reflects the behavior of a configurable software system. However, given the increasing scale and complexity of modern software, modeling and predicting how various configurations can impact performance becomes one of the major challenges in software maintenance. As such, performance is often modeled without having a thorough knowledge of the software system, but relying mainly on data, which fits precisely with the purpose of deep learning.   In this paper, we conduct a comprehensive review exclusively on the topic of deep learning for performance learning of configurable software, covering 948 searched papers spanning six indexing services, based on which 85 primary papers were extracted and analyzed. Our results summarize the key topics and statistics on how the configuration data is prepared; how the deep configuration performance learning model is built; how the model is evalu
    

