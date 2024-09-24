# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluating Large Language Models with Runtime Behavior of Program Execution](https://arxiv.org/abs/2403.16437) | 本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。 |
| [^2] | [Automated Approaches to Detect Self-Admitted Technical Debt: A Systematic Literature Review](https://arxiv.org/abs/2312.15020) | 论文提出了一种特征提取技术和ML/DL算法分类法，旨在比较和基准测试其在技术债务检测中的表现。 |

# 详细

[^1]: 使用程序执行运行时行为评估大型语言模型

    Evaluating Large Language Models with Runtime Behavior of Program Execution

    [https://arxiv.org/abs/2403.16437](https://arxiv.org/abs/2403.16437)

    本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。

    

    大型代码语言模型（即代码LLMs）展示了强大的代码理解和生成能力。为了评估代码LLMs在各个方面的能力，已经提出了许多基准（如HumanEval和ClassEval）。代码推理是代码LLMs最重要的能力之一，但现有的代码推理基准不足。通常，它们重点预测程序的输入和输出，忽略了程序执行过程中的中间行为评估，以及逻辑一致性（例如，如果执行路径预测错误，则模型不应该给出正确的输出）在执行推理时。为了解决这些问题，本文提出了一个名为REval的框架，用于评估代码LLMs的代码推理能力以及与程序执行的一致性。我们利用现有的代码基准，并将它们适应到我们的框架中的新基准中。

    arXiv:2403.16437v1 Announce Type: cross  Abstract: Large language models for code (i.e., code LLMs) have shown strong code understanding and generation capabilities. To evaluate the capabilities of code LLMs in various aspects, many benchmarks have been proposed (e.g., HumanEval and ClassEval). Code reasoning is one of the most essential abilities of code LLMs, but existing benchmarks for code reasoning are not sufficient. Typically, they focus on predicting the input and output of a program, ignoring the evaluation of the intermediate behavior during program execution, as well as the logical consistency (e.g., the model should not give the correct output if the prediction of execution path is wrong) when performing the reasoning. To address these problems, in this paper, we propose a framework, namely REval, for evaluating code reasoning abilities and consistency of code LLMs with program execution. We utilize existing code benchmarks and adapt them to new benchmarks within our framew
    
[^2]: 自动化方法检测自我承认的技术债务：系统文献综述

    Automated Approaches to Detect Self-Admitted Technical Debt: A Systematic Literature Review

    [https://arxiv.org/abs/2312.15020](https://arxiv.org/abs/2312.15020)

    论文提出了一种特征提取技术和ML/DL算法分类法，旨在比较和基准测试其在技术债务检测中的表现。

    

    技术债务是软件开发中普遍存在的问题，通常源自开发过程中做出的权衡，在影响软件可维护性和阻碍未来开发工作方面起到作用。自我承认的技术债务（SATD）指的是开发人员明确承认代码库中存在的代码质量或设计缺陷。自动检测SATD已经成为一个重要的研究领域，旨在帮助开发人员高效地识别和解决技术债务。然而，文献中广泛采用的NLP特征提取方法和算法种类多样化常常阻碍研究人员试图提高其性能。基于此，本系统文献综述提出了一种特征提取技术和ML/DL算法分类法，其目的是比较和基准测试所考察研究中它们的性能。我们选择......

    arXiv:2312.15020v2 Announce Type: replace-cross  Abstract: Technical debt is a pervasive issue in software development, often arising from trade-offs made during development, which can impede software maintainability and hinder future development efforts. Self-admitted technical debt (SATD) refers to instances where developers explicitly acknowledge suboptimal code quality or design flaws in the codebase. Automated detection of SATD has emerged as a critical area of research, aiming to assist developers in identifying and addressing technical debt efficiently. However, the enormous variety of feature extraction approaches of NLP and algorithms employed in the literature often hinder researchers from trying to improve their performance. In light of this, this systematic literature review proposes a taxonomy of feature extraction techniques and ML/DL algorithms used in technical debt detection: its objective is to compare and benchmark their performance in the examined studies. We select
    

