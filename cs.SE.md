# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond](https://arxiv.org/abs/2403.14734) | 神经代码智能领域的调查系统回顾了50多种代表性模型和超过680项相关作品，突出了不同研究阶段的范式和技术转变。 |
| [^2] | [Turbulence: Systematically and Automatically Testing Instruction-Tuned Large Language Models for Code.](http://arxiv.org/abs/2312.14856) | 这项研究提出了一种通过新的基准测试Turbulence来系统评估针对代码生成的指令调整大型语言模型（LLMs）的正确性和鲁棒性的方法。通过构建一组问题模板，可以评估LLMs在解决相似编程问题时的准确性，并发现其代码生成能力的缺陷和异常情况。这项研究在五个LLMs上进行了实验。 |

# 详细

[^1]: 一项神经代码智能的调查：范式、进展与未来

    A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond

    [https://arxiv.org/abs/2403.14734](https://arxiv.org/abs/2403.14734)

    神经代码智能领域的调查系统回顾了50多种代表性模型和超过680项相关作品，突出了不同研究阶段的范式和技术转变。

    

    arXiv:2403.14734v1 公告类型: 跨领域 摘要: 神经代码智能--利用深度学习理解、生成和优化代码--在整个社会上具有巨大的潜力，可产生深远影响。作为自然语言和编程语言之间的桥梁，这一领域在过去几年引起了两个研究社区研究人员的极大关注。本调查系统地和按时间顺序回顾了代码智能方面的进展，包括50多种代表性模型及其变体、20多种任务类别以及超过680项相关作品。我们遵循历史进展，跟踪不同研究阶段的范式转变（例如，从使用循环神经网络对代码建模到大型语言模型时代）。同时，我们重点介绍了不同阶段涵盖的模型、任务和评估的主要技术转变。对于应用，我们

    arXiv:2403.14734v1 Announce Type: cross  Abstract: Neural Code Intelligence -- leveraging deep learning to understand, generate, and optimize code -- holds immense potential for transformative impacts on the whole society. Bridging the gap between Natural Language and Programming Language, this domain has drawn significant attention from researchers in both research communities over the past few years. This survey presents a systematic and chronological review of the advancements in code intelligence, encompassing over 50 representative models and their variants, more than 20 categories of tasks, and an extensive coverage of over 680 related works. We follow the historical progression to trace the paradigm shifts across different research phases (e.g., from modeling code with recurrent neural networks to the era of Large Language Models). Concurrently, we highlight the major technical transitions in models, tasks, and evaluations spanning through different stages. For applications, we 
    
[^2]: 系统化和自动化测试针对代码的指令调整大型语言模型的涡流方法

    Turbulence: Systematically and Automatically Testing Instruction-Tuned Large Language Models for Code. (arXiv:2312.14856v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2312.14856](http://arxiv.org/abs/2312.14856)

    这项研究提出了一种通过新的基准测试Turbulence来系统评估针对代码生成的指令调整大型语言模型（LLMs）的正确性和鲁棒性的方法。通过构建一组问题模板，可以评估LLMs在解决相似编程问题时的准确性，并发现其代码生成能力的缺陷和异常情况。这项研究在五个LLMs上进行了实验。

    

    我们提出了一种通过一个新的基准测试Turbulence，系统评估针对代码生成的指令调整大型语言模型（LLM）的正确性和鲁棒性的方法。Turbulence包含一组大量的自然语言“问题模板”，每个模板都是一个编程问题，参数化使得可以以多种不同形式提问。每个问题模板都有一个相关的“测试预测器”，用来判断LLM返回的代码解决方案是否正确。因此，通过一个问题模板，可以向LLM提问一个非常相似的编程问题“邻域”，并评估每个问题返回的结果的正确性。这允许识别LLM代码生成能力的差距，包括LLM在邻域中解决“几乎所有”问题但对特定参数实例化失败的“异常”。我们针对OpenAI、Co等五个LLM进行了实验。

    We present a method for systematically evaluating the correctness and robustness of instruction-tuned large language models (LLMs) for code generation via a new benchmark, Turbulence. Turbulence consists of a large set of natural language $\textit{question templates}$, each of which is a programming problem, parameterised so that it can be asked in many different forms. Each question template has an associated $\textit{test oracle}$ that judges whether a code solution returned by an LLM is correct. Thus, from a single question template, it is possible to ask an LLM a $\textit{neighbourhood}$ of very similar programming questions, and assess the correctness of the result returned for each question. This allows gaps in an LLM's code generation abilities to be identified, including $\textit{anomalies}$ where the LLM correctly solves $\textit{almost all}$ questions in a neighbourhood but fails for particular parameter instantiations. We present experiments against five LLMs from OpenAI, Co
    

