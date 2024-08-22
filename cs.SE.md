# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quality and Trust in LLM-generated Code](https://arxiv.org/abs/2402.02047) | 本论文研究了机器学习生成代码的质量和信任问题，提出了校准的重要性，并探讨了如何确定模型生成代码的正确性。 |
| [^2] | [Copilot Refinement: Addressing Code Smells in Copilot-Generated Python Code.](http://arxiv.org/abs/2401.14176) | 本研究旨在探索Copilot生成的Python代码中的代码异味，评估Copilot修复这些问题的能力。结果表明，有8种Python代码异味可以在Copilot生成的代码中检测到。 |
| [^3] | [Pre-Training Representations of Binary Code Using Contrastive Learning.](http://arxiv.org/abs/2210.05102) | 提出了一种使用对比学习预训练二进制代码表示的方法，可以将源代码和注释信息纳入二进制代码的表示学习中，对于反向工程和计算机安全任务有重要意义。 |

# 详细

[^1]: 机器学习生成代码的质量和信任

    Quality and Trust in LLM-generated Code

    [https://arxiv.org/abs/2402.02047](https://arxiv.org/abs/2402.02047)

    本论文研究了机器学习生成代码的质量和信任问题，提出了校准的重要性，并探讨了如何确定模型生成代码的正确性。

    

    机器学习模型广泛应用，但常常会出错。用户需要可靠的指示，以确定给定模型的输出是否可信，从而可以做出理性决策是否使用该输出。例如，可以将输出与置信度相关联；如果置信度与正确性的可能性强相关，则称该模型为良好校准。在这种情况下，高置信度的输出可以安全接受，低置信度的输出可以拒绝。校准迄今主要在非生成性（例如分类）环境中进行研究，特别是在软件工程领域。然而，生成代码很容易出错：开发人员需要知道何时直接使用、经过仔细审查后使用或丢弃模型生成的代码，因此在生成环境中，校准非常重要。然而，生成代码的正确性概念并不简单，因此校准也是如此。

    Machine learning models are widely used but can also often be wrong. Users would benefit from a reliable indication of whether a given output from a given model should be trusted, so a rational decision can be made whether to use the output or not. For example, outputs can be associated with a confidence measure; if this confidence measure is strongly associated with likelihood of correctness, then the model is said to be well-calibrated. In this case, for example, high-confidence outputs could be safely accepted, and low-confidence outputs rejected.   Calibration has so far been studied in non-generative (e.g., classification) settings, especially in Software Engineering. However, generated code can quite often be wrong: Developers need to know when they should e.g., directly use, use after careful review, or discard model-generated code; thus Calibration is vital in generative settings. However, the notion of correctness of generated code is non-trivial, and thus so is Calibration. I
    
[^2]: Copilot细化：解决Copilot生成的Python代码中的代码异味

    Copilot Refinement: Addressing Code Smells in Copilot-Generated Python Code. (arXiv:2401.14176v1 [cs.SE])

    [http://arxiv.org/abs/2401.14176](http://arxiv.org/abs/2401.14176)

    本研究旨在探索Copilot生成的Python代码中的代码异味，评估Copilot修复这些问题的能力。结果表明，有8种Python代码异味可以在Copilot生成的代码中检测到。

    

    作为最流行的动态语言之一，Python在存在代码异味时可读性和可维护性会下降。大型语言模型的最新进展引发了对AI支持的代码生成和重构工具的日益关注。GitHub Copilot是其中一种被广泛使用的工具。Copilot Chat是在2023年9月发布的一种交互式工具，旨在为自然语言驱动的编码提供便利。然而，对于理解Copilot生成的Python代码中的代码异味以及Copilot修复其生成的代码异味的能力，人们并没有给予足够的关注。为此，我们构建了一个包含102个Copilot生成的Python代码中的代码异味的数据集。我们的目标是首先探索Copilot生成的Python代码中代码异味的发生情况，然后评估Copilot在使用不同提示修复这些代码异味时的有效性。结果显示，10种Python代码异味中有8种可以在Copilot生成的代码中检测到。

    As one of the most popular dynamic languages, Python experiences a decrease in readability and maintainability when code smells are present. Recent advancements in Large Language Models have sparked growing interest in AI-enabled tools for both code generation and refactoring. GitHub Copilot is one such tool that has gained widespread usage. Copilot Chat, released on September 2023, functions as an interactive tool aims at facilitating natural language-powered coding. However, limited attention has been given to understanding code smells in Copilot-generated Python code and Copilot's ability to fix the code smells it generates. To this end, we built a dataset comprising 102 code smells in Copilot-generated Python code. Our aim is to first explore the occurrence of code smells in Copilot-generated Python code and then evaluate the effectiveness of Copilot in fixing these code smells employing different prompts. The results show that 8 out of 10 types of Python smells can be detected in 
    
[^3]: 使用对比学习预训练二进制代码表示

    Pre-Training Representations of Binary Code Using Contrastive Learning. (arXiv:2210.05102v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2210.05102](http://arxiv.org/abs/2210.05102)

    提出了一种使用对比学习预训练二进制代码表示的方法，可以将源代码和注释信息纳入二进制代码的表示学习中，对于反向工程和计算机安全任务有重要意义。

    

    编译后的软件以可执行的二进制代码形式交付。开发人员编写源代码来表达软件的语义，但编译器将其转换为CPU可以直接执行的二进制格式。因此，二进制代码分析对于反向工程和计算机安全任务等没有源代码的应用程序至关重要。然而，与包含丰富语义信息的源代码和自然语言不同，二进制代码通常难以理解和分析。虽然现有的工作使用AI模型辅助源代码分析，但很少有研究考虑二进制代码。在本文中，我们提出了一种将源代码和注释信息纳入二进制代码进行表示学习的对比学习模型，称为COMBO。具体而言，我们在COMBO中提出了三个组件：（1）用于冷启动预训练的主要对比学习方法，（2）用于将源代码和注释信息插入到二进制代码中的单纯插值方法。

    Compiled software is delivered as executable binary code. Developers write source code to express the software semantics, but the compiler converts it to a binary format that the CPU can directly execute. Therefore, binary code analysis is critical to applications in reverse engineering and computer security tasks where source code is not available. However, unlike source code and natural language that contain rich semantic information, binary code is typically difficult for human engineers to understand and analyze. While existing work uses AI models to assist source code analysis, few studies have considered binary code. In this paper, we propose a COntrastive learning Model for Binary cOde Analysis, or COMBO, that incorporates source code and comment information into binary code during representation learning. Specifically, we present three components in COMBO: (1) a primary contrastive learning method for cold-start pre-training, (2) a simplex interpolation method to incorporate so
    

