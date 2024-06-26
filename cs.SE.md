# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Effort and Size Estimation in Software Projects with Large Language Model-based Intelligent Interfaces](https://arxiv.org/abs/2402.07158) | 本文提出了一种基于大型语言模型的智能界面在软件项目中进行工作量和规模估计的方法，并通过比较传统方法，探讨了如何通过增强基于自然语言的问题规范来实现开发工作量的准确估计。 |
| [^2] | [ChIRAAG: ChatGPT Informed Rapid and Automated Assertion Generation](https://arxiv.org/abs/2402.00093) | 本研究设计了一个基于大语言模型的流水线，通过自然语言规格生成英语、线性时态逻辑和SVA断言，并成功减少了断言错误率。 |
| [^3] | [A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification.](http://arxiv.org/abs/2305.14752) | 本文介绍了一个结合大语言模型和形式验证的方法来自动验证和修复软件漏洞，并通过ESBMC-AI做出了概念验证。 |

# 详细

[^1]: 基于大型语言模型的智能界面在软件项目中的工作量和规模估计

    Effort and Size Estimation in Software Projects with Large Language Model-based Intelligent Interfaces

    [https://arxiv.org/abs/2402.07158](https://arxiv.org/abs/2402.07158)

    本文提出了一种基于大型语言模型的智能界面在软件项目中进行工作量和规模估计的方法，并通过比较传统方法，探讨了如何通过增强基于自然语言的问题规范来实现开发工作量的准确估计。

    

    大型语言模型（LLM）的发展也导致其应用的广泛增加。软件设计作为其中之一，在使用LLM作为扩展固定用户故事的接口组件方面获得了巨大的好处。然而，将基于LLM的人工智能代理包含在软件设计中常常带来意想不到的挑战，特别是在开发工作量的估计方面。通过基于用户界面的用户故事的例子，我们对比了传统方法，并提出了一种新的方法来增强基于自然语言的问题的规范，通过考虑数据源、接口和算法来进行开发工作量的估计。

    The advancement of Large Language Models (LLM) has also resulted in an equivalent proliferation in its applications. Software design, being one, has gained tremendous benefits in using LLMs as an interface component that extends fixed user stories. However, inclusion of LLM-based AI agents in software design often poses unexpected challenges, especially in the estimation of development efforts. Through the example of UI-based user stories, we provide a comparison against traditional methods and propose a new way to enhance specifications of natural language-based questions that allows for the estimation of development effort by taking into account data sources, interfaces and algorithms.
    
[^2]: ChIRAAG: 通过ChatGPT生成快速和自动断言的方法

    ChIRAAG: ChatGPT Informed Rapid and Automated Assertion Generation

    [https://arxiv.org/abs/2402.00093](https://arxiv.org/abs/2402.00093)

    本研究设计了一个基于大语言模型的流水线，通过自然语言规格生成英语、线性时态逻辑和SVA断言，并成功减少了断言错误率。

    

    System Verilog Assertion (SVA)的形式化是Formal Property Verification (FPV)过程中的一个关键但复杂的任务。传统上，SVA的形式化需要经验丰富的专家解释规格。这是耗时且容易出错的。然而，最近大语言模型（LLM）的进展使得基于LLM的自动断言生成引起了人们的兴趣。我们设计了一种新颖的基于LLM的流水线，用于从自然语言规格中生成英语、线性时态逻辑和SVA的断言。我们开发了一个基于OpenAI GPT4的自定义LLM用于实验。此外，我们还开发了测试平台来验证LLM生成的断言。只有43%的LLM生成的原始断言存在错误，包括语法和逻辑错误。通过使用从测试案例失败中得出的精心设计的提示，迭代地促使LLM，该流水线在最多九次提示迭代后可以生成正确的SVA。

    System Verilog Assertion (SVA) formulation, a critical yet complex task, is a pre-requisite in the Formal Property Verification (FPV) process. Traditionally, SVA formulation involves expert-driven interpretation of specifications. This is time consuming and prone to human error. However, recent advances in Large Language Models (LLM), LLM-informed automatic assertion generation is gaining interest. We designed a novel LLM-based pipeline to generate assertions in English Language, Linear Temporal Logic, and SVA from natural language specifications. We developed a custom LLM-based on OpenAI GPT4 for our experiments. Furthermore, we developed testbenches to verify/validate the LLM-generated assertions. Only 43% of LLM-generated raw assertions had errors, including syntax and logical errors. By iteratively prompting the LLMs using carefully crafted prompts derived from test case failures, the pipeline could generate correct SVAs after a maximum of nine iterations of prompting. Our results 
    
[^3]: 走向软件自愈：结合大语言模型和形式验证解决软件安全问题

    A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification. (arXiv:2305.14752v1 [cs.SE])

    [http://arxiv.org/abs/2305.14752](http://arxiv.org/abs/2305.14752)

    本文介绍了一个结合大语言模型和形式验证的方法来自动验证和修复软件漏洞，并通过ESBMC-AI做出了概念验证。

    

    本文提出了一种新方法，将大语言模型和形式化验证策略相结合，使得软件漏洞可以得到验证和自动修复。首先利用有限模型检查（BMC）定位软件漏洞和派生反例。然后，将反例和源代码提供给大语言模型引擎进行代码调试和生成，从而找到漏洞的根本原因并修复代码。最后，则使用BMC验证大语言模型生成的修正版本的代码。 作为概念证明，我们创建了ESBMC-AI，它基于高效的基于SMT的上下文有界模型检查器（ESBMC）和一个预训练的Transformer模型gpt-3.5-turbo来检测和修复C程序中的错误。

    In this paper we present a novel solution that combines the capabilities of Large Language Models (LLMs) with Formal Verification strategies to verify and automatically repair software vulnerabilities. Initially, we employ Bounded Model Checking (BMC) to locate the software vulnerability and derive a counterexample. The counterexample provides evidence that the system behaves incorrectly or contains a vulnerability. The counterexample that has been detected, along with the source code, are provided to the LLM engine. Our approach involves establishing a specialized prompt language for conducting code debugging and generation to understand the vulnerability's root cause and repair the code. Finally, we use BMC to verify the corrected version of the code generated by the LLM. As a proof of concept, we create ESBMC-AI based on the Efficient SMT-based Context-Bounded Model Checker (ESBMC) and a pre-trained Transformer model, specifically gpt-3.5-turbo, to detect and fix errors in C program
    

