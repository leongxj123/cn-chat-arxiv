# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can LLMs Patch Security Issues?.](http://arxiv.org/abs/2312.00024) | 本文提出了一种新的方法, Feedback-Driven Solution Synthesis (FDSS), 旨在通过将LLMs与静态代码分析工具Bandit结合，解决代码中的安全漏洞问题。该方法在现有方法的基础上有显著改进，并引入了一个新的数据集PythonSecurityEval。 |

# 详细

[^1]: LLMs能够修复安全问题吗？

    Can LLMs Patch Security Issues?. (arXiv:2312.00024v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2312.00024](http://arxiv.org/abs/2312.00024)

    本文提出了一种新的方法, Feedback-Driven Solution Synthesis (FDSS), 旨在通过将LLMs与静态代码分析工具Bandit结合，解决代码中的安全漏洞问题。该方法在现有方法的基础上有显著改进，并引入了一个新的数据集PythonSecurityEval。

    

    大型语言模型(LLMs)在代码生成方面显示出了令人印象深刻的能力。然而，类似于人类开发者，这些模型可能会生成包含安全漏洞和缺陷的代码。编写安全代码仍然是一个重大挑战，因为漏洞通常在程序与外部系统或服务（如数据库和操作系统）之间的交互过程中出现。在本文中，我们提出了一种新颖的方法，即基于反馈的解决方案合成（FDSS），旨在探索使用LLMs接收来自静态代码分析工具Bandit的反馈，然后LLMs生成潜在解决方案来解决安全漏洞。每个解决方案以及易受攻击的代码随后被送回LLMs进行代码完善。我们的方法在基线上表现出显著改进，并优于现有方法。此外，我们引入了一个新的数据集PythonSecurityEval，该数据集收集了来自Stack Overflow的真实场景数据。

    Large Language Models (LLMs) have shown impressive proficiency in code generation. Nonetheless, similar to human developers, these models might generate code that contains security vulnerabilities and flaws. Writing secure code remains a substantial challenge, as vulnerabilities often arise during interactions between programs and external systems or services, such as databases and operating systems. In this paper, we propose a novel approach, Feedback-Driven Solution Synthesis (FDSS), designed to explore the use of LLMs in receiving feedback from Bandit, which is a static code analysis tool, and then the LLMs generate potential solutions to resolve security vulnerabilities. Each solution, along with the vulnerable code, is then sent back to the LLM for code refinement. Our approach shows a significant improvement over the baseline and outperforms existing approaches. Furthermore, we introduce a new dataset, PythonSecurityEval, collected from real-world scenarios on Stack Overflow to e
    

