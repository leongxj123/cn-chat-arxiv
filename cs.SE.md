# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RepairAgent: An Autonomous, LLM-Based Agent for Program Repair](https://arxiv.org/abs/2403.17134) | RepairAgent是首个通过基于大型语言模型的自主代理程序来解决程序修复挑战的工作，其关键贡献在于提供了一组有助于程序修复的工具以及动态更新的提示格式。 |

# 详细

[^1]: RepairAgent：一种基于LLM的自主代理程序修复工具

    RepairAgent: An Autonomous, LLM-Based Agent for Program Repair

    [https://arxiv.org/abs/2403.17134](https://arxiv.org/abs/2403.17134)

    RepairAgent是首个通过基于大型语言模型的自主代理程序来解决程序修复挑战的工作，其关键贡献在于提供了一组有助于程序修复的工具以及动态更新的提示格式。

    

    自动程序修复已经成为一种强大的技术，可以减轻软件漏洞对系统可靠性和用户体验的影响。本文介绍了RepairAgent，这是第一个通过基于大型语言模型(LLM)的自主代理解决程序修复挑战的工作。与现有的基于深度学习的方法不同，这些方法会用固定的提示或在固定的反馈循环中提示模型，我们的工作将LLM视为一种能够自主规划和执行修复操作的代理程序，通过调用适当的工具修复漏洞。RepairAgent自由地交织着收集有关漏洞的信息、收集修复材料以及验证修复过程，并根据收集到的信息和先前的修复尝试反馈决定调用哪些工具。实现RepairAgent的关键贡献包括一组有助于程序修复的工具，以及一个动态更新的提示格式，使LLM能够进行交互。

    arXiv:2403.17134v1 Announce Type: cross  Abstract: Automated program repair has emerged as a powerful technique to mitigate the impact of software bugs on system reliability and user experience. This paper introduces RepairAgent, the first work to address the program repair challenge through an autonomous agent based on a large language model (LLM). Unlike existing deep learning-based approaches, which prompt a model with a fixed prompt or in a fixed feedback loop, our work treats the LLM as an agent capable of autonomously planning and executing actions to fix bugs by invoking suitable tools. RepairAgent freely interleaves gathering information about the bug, gathering repair ingredients, and validating fixes, while deciding which tools to invoke based on the gathered information and feedback from previous fix attempts. Key contributions that enable RepairAgent include a set of tools that are useful for program repair, a dynamically updated prompt format that allows the LLM to interac
    

