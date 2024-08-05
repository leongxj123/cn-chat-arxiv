# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated System-level Testing of Unmanned Aerial Systems](https://arxiv.org/abs/2403.15857) | 本文提出了一种利用模型测试和人工智能技术自动生成、执行和评估无人机系统级测试的新颖方法。 |
| [^2] | [RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models.](http://arxiv.org/abs/2310.16340) | RCAgent是一个工具增强的LLM自主代理框架，用于云根本原因分析，能够实现自由格式的数据收集和全面的分析，并在各个方面优于当前方法。 |

# 详细

[^1]: 无人机系统级测试的自动化系统

    Automated System-level Testing of Unmanned Aerial Systems

    [https://arxiv.org/abs/2403.15857](https://arxiv.org/abs/2403.15857)

    本文提出了一种利用模型测试和人工智能技术自动生成、执行和评估无人机系统级测试的新颖方法。

    

    无人机系统依赖于各种安全关键和任务关键的航空电子系统。国际安全标准的主要要求之一是对航空电子软件系统进行严格的系统级测试。当前工业实践是手动创建测试方案，使用模拟器手动/自动执行这些方案，并手动评估结果。本文提出了一种新颖的方法来自动化无人机系统级测试。所提出的方法(AITester)利用基于模型的测试和人工智能(AI)技术，自动生成、执行和评估各种测试方案。

    arXiv:2403.15857v1 Announce Type: cross  Abstract: Unmanned aerial systems (UAS) rely on various avionics systems that are safety-critical and mission-critical. A major requirement of international safety standards is to perform rigorous system-level testing of avionics software systems. The current industrial practice is to manually create test scenarios, manually/automatically execute these scenarios using simulators, and manually evaluate outcomes. The test scenarios typically consist of setting certain flight or environment conditions and testing the system under test in these settings. The state-of-the-art approaches for this purpose also require manual test scenario development and evaluation. In this paper, we propose a novel approach to automate the system-level testing of the UAS. The proposed approach (AITester) utilizes model-based testing and artificial intelligence (AI) techniques to automatically generate, execute, and evaluate various test scenarios. The test scenarios a
    
[^2]: RCAgent：基于自主代理和增强的大型语言模型的云根本原因分析

    RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models. (arXiv:2310.16340v1 [cs.SE])

    [http://arxiv.org/abs/2310.16340](http://arxiv.org/abs/2310.16340)

    RCAgent是一个工具增强的LLM自主代理框架，用于云根本原因分析，能够实现自由格式的数据收集和全面的分析，并在各个方面优于当前方法。

    

    最近，云根本原因分析中的大型语言模型（LLM）应用受到了积极的关注。然而，当前方法仍然依赖于手动工作流设置，并没有充分发挥LLMs的决策和环境交互能力。我们提出了RCAgent，这是一个实用和注重隐私的工具增强LLM自主代理框架，用于实际的工业RCA使用。RCAgent在内部部署的模型上运行，而不是GPT系列，能够进行自由格式的数据收集和全面的分析，并结合各种增强功能，包括独特的行动轨迹自一致性和一套用于上下文管理、稳定化和导入领域知识的方法。我们的实验证明RCAgent在RCA的各个方面（预测根本原因、解决方案、证据和责任）以及当前规则未涵盖的任务上都明显优于ReAct，得到了自动化和人工验证的确认。

    Large language model (LLM) applications in cloud root cause analysis (RCA) have been actively explored recently. However, current methods are still reliant on manual workflow settings and do not unleash LLMs' decision-making and environment interaction capabilities. We present RCAgent, a tool-augmented LLM autonomous agent framework for practical and privacy-aware industrial RCA usage. Running on an internally deployed model rather than GPT families, RCAgent is capable of free-form data collection and comprehensive analysis with tools. Our framework combines a variety of enhancements, including a unique Self-Consistency for action trajectories, and a suite of methods for context management, stabilization, and importing domain knowledge. Our experiments show RCAgent's evident and consistent superiority over ReAct across all aspects of RCA -- predicting root causes, solutions, evidence, and responsibilities -- and tasks covered or uncovered by current rules, as validated by both automate
    

