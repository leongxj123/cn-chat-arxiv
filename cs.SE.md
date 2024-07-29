# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models.](http://arxiv.org/abs/2310.16340) | RCAgent是一个工具增强的LLM自主代理框架，用于云根本原因分析，能够实现自由格式的数据收集和全面的分析，并在各个方面优于当前方法。 |

# 详细

[^1]: RCAgent：基于自主代理和增强的大型语言模型的云根本原因分析

    RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models. (arXiv:2310.16340v1 [cs.SE])

    [http://arxiv.org/abs/2310.16340](http://arxiv.org/abs/2310.16340)

    RCAgent是一个工具增强的LLM自主代理框架，用于云根本原因分析，能够实现自由格式的数据收集和全面的分析，并在各个方面优于当前方法。

    

    最近，云根本原因分析中的大型语言模型（LLM）应用受到了积极的关注。然而，当前方法仍然依赖于手动工作流设置，并没有充分发挥LLMs的决策和环境交互能力。我们提出了RCAgent，这是一个实用和注重隐私的工具增强LLM自主代理框架，用于实际的工业RCA使用。RCAgent在内部部署的模型上运行，而不是GPT系列，能够进行自由格式的数据收集和全面的分析，并结合各种增强功能，包括独特的行动轨迹自一致性和一套用于上下文管理、稳定化和导入领域知识的方法。我们的实验证明RCAgent在RCA的各个方面（预测根本原因、解决方案、证据和责任）以及当前规则未涵盖的任务上都明显优于ReAct，得到了自动化和人工验证的确认。

    Large language model (LLM) applications in cloud root cause analysis (RCA) have been actively explored recently. However, current methods are still reliant on manual workflow settings and do not unleash LLMs' decision-making and environment interaction capabilities. We present RCAgent, a tool-augmented LLM autonomous agent framework for practical and privacy-aware industrial RCA usage. Running on an internally deployed model rather than GPT families, RCAgent is capable of free-form data collection and comprehensive analysis with tools. Our framework combines a variety of enhancements, including a unique Self-Consistency for action trajectories, and a suite of methods for context management, stabilization, and importing domain knowledge. Our experiments show RCAgent's evident and consistent superiority over ReAct across all aspects of RCA -- predicting root causes, solutions, evidence, and responsibilities -- and tasks covered or uncovered by current rules, as validated by both automate
    

