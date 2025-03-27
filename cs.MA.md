# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models](https://arxiv.org/abs/2403.17246) | 该论文将经典规划和大型语言模型相结合，通过近似人类直觉，以实现多智能体任务规划。 |
| [^2] | [Framework for developing quantitative agent based models based on qualitative expert knowledge: an organised crime use-case.](http://arxiv.org/abs/2308.00505) | 提出了一个基于定性专家知识的量化代理模型开发框架，该框架通过将定性数据翻译成定量规则，为模型构建者和领域专家提供了一个系统和透明的建模过程。以一个有组织犯罪的应用案例为例，演示了该框架的方法。 |

# 详细

[^1]: TwoStep: 使用经典规划器和大型语言模型进行多智能体任务规划

    TwoStep: Multi-agent Task Planning using Classical Planners and Large Language Models

    [https://arxiv.org/abs/2403.17246](https://arxiv.org/abs/2403.17246)

    该论文将经典规划和大型语言模型相结合，通过近似人类直觉，以实现多智能体任务规划。

    

    类似规划领域定义语言（PDDL）之类的经典规划公式允许确定可实现目标状态的动作序列，只要存在任何可能的初始状态。然而，PDDL中定义的推理问题并未捕获行动进行的时间方面，例如领域中的两个智能体如果彼此的后况不干扰前提条件，则可以同时执行一个动作。人类专家可以将目标分解为大部分独立的组成部分，并将每个智能体分配给其中一个子目标，以利用同时进行动作来加快计划步骤的执行，每个部分仅使用单个智能体规划。相比之下，直接推断计划步骤的大型语言模型（LLMs）并不保证执行成功，但利用常识推理来组装动作序列。我们通过近似人类直觉，结合了经典规划和LLMs的优势

    arXiv:2403.17246v1 Announce Type: new  Abstract: Classical planning formulations like the Planning Domain Definition Language (PDDL) admit action sequences guaranteed to achieve a goal state given an initial state if any are possible. However, reasoning problems defined in PDDL do not capture temporal aspects of action taking, for example that two agents in the domain can execute an action simultaneously if postconditions of each do not interfere with preconditions of the other. A human expert can decompose a goal into largely independent constituent parts and assign each agent to one of these subgoals to take advantage of simultaneous actions for faster execution of plan steps, each using only single agent planning. By contrast, large language models (LLMs) used for directly inferring plan steps do not guarantee execution success, but do leverage commonsense reasoning to assemble action sequences. We combine the strengths of classical planning and LLMs by approximating human intuition
    
[^2]: 基于定性专家知识的量化代理模型开发框架：一个有组织犯罪的应用案例

    Framework for developing quantitative agent based models based on qualitative expert knowledge: an organised crime use-case. (arXiv:2308.00505v1 [cs.AI])

    [http://arxiv.org/abs/2308.00505](http://arxiv.org/abs/2308.00505)

    提出了一个基于定性专家知识的量化代理模型开发框架，该框架通过将定性数据翻译成定量规则，为模型构建者和领域专家提供了一个系统和透明的建模过程。以一个有组织犯罪的应用案例为例，演示了该框架的方法。

    

    为了对执法目的建模犯罪网络，需要将有限的数据转化为经过验证的基于代理的模型。当前刑事学建模中缺少一个为模型构建者和领域专家提供系统和透明框架的方法，该方法建立了计算犯罪建模的建模过程，包括将定性数据转化为定量规则。因此，我们提出了FREIDA（基于专家知识驱动的数据驱动代理模型框架）。在本文中，犯罪可卡因替代模型（CCRM）将作为示例案例，以演示FREIDA方法。对于CCRM，正在建模荷兰的一个有组织可卡因网络，试图通过移除首脑节点，使剩余代理重新组织，并将网络恢复到稳定状态。定性数据源，例如案件文件，文献和采访，被转化为经验法则。

    In order to model criminal networks for law enforcement purposes, a limited supply of data needs to be translated into validated agent-based models. What is missing in current criminological modelling is a systematic and transparent framework for modelers and domain experts that establishes a modelling procedure for computational criminal modelling that includes translating qualitative data into quantitative rules. For this, we propose FREIDA (Framework for Expert-Informed Data-driven Agent-based models). Throughout the paper, the criminal cocaine replacement model (CCRM) will be used as an example case to demonstrate the FREIDA methodology. For the CCRM, a criminal cocaine network in the Netherlands is being modelled where the kingpin node is being removed, the goal being for the remaining agents to reorganize after the disruption and return the network into a stable state. Qualitative data sources such as case files, literature and interviews are translated into empirical laws, and c
    

