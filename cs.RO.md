# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [To Help or Not to Help: LLM-based Attentive Support for Human-Robot Group Interactions](https://arxiv.org/abs/2403.12533) | 该论文提出了基于LLM的专注支持交互概念，使机器人能够在不干扰群体的情况下支持和帮助人类。 |
| [^2] | [CoPAL: Corrective Planning of Robot Actions with Large Language Models.](http://arxiv.org/abs/2310.07263) | 本文提出了一个具有大规模语言模型的机器人动作纠正规划系统，通过处理生成计划中的物理基础、逻辑和语义错误的再规划策略，实现了在复杂环境中的任务和动作规划。通过仿真和实际场景的验证，证明了该系统的有效性。 |
| [^3] | [Learning Type-Generalized Actions for Symbolic Planning.](http://arxiv.org/abs/2308.04867) | 本文提出了一种学习通用型符号规划动作的新方法，通过给定实体层次结构和观察到的相似行为来实现通用化。在模拟的厨房环境中验证了该方法的有效性。 |

# 详细

[^1]: 是否帮助：基于LLM的专注支持与人机群体互动

    To Help or Not to Help: LLM-based Attentive Support for Human-Robot Group Interactions

    [https://arxiv.org/abs/2403.12533](https://arxiv.org/abs/2403.12533)

    该论文提出了基于LLM的专注支持交互概念，使机器人能够在不干扰群体的情况下支持和帮助人类。

    

    机器人如何在人类群体中提供不引人注目的物理支持？我们提出了Attentive Support，这是一个新颖的机器人与人类群体进行支持的交互概念。它将场景感知、对话获取、情况理解和行为生成与大规模语言模型（LLMs）的常识推理能力相结合。除了遵循用户的指令，Attentive Support能够决定何时以及如何支持人类，并在不干扰群体时保持沉默。通过多样化的场景，我们展示和评估了机器人的专注行为，当需要时支持和帮助人类，而如果不需要帮助，则不会干扰。

    arXiv:2403.12533v1 Announce Type: cross  Abstract: How can a robot provide unobtrusive physical support within a group of humans? We present Attentive Support, a novel interaction concept for robots to support a group of humans. It combines scene perception, dialogue acquisition, situation understanding, and behavior generation with the common-sense reasoning capabilities of Large Language Models (LLMs). In addition to following user instructions, Attentive Support is capable of deciding when and how to support the humans, and when to remain silent to not disturb the group. With a diverse set of scenarios, we show and evaluate the robot's attentive behavior, which supports and helps the humans when required, while not disturbing if no help is needed.
    
[^2]: CoPAL:具有大规模语言模型的机器人动作纠正规划

    CoPAL: Corrective Planning of Robot Actions with Large Language Models. (arXiv:2310.07263v1 [cs.RO])

    [http://arxiv.org/abs/2310.07263](http://arxiv.org/abs/2310.07263)

    本文提出了一个具有大规模语言模型的机器人动作纠正规划系统，通过处理生成计划中的物理基础、逻辑和语义错误的再规划策略，实现了在复杂环境中的任务和动作规划。通过仿真和实际场景的验证，证明了该系统的有效性。

    

    为了实现完全自主的机器人系统能够接管人类传统执行的任务，开放世界环境的复杂性提出了巨大的挑战。在这一背景下，本研究为应用于机器人任务和动作规划的大规模语言模型领域做出了贡献。我们提出了一个系统架构，协调多个认知层次之间的无缝交互，包括推理、规划和动作生成。其核心是一种处理生成的计划中的物理基础、逻辑和语义错误的新型再规划策略。通过在仿真环境和两个复杂的实际场景（方块世界、酒吧和比萨制作）中进行实证评估，我们展示了所提出的反馈架构的有效性，尤其是对可执行性、正确性和时间复杂性的影响。

    In the pursuit of fully autonomous robotic systems capable of taking over tasks traditionally performed by humans, the complexity of open-world environments poses a considerable challenge. Addressing this imperative, this study contributes to the field of Large Language Models (LLMs) applied to task and motion planning for robots. We propose a system architecture that orchestrates a seamless interplay between multiple cognitive levels, encompassing reasoning, planning, and motion generation. At its core lies a novel replanning strategy that handles physically grounded, logical, and semantic errors in the generated plans. We demonstrate the efficacy of the proposed feedback architecture, particularly its impact on executability, correctness, and time complexity via empirical evaluation in the context of a simulation and two intricate real-world scenarios: blocks world, barman and pizza preparation.
    
[^3]: 学习通用型符号规划的动作

    Learning Type-Generalized Actions for Symbolic Planning. (arXiv:2308.04867v1 [cs.AI])

    [http://arxiv.org/abs/2308.04867](http://arxiv.org/abs/2308.04867)

    本文提出了一种学习通用型符号规划动作的新方法，通过给定实体层次结构和观察到的相似行为来实现通用化。在模拟的厨房环境中验证了该方法的有效性。

    

    符号规划是一种强大的技术，用于解决需要长序列动作并装备智能体具有复杂行为的复杂任务。这种方法的缺点是需要合适的符号表示来描述环境的状态以及能够改变状态的动作。传统上，这些表示是由专家为不同的问题域精心设计的，这限制了它们在不同问题和环境复杂性上的可转移性。在这篇论文中，我们提出了一种新的概念，使用给定的实体层次结构和观察到的相似行为来通用化符号动作。在一个模拟的基于网格的厨房环境中，我们展示了从少量观察中学习到的通用型动作能够泛化到新的情况。在规划过程中引入额外的即时通用化机制，可以处理未见过的任务组合、较长序列、新实体和意外环境行为。

    Symbolic planning is a powerful technique to solve complex tasks that require long sequences of actions and can equip an intelligent agent with complex behavior. The downside of this approach is the necessity for suitable symbolic representations describing the state of the environment as well as the actions that can change it. Traditionally such representations are carefully hand-designed by experts for distinct problem domains, which limits their transferability to different problems and environment complexities. In this paper, we propose a novel concept to generalize symbolic actions using a given entity hierarchy and observed similar behavior. In a simulated grid-based kitchen environment, we show that type-generalized actions can be learned from few observations and generalize to novel situations. Incorporating an additional on-the-fly generalization mechanism during planning, unseen task combinations, involving longer sequences, novel entities and unexpected environment behavior,
    

