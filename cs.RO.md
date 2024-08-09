# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MANGO: A Benchmark for Evaluating Mapping and Navigation Abilities of Large Language Models](https://arxiv.org/abs/2403.19913) | 提出了用于评估大型语言模型执行文本映射和导航能力的MANGO基准，发现即使是迄今为止最好的语言模型GPT-4在回答涉及映射和导航的问题时表现不佳。 |
| [^2] | [Guided Data Augmentation for Offline Reinforcement Learning and Imitation Learning.](http://arxiv.org/abs/2310.18247) | 该论文提出了一种人工引导的数据增强框架（GuDA）用于提高演示学习模型的性能。 |
| [^3] | [Representation Abstractions as Incentives for Reinforcement Learning Agents: A Robotic Grasping Case Study.](http://arxiv.org/abs/2309.11984) | 本文研究了不同状态表示对强化学习代理在机器人抓取任务上的影响，结果显示使用数字状态的代理能够在模拟环境中成功解决问题，并在真实机器人上实现了学习策略的可转移性。 |
| [^4] | [Conformal Temporal Logic Planning using Large Language Models: Knowing When to Do What and When to Ask for Help.](http://arxiv.org/abs/2309.10092) | 本文提出了一个使用大型语言模型的一致时间逻辑规划方法，用于解决多个高级子任务的移动机器人运动规划问题。其中的一个关键挑战是如何以正确性的角度推理机器人计划与基于自然语言的逻辑任务的关系。 |

# 详细

[^1]: MANGO：用于评估大型语言模型映射和导航能力的基准

    MANGO: A Benchmark for Evaluating Mapping and Navigation Abilities of Large Language Models

    [https://arxiv.org/abs/2403.19913](https://arxiv.org/abs/2403.19913)

    提出了用于评估大型语言模型执行文本映射和导航能力的MANGO基准，发现即使是迄今为止最好的语言模型GPT-4在回答涉及映射和导航的问题时表现不佳。

    

    如ChatGPT和GPT-4等大型语言模型最近在各种自然语言处理任务上取得了惊人的性能。本文提出了MANGO，这是一个用于评估它们执行基于文本映射和导航能力的基准。我们的基准包括来自一套文本游戏的53个迷宫：每个迷宫都与一个游览说明配对，其中包含每个位置的访问但不涵盖所有可能的路径。任务是问答：对于每个迷宫，大型语言模型读取游览说明并回答数百个映射和导航问题，例如“你应该从房子西部如何去阁楼？”和“如果我们从地下室向北和东走，我们会在哪里？”。尽管这些问题对人类来说很容易，但事实证明，迄今为止最好的语言模型GPT-4甚至在回答这些问题时表现不佳。此外，我们的实验表明，强大的映射和导航能力将有利于大型语言模型。

    arXiv:2403.19913v1 Announce Type: cross  Abstract: Large language models such as ChatGPT and GPT-4 have recently achieved astonishing performance on a variety of natural language processing tasks. In this paper, we propose MANGO, a benchmark to evaluate their capabilities to perform text-based mapping and navigation. Our benchmark includes 53 mazes taken from a suite of textgames: each maze is paired with a walkthrough that visits every location but does not cover all possible paths. The task is question-answering: for each maze, a large language model reads the walkthrough and answers hundreds of mapping and navigation questions such as "How should you go to Attic from West of House?" and "Where are we if we go north and east from Cellar?". Although these questions are easy to humans, it turns out that even GPT-4, the best-to-date language model, performs poorly at answering them. Further, our experiments suggest that a strong mapping and navigation ability would benefit large languag
    
[^2]: 为离线增强学习和模仿学习提供指导性数据增强

    Guided Data Augmentation for Offline Reinforcement Learning and Imitation Learning. (arXiv:2310.18247v1 [cs.LG])

    [http://arxiv.org/abs/2310.18247](http://arxiv.org/abs/2310.18247)

    该论文提出了一种人工引导的数据增强框架（GuDA）用于提高演示学习模型的性能。

    

    演示学习是一种使用专家演示来学习机器人控制策略的流行技术。然而，获取专家级演示的难度限制了演示学习方法的适用性：现实世界的数据收集通常很昂贵，并且演示的质量很大程度上取决于演示者的能力和安全问题。一些工作利用数据增强来廉价生成额外的演示数据，但大多数数据增强方法以随机方式生成增强数据，最终产生高度次优的数据。在这项工作中，我们提出了一种人工引导的数据增强框架（GuDA），用于生成高质量的增强数据。GuDA的关键洞见是，虽然演示动作序列可能很难展示产生专家数据所需的动作序列，但用户经常可以轻松地辨别出增强轨迹段表示的任务进展。因此，用户可以施加一系列s

    Learning from demonstration (LfD) is a popular technique that uses expert demonstrations to learn robot control policies. However, the difficulty in acquiring expert-quality demonstrations limits the applicability of LfD methods: real-world data collection is often costly, and the quality of the demonstrations depends greatly on the demonstrator's abilities and safety concerns. A number of works have leveraged data augmentation (DA) to inexpensively generate additional demonstration data, but most DA works generate augmented data in a random fashion and ultimately produce highly suboptimal data. In this work, we propose Guided Data Augmentation (GuDA), a human-guided DA framework that generates expert-quality augmented data. The key insight of GuDA is that while it may be difficult to demonstrate the sequence of actions required to produce expert data, a user can often easily identify when an augmented trajectory segment represents task progress. Thus, the user can impose a series of s
    
[^3]: 表示抽象作为强化学习代理的激励：基于机器人抓取的案例研究

    Representation Abstractions as Incentives for Reinforcement Learning Agents: A Robotic Grasping Case Study. (arXiv:2309.11984v1 [cs.RO])

    [http://arxiv.org/abs/2309.11984](http://arxiv.org/abs/2309.11984)

    本文研究了不同状态表示对强化学习代理在机器人抓取任务上的影响，结果显示使用数字状态的代理能够在模拟环境中成功解决问题，并在真实机器人上实现了学习策略的可转移性。

    

    选择一个适当的环境表示对于强化学习代理的决策过程并不总是简单的。状态表示应该足够包容，以便让代理能够信息地决定其行动，并且足够紧凑，以提高策略训练的样本效率。本文研究了不同状态表示对代理在特定机器人任务（对称和平面物体抓取）上解决问题的影响。从具有完整系统知识的基于模型的方法开始，通过手工数字表示到基于图像的表示，逐渐减少任务特定知识的引入量，定义了一系列状态表示抽象。我们研究了每种表示对代理在仿真环境中解决任务以及学到的策略在真实机器人上的可转移性的影响。结果表明，使用数字状态的强化学习代理能够在模拟环境中解决问题。

    Choosing an appropriate representation of the environment for the underlying decision-making process of the \gls{RL} agent is not always straightforward. The state representation should be inclusive enough to allow the agent to informatively decide on its actions and compact enough to increase sample efficiency for policy training. Given this outlook, this work examines the effect of various state representations in incentivizing the agent to solve a specific robotic task: antipodal and planar object grasping. A continuum of state representation abstractions is defined, starting from a model-based approach with complete system knowledge, through hand-crafted numerical, to image-based representations with decreasing level of induced task-specific knowledge. We examine the effects of each representation in the ability of the agent to solve the task in simulation and the transferability of the learned policy to the real robot. The results show that RL agents using numerical states can per
    
[^4]: 使用大型语言模型的一致时间逻辑规划：知道何时做什么和何时寻求帮助。

    Conformal Temporal Logic Planning using Large Language Models: Knowing When to Do What and When to Ask for Help. (arXiv:2309.10092v1 [cs.RO])

    [http://arxiv.org/abs/2309.10092](http://arxiv.org/abs/2309.10092)

    本文提出了一个使用大型语言模型的一致时间逻辑规划方法，用于解决多个高级子任务的移动机器人运动规划问题。其中的一个关键挑战是如何以正确性的角度推理机器人计划与基于自然语言的逻辑任务的关系。

    

    本文解决了一个新的移动机器人运动规划问题，任务是以自然语言（NL）表达并以时间和逻辑顺序完成多个高级子任务。为了正式定义这样的任务，我们利用基于NL的原子谓词在LTL上定义了模型。这与相关的规划方法形成对比，这些方法在原子谓词上定义了捕捉所需低级系统配置的LTL任务。我们的目标是设计机器人计划，满足基于NL的原子命题定义的LTL任务。在这个设置中出现的一个新的技术挑战在于推理机器人计划的正确性与这些LTL编码的任务的关系。为了解决这个问题，我们提出了HERACLEs，一个分层一致的自然语言规划器，它依赖于现有工具的新型整合，包括（i）自动机理论，以确定机器人应该完成的NL指定的子任务以推进任务进展；

    This paper addresses a new motion planning problem for mobile robots tasked with accomplishing multiple high-level sub-tasks, expressed using natural language (NL), in a temporal and logical order. To formally define such missions, we leverage LTL defined over NL-based atomic predicates modeling the considered NL-based sub-tasks. This is contrast to related planning approaches that define LTL tasks over atomic predicates capturing desired low-level system configurations. Our goal is to design robot plans that satisfy LTL tasks defined over NL-based atomic propositions. A novel technical challenge arising in this setup lies in reasoning about correctness of a robot plan with respect to such LTL-encoded tasks. To address this problem, we propose HERACLEs, a hierarchical conformal natural language planner, that relies on a novel integration of existing tools that include (i) automata theory to determine the NL-specified sub-task the robot should accomplish next to make mission progress; (
    

