# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Homeostatic motion planning with innate physics knowledge](https://arxiv.org/abs/2402.15384) | 通过定义"任务"的方式和引入具有物理和因果关系理解的监督模块，我们提出了一种具有固有物理知识的稳态运动规划框架，可以在机器人上实现复杂计划。 |
| [^2] | [BackpropTools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control.](http://arxiv.org/abs/2306.03530) | BackpropTools是一款快速、可移植的连续控制深度强化学习库，它通过模板元编程提供紧密集成的可组合组件，并在异构平台集合上无缝使用，同时在连续控制问题的深度RL代理高效可扩展训练方面具有优势。由于其可移植性和实时保证，它成为了在嵌入式设备上部署学来的策略的有价值的工具。 |

# 详细

[^1]: 具有固有物理知识的稳态运动规划

    Homeostatic motion planning with innate physics knowledge

    [https://arxiv.org/abs/2402.15384](https://arxiv.org/abs/2402.15384)

    通过定义"任务"的方式和引入具有物理和因果关系理解的监督模块，我们提出了一种具有固有物理知识的稳态运动规划框架，可以在机器人上实现复杂计划。

    

    生物体以闭环方式与周围环境进行互动，其中感官输入决定行为的启动和终止。即使是简单的动物也能制定并执行复杂计划，但纯闭环输入控制的机器人尚未复制这一点。我们提出通过定义一组离散临时闭环控制器，称为“任务”，每个任务代表一个闭环行为，来解决这个问题。我们进一步引入了一个具有固有物理和因果关系理解的监督模块，通过该模块可以模拟随时间执行任务序列并将结果存储在环境模型中。基于这个模型，可以通过链接临时闭环控制器进行制定计划。所提出的框架已在实际机器人中实施，并在两种场景下作为概念验证进行了测试。

    arXiv:2402.15384v1 Announce Type: cross  Abstract: Living organisms interact with their surroundings in a closed-loop fashion, where sensory inputs dictate the initiation and termination of behaviours. Even simple animals are able to develop and execute complex plans, which has not yet been replicated in robotics using pure closed-loop input control. We propose a solution to this problem by defining a set of discrete and temporary closed-loop controllers, called "tasks", each representing a closed-loop behaviour. We further introduce a supervisory module which has an innate understanding of physics and causality, through which it can simulate the execution of task sequences over time and store the results in a model of the environment. On the basis of this model, plans can be made by chaining temporary closed-loop controllers. The proposed framework was implemented for a real robot and tested in two scenarios as proof of concept.
    
[^2]: BackpropTools: 一款快速、可移植的连续控制深度强化学习库

    BackpropTools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control. (arXiv:2306.03530v1 [cs.LG])

    [http://arxiv.org/abs/2306.03530](http://arxiv.org/abs/2306.03530)

    BackpropTools是一款快速、可移植的连续控制深度强化学习库，它通过模板元编程提供紧密集成的可组合组件，并在异构平台集合上无缝使用，同时在连续控制问题的深度RL代理高效可扩展训练方面具有优势。由于其可移植性和实时保证，它成为了在嵌入式设备上部署学来的策略的有价值的工具。

    

    深度强化学习在许多领域中已被证明可以产生出具有能力的代理和控制策略，但常常受到训练时间过长的困扰。此外，在连续控制问题的情况下，现有深度学习库的实时性和可移植性的缺乏限制了学习策略在实际嵌入式设备上的应用。为了解决这些问题，我们提出了BackpropTools，一种依赖性-free、header-only、pure C++的深度监督和强化学习库。利用最近C++标准的模板元编程能力，我们提供了可以由编译器紧密集成的可组合组件。其新颖的架构允许BackpropTools在异构平台集合上无缝使用，从HPC集群、工作站和笔记本电脑到智能手机、智能手表和微控制器。具体来说，由于RL算法与模拟环境的紧密集成，BackpropTools在连续控制问题的深度RL代理的高效可扩展训练方面具有优势。此外，它的可移植性和实时保证使其成为在嵌入式设备上部署学来的策略的有价值的工具。

    Deep Reinforcement Learning (RL) has been demonstrated to yield capable agents and control policies in several domains but is commonly plagued by prohibitively long training times. Additionally, in the case of continuous control problems, the applicability of learned policies on real-world embedded devices is limited due to the lack of real-time guarantees and portability of existing deep learning libraries. To address these challenges, we present BackpropTools, a dependency-free, header-only, pure C++ library for deep supervised and reinforcement learning. Leveraging the template meta-programming capabilities of recent C++ standards, we provide composable components that can be tightly integrated by the compiler. Its novel architecture allows BackpropTools to be used seamlessly on a heterogeneous set of platforms, from HPC clusters over workstations and laptops to smartphones, smartwatches, and microcontrollers. Specifically, due to the tight integration of the RL algorithms with simu
    

