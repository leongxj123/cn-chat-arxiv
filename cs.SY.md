# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Homeostatic motion planning with innate physics knowledge](https://arxiv.org/abs/2402.15384) | 通过定义"任务"的方式和引入具有物理和因果关系理解的监督模块，我们提出了一种具有固有物理知识的稳态运动规划框架，可以在机器人上实现复杂计划。 |

# 详细

[^1]: 具有固有物理知识的稳态运动规划

    Homeostatic motion planning with innate physics knowledge

    [https://arxiv.org/abs/2402.15384](https://arxiv.org/abs/2402.15384)

    通过定义"任务"的方式和引入具有物理和因果关系理解的监督模块，我们提出了一种具有固有物理知识的稳态运动规划框架，可以在机器人上实现复杂计划。

    

    生物体以闭环方式与周围环境进行互动，其中感官输入决定行为的启动和终止。即使是简单的动物也能制定并执行复杂计划，但纯闭环输入控制的机器人尚未复制这一点。我们提出通过定义一组离散临时闭环控制器，称为“任务”，每个任务代表一个闭环行为，来解决这个问题。我们进一步引入了一个具有固有物理和因果关系理解的监督模块，通过该模块可以模拟随时间执行任务序列并将结果存储在环境模型中。基于这个模型，可以通过链接临时闭环控制器进行制定计划。所提出的框架已在实际机器人中实施，并在两种场景下作为概念验证进行了测试。

    arXiv:2402.15384v1 Announce Type: cross  Abstract: Living organisms interact with their surroundings in a closed-loop fashion, where sensory inputs dictate the initiation and termination of behaviours. Even simple animals are able to develop and execute complex plans, which has not yet been replicated in robotics using pure closed-loop input control. We propose a solution to this problem by defining a set of discrete and temporary closed-loop controllers, called "tasks", each representing a closed-loop behaviour. We further introduce a supervisory module which has an innate understanding of physics and causality, through which it can simulate the execution of task sequences over time and store the results in a model of the environment. On the basis of this model, plans can be made by chaining temporary closed-loop controllers. The proposed framework was implemented for a real robot and tested in two scenarios as proof of concept.
    

