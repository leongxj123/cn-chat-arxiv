# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Realtime Motion Generation with Active Perception Using Attention Mechanism for Cooking Robot.](http://arxiv.org/abs/2309.14837) | 该论文介绍了一种使用注意机制的预测性递归神经网络，能够实现实时感知和动作生成，以支持烹饪机器人在煮鸡蛋过程中对鸡蛋状态的感知和搅拌动作的调整。 |
| [^2] | [Conformal Temporal Logic Planning using Large Language Models: Knowing When to Do What and When to Ask for Help.](http://arxiv.org/abs/2309.10092) | 本文提出了一个使用大型语言模型的一致时间逻辑规划方法，用于解决多个高级子任务的移动机器人运动规划问题。其中的一个关键挑战是如何以正确性的角度推理机器人计划与基于自然语言的逻辑任务的关系。 |

# 详细

[^1]: 使用注意机制进行实时动作生成和主动感知的烹饪机器人

    Realtime Motion Generation with Active Perception Using Attention Mechanism for Cooking Robot. (arXiv:2309.14837v1 [cs.RO])

    [http://arxiv.org/abs/2309.14837](http://arxiv.org/abs/2309.14837)

    该论文介绍了一种使用注意机制的预测性递归神经网络，能够实现实时感知和动作生成，以支持烹饪机器人在煮鸡蛋过程中对鸡蛋状态的感知和搅拌动作的调整。

    

    为了支持人类的日常生活，机器人需要自主学习，适应物体和环境，并执行适当的动作。我们尝试使用真实的食材煮炒鸡蛋的任务，其中机器人需要实时感知鸡蛋的状态并调整搅拌动作，同时鸡蛋被加热且状态不断变化。在以前的研究中，处理变化的物体被发现是具有挑战性的，因为感知信息包括动态的、重要或嘈杂的信息，而且每次应该关注的模态不断变化，这使得实现实时感知和动作生成变得困难。我们提出了一个带有注意机制的预测性递归神经网络，可以权衡传感器输入，区分每种模态的重要性和可靠性，实现快速和高效的感知和动作生成。模型通过示范学习进行训练，并允许不断更新。

    To support humans in their daily lives, robots are required to autonomously learn, adapt to objects and environments, and perform the appropriate actions. We tackled on the task of cooking scrambled eggs using real ingredients, in which the robot needs to perceive the states of the egg and adjust stirring movement in real time, while the egg is heated and the state changes continuously. In previous works, handling changing objects was found to be challenging because sensory information includes dynamical, both important or noisy information, and the modality which should be focused on changes every time, making it difficult to realize both perception and motion generation in real time. We propose a predictive recurrent neural network with an attention mechanism that can weigh the sensor input, distinguishing how important and reliable each modality is, that realize quick and efficient perception and motion generation. The model is trained with learning from the demonstration, and allow
    
[^2]: 使用大型语言模型的一致时间逻辑规划：知道何时做什么和何时寻求帮助。

    Conformal Temporal Logic Planning using Large Language Models: Knowing When to Do What and When to Ask for Help. (arXiv:2309.10092v1 [cs.RO])

    [http://arxiv.org/abs/2309.10092](http://arxiv.org/abs/2309.10092)

    本文提出了一个使用大型语言模型的一致时间逻辑规划方法，用于解决多个高级子任务的移动机器人运动规划问题。其中的一个关键挑战是如何以正确性的角度推理机器人计划与基于自然语言的逻辑任务的关系。

    

    本文解决了一个新的移动机器人运动规划问题，任务是以自然语言（NL）表达并以时间和逻辑顺序完成多个高级子任务。为了正式定义这样的任务，我们利用基于NL的原子谓词在LTL上定义了模型。这与相关的规划方法形成对比，这些方法在原子谓词上定义了捕捉所需低级系统配置的LTL任务。我们的目标是设计机器人计划，满足基于NL的原子命题定义的LTL任务。在这个设置中出现的一个新的技术挑战在于推理机器人计划的正确性与这些LTL编码的任务的关系。为了解决这个问题，我们提出了HERACLEs，一个分层一致的自然语言规划器，它依赖于现有工具的新型整合，包括（i）自动机理论，以确定机器人应该完成的NL指定的子任务以推进任务进展；

    This paper addresses a new motion planning problem for mobile robots tasked with accomplishing multiple high-level sub-tasks, expressed using natural language (NL), in a temporal and logical order. To formally define such missions, we leverage LTL defined over NL-based atomic predicates modeling the considered NL-based sub-tasks. This is contrast to related planning approaches that define LTL tasks over atomic predicates capturing desired low-level system configurations. Our goal is to design robot plans that satisfy LTL tasks defined over NL-based atomic propositions. A novel technical challenge arising in this setup lies in reasoning about correctness of a robot plan with respect to such LTL-encoded tasks. To address this problem, we propose HERACLEs, a hierarchical conformal natural language planner, that relies on a novel integration of existing tools that include (i) automata theory to determine the NL-specified sub-task the robot should accomplish next to make mission progress; (
    

