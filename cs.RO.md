# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight](https://arxiv.org/abs/2403.12203) | 在基于视觉的自主无人机竞速中，本研究提出了将强化学习和模仿学习相结合的新型训练框架，以克服样本效率和计算需求方面的挑战，并通过三个阶段的方法进行性能受限的自适应RL微调 |
| [^2] | [Dynamic planning in hierarchical active inference](https://arxiv.org/abs/2402.11658) | 通过研究在动态规划领域中模拟工具使用的目标，我们深入探讨了主动推断中的动态规划，该领域考虑到生物目标导向行为的两个关键方面 |
| [^3] | [Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation.](http://arxiv.org/abs/2401.12275) | 本文提出了一种多Agent动态关系推理方法，通过明确推断关系结构的演化，来实现在社交机器人导航中的有效性。方法包括推断超边缘以实现群体推理和轨迹预测器生成未来状态。 |

# 详细

[^1]: 基于模仿的增强学习为基于视觉的敏捷飞行引导引导

    Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight

    [https://arxiv.org/abs/2403.12203](https://arxiv.org/abs/2403.12203)

    在基于视觉的自主无人机竞速中，本研究提出了将强化学习和模仿学习相结合的新型训练框架，以克服样本效率和计算需求方面的挑战，并通过三个阶段的方法进行性能受限的自适应RL微调

    

    我们在基于视觉的自主无人机竞速的背景下，将强化学习（RL）的有效性和模仿学习（IL）的效率结合在一起。我们专注于直接处理视觉输入，而无需明确的状态估计。虽然强化学习通过试错提供了一个学习复杂控制器的通用框架，但面临着样本效率和计算需求的挑战，因为视觉输入的维度较高。相反，IL在从视觉演示中学习方面表现出效率，但受到演示质量的限制，并面临诸如协变量漂移的问题。为了克服这些限制，我们提出了一个结合RL和IL优势的新型训练框架。我们的框架包括三个阶段：使用特权状态信息的师傅策略的初始训练，使用IL将此策略蒸馏为学生策略，以及性能受限的自适应RL微调

    arXiv:2403.12203v1 Announce Type: cross  Abstract: We combine the effectiveness of Reinforcement Learning (RL) and the efficiency of Imitation Learning (IL) in the context of vision-based, autonomous drone racing. We focus on directly processing visual input without explicit state estimation. While RL offers a general framework for learning complex controllers through trial and error, it faces challenges regarding sample efficiency and computational demands due to the high dimensionality of visual inputs. Conversely, IL demonstrates efficiency in learning from visual demonstrations but is limited by the quality of those demonstrations and faces issues like covariate shift. To overcome these limitations, we propose a novel training framework combining RL and IL's advantages. Our framework involves three stages: initial training of a teacher policy using privileged state information, distilling this policy into a student policy using IL, and performance-constrained adaptive RL fine-tunin
    
[^2]: 分层主动推断中的动态规划

    Dynamic planning in hierarchical active inference

    [https://arxiv.org/abs/2402.11658](https://arxiv.org/abs/2402.11658)

    通过研究在动态规划领域中模拟工具使用的目标，我们深入探讨了主动推断中的动态规划，该领域考虑到生物目标导向行为的两个关键方面

    

    通过动态规划，我们指的是人类大脑推断和施加与认知决策相关的运动轨迹的能力。最近的一个范式，主动推断，为生物有机体适应带来了基本见解，不断努力最小化预测误差以将自己限制在与生命兼容的状态。在过去的几年里，许多研究表明人类和动物行为可以解释为主动推断过程，无论是作为离散决策还是连续运动控制，都激发了机器人技术和人工智能中的创新解决方案。然而，文献缺乏对如何有效地在变化环境中规划行动的全面展望。我们设定了对工具使用进行建模的目标，深入研究了主动推断中的动态规划主题，牢记两个生物目标导向行为的关键方面：理解……

    arXiv:2402.11658v1 Announce Type: new  Abstract: By dynamic planning, we refer to the ability of the human brain to infer and impose motor trajectories related to cognitive decisions. A recent paradigm, active inference, brings fundamental insights into the adaptation of biological organisms, constantly striving to minimize prediction errors to restrict themselves to life-compatible states. Over the past years, many studies have shown how human and animal behavior could be explained in terms of an active inferential process -- either as discrete decision-making or continuous motor control -- inspiring innovative solutions in robotics and artificial intelligence. Still, the literature lacks a comprehensive outlook on how to effectively plan actions in changing environments. Setting ourselves the goal of modeling tool use, we delve into the topic of dynamic planning in active inference, keeping in mind two crucial aspects of biological goal-directed behavior: the capacity to understand a
    
[^3]: 多Agent动态关系推理用于社交机器人导航

    Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation. (arXiv:2401.12275v1 [cs.RO])

    [http://arxiv.org/abs/2401.12275](http://arxiv.org/abs/2401.12275)

    本文提出了一种多Agent动态关系推理方法，通过明确推断关系结构的演化，来实现在社交机器人导航中的有效性。方法包括推断超边缘以实现群体推理和轨迹预测器生成未来状态。

    

    社交机器人导航在日常生活的各种情景下可以提供帮助，但需要安全的人机交互和高效的轨迹规划。在多Agent交互系统中，建模成对的关系已经被广泛研究，但是捕捉更大规模的群体活动的能力有限。在本文中，我们提出了一种系统的关系推理方法，通过明确推断正在演变的关系结构，展示了其在多Agent轨迹预测和社交机器人导航中的有效性。除了节点对之间的边缘（即Agent），我们还提出了推断超边缘的方法，以自适应地连接多个节点，以便进行群体推理。我们的方法推断动态演化的关系图和超图，以捕捉关系的演化，轨迹预测器利用这些图来生成未来状态。同时，我们提出了对锐度和逻辑稀疏性进行正则化的方法。

    Social robot navigation can be helpful in various contexts of daily life but requires safe human-robot interactions and efficient trajectory planning. While modeling pairwise relations has been widely studied in multi-agent interacting systems, the ability to capture larger-scale group-wise activities is limited. In this paper, we propose a systematic relational reasoning approach with explicit inference of the underlying dynamically evolving relational structures, and we demonstrate its effectiveness for multi-agent trajectory prediction and social robot navigation. In addition to the edges between pairs of nodes (i.e., agents), we propose to infer hyperedges that adaptively connect multiple nodes to enable group-wise reasoning in an unsupervised manner. Our approach infers dynamically evolving relation graphs and hypergraphs to capture the evolution of relations, which the trajectory predictor employs to generate future states. Meanwhile, we propose to regularize the sharpness and sp
    

