# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond Joint Demonstrations: Personalized Expert Guidance for Efficient Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2403.08936) | 提出了个性化专家示范的概念，为每个智体或不同类型的智体提供针对个人目标的指导，解决了多智体强化学习中联合示范困难的问题。 |
| [^2] | [Grasp, See and Place: Efficient Unknown Object Rearrangement with Policy Structure Prior](https://arxiv.org/abs/2402.15402) | 该论文提出了一种具有策略结构先验的高效未知物体重新排列系统，通过内外环的学习，实现了抓取、观察和放置在感知噪声中的优化。 |

# 详细

[^1]: 超越联合示范：个性化专家指导用于高效多智体强化学习

    Beyond Joint Demonstrations: Personalized Expert Guidance for Efficient Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2403.08936](https://arxiv.org/abs/2403.08936)

    提出了个性化专家示范的概念，为每个智体或不同类型的智体提供针对个人目标的指导，解决了多智体强化学习中联合示范困难的问题。

    

    多智体强化学习算法面临有效探索的挑战，因为联合状态-动作空间的大小呈指数增长。虽然示范引导学习在单智体环境中已被证明是有益的，但其直接应用于多智体强化学习受到获得联合专家示范的实际困难的阻碍。在这项工作中，我们引入了个性化专家示范的新概念，针对每个单个智体或更广泛地说，团队中每种类型的智体进行了定制。这些示范仅涉及单智体行为以及每个智体如何实现个人目标，而不涉及任何合作元素，因此盲目模仿它们不会实现合作由于潜在冲突。为此，我们提出了一种方法，选择性地利用个性化专家示范作为指导，并允许智体学习协

    arXiv:2403.08936v1 Announce Type: cross  Abstract: Multi-Agent Reinforcement Learning (MARL) algorithms face the challenge of efficient exploration due to the exponential increase in the size of the joint state-action space. While demonstration-guided learning has proven beneficial in single-agent settings, its direct applicability to MARL is hindered by the practical difficulty of obtaining joint expert demonstrations. In this work, we introduce a novel concept of personalized expert demonstrations, tailored for each individual agent or, more broadly, each individual type of agent within a heterogeneous team. These demonstrations solely pertain to single-agent behaviors and how each agent can achieve personal goals without encompassing any cooperative elements, thus naively imitating them will not achieve cooperation due to potential conflicts. To this end, we propose an approach that selectively utilizes personalized expert demonstrations as guidance and allows agents to learn to coo
    
[^2]: 抓取、观察和放置：具有策略结构先验的高效未知物体重新排列

    Grasp, See and Place: Efficient Unknown Object Rearrangement with Policy Structure Prior

    [https://arxiv.org/abs/2402.15402](https://arxiv.org/abs/2402.15402)

    该论文提出了一种具有策略结构先验的高效未知物体重新排列系统，通过内外环的学习，实现了抓取、观察和放置在感知噪声中的优化。

    

    我们关注未知物体重新排列任务，即机器人应重新配置物体到由RGB-D图像指定的期望目标配置中。最近的研究通过整合基于学习的感知模块来探索未知物体重新排列系统。然而，它们对感知误差敏感，并且较少关注任务级性能。本文旨在开发一个有效的系统，用于在感知噪声中重新排列未知物体。我们在理论上揭示了噪声感知如何以分离的方式影响抓取和放置，并展示这样的分离结构不容易改善任务的最优性。我们提出了具有分离结构作为先验的GSP，一个双环系统。对于内环，我们学习主动观察策略以提高放置的感知。对于外环，我们学习一个抓取策略，意识到物体匹配和抓取能力。

    arXiv:2402.15402v1 Announce Type: cross  Abstract: We focus on the task of unknown object rearrangement, where a robot is supposed to re-configure the objects into a desired goal configuration specified by an RGB-D image. Recent works explore unknown object rearrangement systems by incorporating learning-based perception modules. However, they are sensitive to perception error, and pay less attention to task-level performance. In this paper, we aim to develop an effective system for unknown object rearrangement amidst perception noise. We theoretically reveal the noisy perception impacts grasp and place in a decoupled way, and show such a decoupled structure is non-trivial to improve task optimality. We propose GSP, a dual-loop system with the decoupled structure as prior. For the inner loop, we learn an active seeing policy for self-confident object matching to improve the perception of place. For the outer loop, we learn a grasp policy aware of object matching and grasp capability gu
    

