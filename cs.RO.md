# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation](https://arxiv.org/abs/2403.03949) | 该论文提出了一种名为RialTo的系统，通过在“数字孪生”模拟环境中进行强化学习来稳健化真实世界的模仿学习策略，以实现在不需要大量不安全真实世界数据采集或广泛人类监督的情况下学习性能优越、稳健的策略。 |
| [^2] | [Grasp, See and Place: Efficient Unknown Object Rearrangement with Policy Structure Prior](https://arxiv.org/abs/2402.15402) | 该论文提出了一种具有策略结构先验的高效未知物体重新排列系统，通过内外环的学习，实现了抓取、观察和放置在感知噪声中的优化。 |
| [^3] | [LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery.](http://arxiv.org/abs/2311.02058) | LOTUS是一种持续模仿学习算法，通过无监督技能发现，使得机器人能够在其整个寿命中持续学习解决新的操作任务。该算法通过构建技能库，并使用元控制器灵活组合技能来提高成功率，在实验中表现出优越的知识传递能力。 |

# 详细

[^1]: 通过模拟调和现实：一种用于稳健操作的实-模-实方法

    Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation

    [https://arxiv.org/abs/2403.03949](https://arxiv.org/abs/2403.03949)

    该论文提出了一种名为RialTo的系统，通过在“数字孪生”模拟环境中进行强化学习来稳健化真实世界的模仿学习策略，以实现在不需要大量不安全真实世界数据采集或广泛人类监督的情况下学习性能优越、稳健的策略。

    

    仿真学习方法需要大量人类监督来学习对物体姿势变化、物理干扰和视觉扰动鲁棒的策略。另一方面，强化学习可以自主探索环境以学习稳健行为，但可能需要大量不安全的真实世界数据采集。为了在没有不安全真实世界数据采集或广泛人类监督的负担下学习性能优越、稳健的策略，我们提出了RialTo，一个通过在即将从少量真实世界数据构建的“数字孪生”模拟环境中进行强化学习来稳健化真实世界的模仿学习策略的系统。为了实现这种实-模-实流水线，RialTo提出了一个易于使用的接口，用于快速扫描和构建真实世界环境的数字孪生。我们还引入了一种新颖的“反向提炼”过程，用于给真实世界演示带来

    arXiv:2403.03949v1 Announce Type: cross  Abstract: Imitation learning methods need significant human supervision to learn policies robust to changes in object poses, physical disturbances, and visual distractors. Reinforcement learning, on the other hand, can explore the environment autonomously to learn robust behaviors but may require impractical amounts of unsafe real-world data collection. To learn performant, robust policies without the burden of unsafe real-world data collection or extensive human supervision, we propose RialTo, a system for robustifying real-world imitation learning policies via reinforcement learning in "digital twin" simulation environments constructed on the fly from small amounts of real-world data. To enable this real-to-sim-to-real pipeline, RialTo proposes an easy-to-use interface for quickly scanning and constructing digital twins of real-world environments. We also introduce a novel "inverse distillation" procedure for bringing real-world demonstrations
    
[^2]: 抓取、观察和放置：具有策略结构先验的高效未知物体重新排列

    Grasp, See and Place: Efficient Unknown Object Rearrangement with Policy Structure Prior

    [https://arxiv.org/abs/2402.15402](https://arxiv.org/abs/2402.15402)

    该论文提出了一种具有策略结构先验的高效未知物体重新排列系统，通过内外环的学习，实现了抓取、观察和放置在感知噪声中的优化。

    

    我们关注未知物体重新排列任务，即机器人应重新配置物体到由RGB-D图像指定的期望目标配置中。最近的研究通过整合基于学习的感知模块来探索未知物体重新排列系统。然而，它们对感知误差敏感，并且较少关注任务级性能。本文旨在开发一个有效的系统，用于在感知噪声中重新排列未知物体。我们在理论上揭示了噪声感知如何以分离的方式影响抓取和放置，并展示这样的分离结构不容易改善任务的最优性。我们提出了具有分离结构作为先验的GSP，一个双环系统。对于内环，我们学习主动观察策略以提高放置的感知。对于外环，我们学习一个抓取策略，意识到物体匹配和抓取能力。

    arXiv:2402.15402v1 Announce Type: cross  Abstract: We focus on the task of unknown object rearrangement, where a robot is supposed to re-configure the objects into a desired goal configuration specified by an RGB-D image. Recent works explore unknown object rearrangement systems by incorporating learning-based perception modules. However, they are sensitive to perception error, and pay less attention to task-level performance. In this paper, we aim to develop an effective system for unknown object rearrangement amidst perception noise. We theoretically reveal the noisy perception impacts grasp and place in a decoupled way, and show such a decoupled structure is non-trivial to improve task optimality. We propose GSP, a dual-loop system with the decoupled structure as prior. For the inner loop, we learn an active seeing policy for self-confident object matching to improve the perception of place. For the outer loop, we learn a grasp policy aware of object matching and grasp capability gu
    
[^3]: LOTUS：通过无监督技能发现的持续模仿学习，用于机器人操作

    LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery. (arXiv:2311.02058v1 [cs.RO])

    [http://arxiv.org/abs/2311.02058](http://arxiv.org/abs/2311.02058)

    LOTUS是一种持续模仿学习算法，通过无监督技能发现，使得机器人能够在其整个寿命中持续学习解决新的操作任务。该算法通过构建技能库，并使用元控制器灵活组合技能来提高成功率，在实验中表现出优越的知识传递能力。

    

    我们介绍了一种名为LOTUS的持续模仿学习算法，它使得物理机器人能够在其整个寿命中持续而高效地学习解决新的操作任务。LOTUS的核心思想是通过一系列新任务的少量人类演示构建一个不断增长的技能库。LOTUS首先使用开放词汇视觉模型进行持续技能发现过程，该模型从未分段的演示中提取重复出现的技能模式。持续技能发现更新现有技能以避免对以前任务的灾难性遗忘，并添加新技能以解决新任务。LOTUS训练一个元控制器，在终身学习过程中灵活地组合各种技能来解决基于视觉的操作任务。我们的综合实验证明，与先前方法相比，LOTUS在成功率上超过了现有技术基线方法11％以上，显示了其优越的知识传递能力。

    We introduce LOTUS, a continual imitation learning algorithm that empowers a physical robot to continuously and efficiently learn to solve new manipulation tasks throughout its lifespan. The core idea behind LOTUS is constructing an ever-growing skill library from a sequence of new tasks with a small number of human demonstrations. LOTUS starts with a continual skill discovery process using an open-vocabulary vision model, which extracts skills as recurring patterns presented in unsegmented demonstrations. Continual skill discovery updates existing skills to avoid catastrophic forgetting of previous tasks and adds new skills to solve novel tasks. LOTUS trains a meta-controller that flexibly composes various skills to tackle vision-based manipulation tasks in the lifelong learning process. Our comprehensive experiments show that LOTUS outperforms state-of-the-art baselines by over 11% in success rate, showing its superior knowledge transfer ability compared to prior methods. More result
    

