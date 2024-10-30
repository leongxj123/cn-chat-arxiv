# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation](https://arxiv.org/abs/2403.03949) | 该论文提出了一种名为RialTo的系统，通过在“数字孪生”模拟环境中进行强化学习来稳健化真实世界的模仿学习策略，以实现在不需要大量不安全真实世界数据采集或广泛人类监督的情况下学习性能优越、稳健的策略。 |

# 详细

[^1]: 通过模拟调和现实：一种用于稳健操作的实-模-实方法

    Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation

    [https://arxiv.org/abs/2403.03949](https://arxiv.org/abs/2403.03949)

    该论文提出了一种名为RialTo的系统，通过在“数字孪生”模拟环境中进行强化学习来稳健化真实世界的模仿学习策略，以实现在不需要大量不安全真实世界数据采集或广泛人类监督的情况下学习性能优越、稳健的策略。

    

    仿真学习方法需要大量人类监督来学习对物体姿势变化、物理干扰和视觉扰动鲁棒的策略。另一方面，强化学习可以自主探索环境以学习稳健行为，但可能需要大量不安全的真实世界数据采集。为了在没有不安全真实世界数据采集或广泛人类监督的负担下学习性能优越、稳健的策略，我们提出了RialTo，一个通过在即将从少量真实世界数据构建的“数字孪生”模拟环境中进行强化学习来稳健化真实世界的模仿学习策略的系统。为了实现这种实-模-实流水线，RialTo提出了一个易于使用的接口，用于快速扫描和构建真实世界环境的数字孪生。我们还引入了一种新颖的“反向提炼”过程，用于给真实世界演示带来

    arXiv:2403.03949v1 Announce Type: cross  Abstract: Imitation learning methods need significant human supervision to learn policies robust to changes in object poses, physical disturbances, and visual distractors. Reinforcement learning, on the other hand, can explore the environment autonomously to learn robust behaviors but may require impractical amounts of unsafe real-world data collection. To learn performant, robust policies without the burden of unsafe real-world data collection or extensive human supervision, we propose RialTo, a system for robustifying real-world imitation learning policies via reinforcement learning in "digital twin" simulation environments constructed on the fly from small amounts of real-world data. To enable this real-to-sim-to-real pipeline, RialTo proposes an easy-to-use interface for quickly scanning and constructing digital twins of real-world environments. We also introduce a novel "inverse distillation" procedure for bringing real-world demonstrations
    

