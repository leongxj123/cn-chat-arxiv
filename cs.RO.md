# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CoBOS: Constraint-Based Online Scheduler for Human-Robot Collaboration](https://arxiv.org/abs/2403.18459) | CoBOS提出了一种新颖的在线基于约束的调度方法，在人机协作中实现了机器人对不确定事件的适应性，大大减轻了用户的压力，提高了工作效率。 |
| [^2] | [Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights.](http://arxiv.org/abs/2309.08731) | 本文提出了一种深度学习方法，通过学习到的ICP权重优化雷达-激光雷达的定位，从而改善了雷达测量对激光雷达地图的定位效果。这一方法在保持高质量地图定位性能的同时，提高了在降水和大雾等恶劣天气条件下的定位准确性。 |

# 详细

[^1]: CoBOS: 基于约束的人机协作在线调度器

    CoBOS: Constraint-Based Online Scheduler for Human-Robot Collaboration

    [https://arxiv.org/abs/2403.18459](https://arxiv.org/abs/2403.18459)

    CoBOS提出了一种新颖的在线基于约束的调度方法，在人机协作中实现了机器人对不确定事件的适应性，大大减轻了用户的压力，提高了工作效率。

    

    涉及人类和机器人的装配过程是具有挑战性的场景，因为个人活动和共享工作空间的访问必须协调。固定的机器人程序不允许偏离固定协议。在这样的过程中工作可能会让用户感到有压力，并导致行为无效或失败。我们提出了一种新颖的在线基于约束的调度方法，位于支持行为树的反应式执行控制框架中，名为CoBOS。这使得机器人能够适应延迟活动完成和活动选择（由人类）等不确定事件。用户将体验到较少的压力，因为机器人同事会调整其行为以最好地补充人类选择的活动，以完成共同任务。除了改善的工作条件，我们的算法还导致了效率的提高，即使在高度不确定的情况下也是如此。我们使用一个概率性的si来评估我们的算法

    arXiv:2403.18459v1 Announce Type: cross  Abstract: Assembly processes involving humans and robots are challenging scenarios because the individual activities and access to shared workspace have to be coordinated. Fixed robot programs leave no room to diverge from a fixed protocol. Working on such a process can be stressful for the user and lead to ineffective behavior or failure. We propose a novel approach of online constraint-based scheduling in a reactive execution control framework facilitating behavior trees called CoBOS. This allows the robot to adapt to uncertain events such as delayed activity completions and activity selection (by the human). The user will experience less stress as the robotic coworkers adapt their behavior to best complement the human-selected activities to complete the common task. In addition to the improved working conditions, our algorithm leads to increased efficiency, even in highly uncertain scenarios. We evaluate our algorithm using a probabilistic si
    
[^2]: 指引的方法：利用学习到的ICP权重改进雷达-激光雷达定位

    Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights. (arXiv:2309.08731v1 [cs.RO])

    [http://arxiv.org/abs/2309.08731](http://arxiv.org/abs/2309.08731)

    本文提出了一种深度学习方法，通过学习到的ICP权重优化雷达-激光雷达的定位，从而改善了雷达测量对激光雷达地图的定位效果。这一方法在保持高质量地图定位性能的同时，提高了在降水和大雾等恶劣天气条件下的定位准确性。

    

    本文提出了一种基于深度学习的新方法，用于改进雷达测量对激光雷达地图的定位。虽然目前定位的技术水平是将激光雷达数据与激光雷达地图进行匹配，但是雷达被认为是一种有前途的替代方法，因为它对降水和大雾等恶劣天气具有更强的韧性。为了利用现有的高质量激光雷达地图，同时在恶劣天气下保持性能，将雷达数据与激光雷达地图进行匹配具有重要意义。然而，由于雷达测量中存在的独特伪影，雷达-激光雷达定位一直难以达到与激光雷达-激光雷达系统相媲美的性能，使其无法用于自动驾驶。本工作在基于ICP的雷达-激光雷达定位系统基础上，包括一个学习的预处理步骤，根据高层次的扫描信息对雷达点进行加权。将经过验证的分析方法与学习到的权重相结合，减小了雷达定位中的误差。

    This paper presents a novel deep-learning-based approach to improve localizing radar measurements against lidar maps. Although the state of the art for localization is matching lidar data to lidar maps, radar has been considered as a promising alternative, as it is potentially more resilient against adverse weather such as precipitation and heavy fog. To make use of existing high-quality lidar maps, while maintaining performance in adverse weather, matching radar data to lidar maps is of interest. However, owing in part to the unique artefacts present in radar measurements, radar-lidar localization has struggled to achieve comparable performance to lidar-lidar systems, preventing it from being viable for autonomous driving. This work builds on an ICP-based radar-lidar localization system by including a learned preprocessing step that weights radar points based on high-level scan information. Combining a proven analytical approach with a learned weight reduces localization errors in rad
    

