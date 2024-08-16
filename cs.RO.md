# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [WATonoBus: An All Weather Autonomous Shuttle](https://arxiv.org/abs/2312.00938) | 提出了一种考虑恶劣天气的多模块和模块化系统架构，在WATonoBus平台上进行了实际测试，证明其能够解决全天候自动驾驶车辆面临的挑战 |
| [^2] | [Simultaneous Task Allocation and Planning for Multi-Robots under Hierarchical Temporal Logic Specifications.](http://arxiv.org/abs/2401.04003) | 该论文介绍了在多机器人系统中，利用层次化时间逻辑规范实现同时的任务分配和规划的方法。通过引入层次化结构到LTL规范中，该方法更具表达能力。采用基于搜索的方法来综合多机器人系统的计划，将搜索空间拆分为松散相互连接的子空间，以便更高效地进行任务分配和规划。 |
| [^3] | [End-to-end Autonomous Driving: Challenges and Frontiers.](http://arxiv.org/abs/2306.16927) | 这项研究调查了端到端自动驾驶领域中的关键挑战和未来趋势，包括多模态、可解释性、因果混淆、鲁棒性和世界模型等。通过联合特征优化感知和规划，端到端系统在感知和规划上获得了更好的效果。 |
| [^4] | [Safe Imitation Learning of Nonlinear Model Predictive Control for Flexible Robots.](http://arxiv.org/abs/2212.02941) | 本文提出了一种使用模仿学习和预测安全过滤器进行非线性模型预测控制(NMPC)的安全近似的框架，以实现对柔性机器人的快速控制。与NMPC相比，在保证安全约束的情况下，我们的框架在计算时间上改善了8倍以上。 |

# 详细

[^1]: WATonoBus：一种全天候自动巡航车

    WATonoBus: An All Weather Autonomous Shuttle

    [https://arxiv.org/abs/2312.00938](https://arxiv.org/abs/2312.00938)

    提出了一种考虑恶劣天气的多模块和模块化系统架构，在WATonoBus平台上进行了实际测试，证明其能够解决全天候自动驾驶车辆面临的挑战

    

    自动驾驶车辆在全天候运行中面临显著挑战，涵盖了从感知和决策到路径规划和控制的各个模块。复杂性源于需要解决像雨、雪和雾等恶劣天气条件在自主性堆栈中的问题。传统的基于模型和单模块方法通常缺乏与上游或下游任务的整体集成。我们通过提出一个考虑恶劣天气的多模块和模块化系统架构来解决这个问题，涵盖了从感知水平到决策和安全监测的各个方面，例如覆盖雪的路缘检测。通过在WATonoBus平台上每周日常服务近一年，我们展示了我们提出的方法能够解决恶劣天气条件，并从运营中观察到的极端情况中获得宝贵的经验教训。

    arXiv:2312.00938v1 Announce Type: cross  Abstract: Autonomous vehicle all-weather operation poses significant challenges, encompassing modules from perception and decision-making to path planning and control. The complexity arises from the need to address adverse weather conditions like rain, snow, and fog across the autonomy stack. Conventional model-based and single-module approaches often lack holistic integration with upstream or downstream tasks. We tackle this problem by proposing a multi-module and modular system architecture with considerations for adverse weather across the perception level, through features such as snow covered curb detection, to decision-making and safety monitoring. Through daily weekday service on the WATonoBus platform for almost a year, we demonstrate that our proposed approach is capable of addressing adverse weather conditions and provide valuable learning from edge cases observed during operation.
    
[^2]: 多机器人在层次化时间逻辑规范下的任务分配和规划

    Simultaneous Task Allocation and Planning for Multi-Robots under Hierarchical Temporal Logic Specifications. (arXiv:2401.04003v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2401.04003](http://arxiv.org/abs/2401.04003)

    该论文介绍了在多机器人系统中，利用层次化时间逻辑规范实现同时的任务分配和规划的方法。通过引入层次化结构到LTL规范中，该方法更具表达能力。采用基于搜索的方法来综合多机器人系统的计划，将搜索空间拆分为松散相互连接的子空间，以便更高效地进行任务分配和规划。

    

    过去关于机器人规划与时间逻辑规范的研究，特别是线性时间逻辑（LTL），主要是基于针对个体或群体机器人的单一公式。但随着任务复杂性的增加，LTL公式不可避免地变得冗长，使解释和规范生成变得复杂，同时还对规划器的计算能力造成压力。通过利用任务的内在结构，我们引入了一种层次化结构到具有语法和语义需求的LTL规范中，并证明它们比扁平规范更具表达能力。其次，我们采用基于搜索的方法来综合多机器人系统的计划，实现同时的任务分配和规划。搜索空间由松散相互连接的子空间近似表示，每个子空间对应一个LTL规范。搜索主要受限于单个子空间，根据特定条件转移到另一个子空间。

    Past research into robotic planning with temporal logic specifications, notably Linear Temporal Logic (LTL), was largely based on singular formulas for individual or groups of robots. But with increasing task complexity, LTL formulas unavoidably grow lengthy, complicating interpretation and specification generation, and straining the computational capacities of the planners. By leveraging the intrinsic structure of tasks, we introduced a hierarchical structure to LTL specifications with requirements on syntax and semantics, and proved that they are more expressive than their flat counterparts. Second, we employ a search-based approach to synthesize plans for a multi-robot system, accomplishing simultaneous task allocation and planning. The search space is approximated by loosely interconnected sub-spaces, with each sub-space corresponding to one LTL specification. The search is predominantly confined to a single sub-space, transitioning to another sub-space under certain conditions, de
    
[^3]: 线束自动驾驶：挑战与前景

    End-to-end Autonomous Driving: Challenges and Frontiers. (arXiv:2306.16927v1 [cs.RO])

    [http://arxiv.org/abs/2306.16927](http://arxiv.org/abs/2306.16927)

    这项研究调查了端到端自动驾驶领域中的关键挑战和未来趋势，包括多模态、可解释性、因果混淆、鲁棒性和世界模型等。通过联合特征优化感知和规划，端到端系统在感知和规划上获得了更好的效果。

    

    自动驾驶领域正在迅速发展，越来越多的方法采用端到端算法框架，利用原始传感器输入生成车辆运动计划，而不是专注于诸如检测和运动预测等单个任务。与模块化流水线相比，端到端系统通过联合特征优化感知和规划来获益。这一领域因大规模数据集的可用性、闭环评估以及自动驾驶算法在挑战性场景中的有效执行所需的需求而蓬勃发展。在本调查中，我们全面分析了250多篇论文，涵盖了端到端自动驾驶的动机、路线图、方法论、挑战和未来趋势。我们深入探讨了多模态、可解释性、因果混淆、鲁棒性和世界模型等几个关键挑战。此外，我们还讨论了基础技术的最新进展。

    The autonomous driving community has witnessed a rapid growth in approaches that embrace an end-to-end algorithm framework, utilizing raw sensor input to generate vehicle motion plans, instead of concentrating on individual tasks such as detection and motion prediction. End-to-end systems, in comparison to modular pipelines, benefit from joint feature optimization for perception and planning. This field has flourished due to the availability of large-scale datasets, closed-loop evaluation, and the increasing need for autonomous driving algorithms to perform effectively in challenging scenarios. In this survey, we provide a comprehensive analysis of more than 250 papers, covering the motivation, roadmap, methodology, challenges, and future trends in end-to-end autonomous driving. We delve into several critical challenges, including multi-modality, interpretability, causal confusion, robustness, and world models, amongst others. Additionally, we discuss current advancements in foundation
    
[^4]: 非线性模型预测控制在柔性机器人中的安全模仿学习

    Safe Imitation Learning of Nonlinear Model Predictive Control for Flexible Robots. (arXiv:2212.02941v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2212.02941](http://arxiv.org/abs/2212.02941)

    本文提出了一种使用模仿学习和预测安全过滤器进行非线性模型预测控制(NMPC)的安全近似的框架，以实现对柔性机器人的快速控制。与NMPC相比，在保证安全约束的情况下，我们的框架在计算时间上改善了8倍以上。

    

    柔性机器人可以解决一些工业领域的主要挑战，如实现固有安全的人机协作和实现更高的负载重量比。然而，由于其复杂的动力学特性，包括振荡行为和高维状态空间，控制柔性机器人非常复杂。非线性模型预测控制(NMPC)能够有效控制此类机器人，但其计算需求较高常常限制其在实时场景中的应用。为了实现对柔性机器人的快速控制，我们提出了一种使用模仿学习和预测安全过滤器进行NMPC的安全近似的框架。我们的框架显著减少了计算时间，同时在性能上略有损失。与NMPC相比，在模拟中控制一个三维柔性机械臂时，我们的框架在计算时间上改善了8倍以上，同时保证了安全约束。值得注意的是，我们的方法优于传统的强化学习方法。

    Flexible robots may overcome some of the industry's major challenges, such as enabling intrinsically safe human-robot collaboration and achieving a higher load-to-mass ratio. However, controlling flexible robots is complicated due to their complex dynamics, which include oscillatory behavior and a high-dimensional state space. NMPC offers an effective means to control such robots, but its extensive computational demands often limit its application in real-time scenarios. To enable fast control of flexible robots, we propose a framework for a safe approximation of NMPC using imitation learning and a predictive safety filter. Our framework significantly reduces computation time while incurring a slight loss in performance. Compared to NMPC, our framework shows more than a eightfold improvement in computation time when controlling a three-dimensional flexible robot arm in simulation, all while guaranteeing safety constraints. Notably, our approach outperforms conventional reinforcement le
    

