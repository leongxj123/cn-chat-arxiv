# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space](https://arxiv.org/abs/2404.01752) | 提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案 |
| [^2] | [Hallucination Detection in Foundation Models for Decision-Making: A Flexible Definition and Review of the State of the Art](https://arxiv.org/abs/2403.16527) | 基于基础模型的幻觉检测旨在填补现有规划者缺少的常识推理，以适用于超出分布任务的场景。 |
| [^3] | [NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning](https://arxiv.org/abs/2403.06828) | NeuPAN 是一种实时、高度准确、无地图、适用于各种机器人且对环境不变的机器人导航解决方案，最大的创新在于将原始点直接映射到学习到的多帧距离空间，并具有端到端模型学习的可解释性，从而实现了可证明的收敛。 |

# 详细

[^1]: 安全间隔RRT*用于连续空间中可扩展的多机器人路径规划

    Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space

    [https://arxiv.org/abs/2404.01752](https://arxiv.org/abs/2404.01752)

    提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案

    

    在本文中，我们考虑了在连续空间中解决多机器人路径规划（MRPP）问题以找到无冲突路径的问题。问题的困难主要来自两个因素。首先，涉及多个机器人会导致组合决策，使搜索空间呈指数级增长。其次，连续空间呈现出潜在无限的状态和动作。针对这个问题，我们提出了一个两级方法，低级是基于采样的规划器安全间隔RRT*（SI-RRT*），用于找到单个机器人的无碰撞轨迹。高级可以使用能够解决机器人间冲突的任何方法，我们采用了两种代表性方法，即优先规划（SI-CPP）和基于冲突的搜索（SI-CCBS）。实验结果表明，SI-RRT*能够快速找到高质量解决方案，并且所需的样本数量较少。SI-CPP表现出更好的可扩展性，而SI-CCBS产生

    arXiv:2404.01752v1 Announce Type: cross  Abstract: In this paper, we consider the problem of Multi-Robot Path Planning (MRPP) in continuous space to find conflict-free paths. The difficulty of the problem arises from two primary factors. First, the involvement of multiple robots leads to combinatorial decision-making, which escalates the search space exponentially. Second, the continuous space presents potentially infinite states and actions. For this problem, we propose a two-level approach where the low level is a sampling-based planner Safe Interval RRT* (SI-RRT*) that finds a collision-free trajectory for individual robots. The high level can use any method that can resolve inter-robot conflicts where we employ two representative methods that are Prioritized Planning (SI-CPP) and Conflict Based Search (SI-CCBS). Experimental results show that SI-RRT* can find a high-quality solution quickly with a small number of samples. SI-CPP exhibits improved scalability while SI-CCBS produces 
    
[^2]: 基于基础模型的幻觉检测：灵活定义与现有技术综述

    Hallucination Detection in Foundation Models for Decision-Making: A Flexible Definition and Review of the State of the Art

    [https://arxiv.org/abs/2403.16527](https://arxiv.org/abs/2403.16527)

    基于基础模型的幻觉检测旨在填补现有规划者缺少的常识推理，以适用于超出分布任务的场景。

    

    自主系统即将无处不在，从制造业的自主性到农业领域的机器人，从医疗助理到娱乐产业。大多数系统是通过模块化的子组件开发的，用于决策、规划和控制，这些组件可能是手工设计的，也可能是基于学习的。虽然现有方法在它们专门设计的情境下表现良好，但在测试时不可避免地会在罕见的、超出分布范围的场景下表现特别差。基于多个任务训练的基础模型的兴起，以及从各个领域采集的令人印象深刻的大型数据集，使研究人员相信这些模型可能提供现有规划者所缺乏的常识推理。研究人员认为，这种常识推理将弥合算法开发和部署之间的差距，适用于超出分布任务的情况。

    arXiv:2403.16527v1 Announce Type: new  Abstract: Autonomous systems are soon to be ubiquitous, from manufacturing autonomy to agricultural field robots, and from health care assistants to the entertainment industry. The majority of these systems are developed with modular sub-components for decision-making, planning, and control that may be hand-engineered or learning-based. While these existing approaches have been shown to perform well under the situations they were specifically designed for, they can perform especially poorly in rare, out-of-distribution scenarios that will undoubtedly arise at test-time. The rise of foundation models trained on multiple tasks with impressively large datasets from a variety of fields has led researchers to believe that these models may provide common sense reasoning that existing planners are missing. Researchers posit that this common sense reasoning will bridge the gap between algorithm development and deployment to out-of-distribution tasks, like
    
[^3]: NeuPAN:直接点机器人导航的端到端基于模型学习

    NeuPAN: Direct Point Robot Navigation with End-to-End Model-based Learning

    [https://arxiv.org/abs/2403.06828](https://arxiv.org/abs/2403.06828)

    NeuPAN 是一种实时、高度准确、无地图、适用于各种机器人且对环境不变的机器人导航解决方案，最大的创新在于将原始点直接映射到学习到的多帧距离空间，并具有端到端模型学习的可解释性，从而实现了可证明的收敛。

    

    在拥挤环境中对非全向机器人进行导航需要极其精确的感知和运动以避免碰撞。本文提出NeuPAN：一种实时、高度准确、无地图、适用于各种机器人，且对环境不变的机器人导航解决方案。NeuPAN采用紧耦合的感知-运动框架，与现有方法相比有两个关键创新：1）它直接将原始点映射到学习到的多帧距离空间，避免了从感知到控制的误差传播；2）从端到端基于模型学习的角度进行解释，实现了可证明的收敛。NeuPAN的关键在于利用插拔式（PnP）交替最小化传感器（PAN）网络解高维端到端数学模型，其中包含各种点级约束，使NeuPAN能够直接生成实时、端到端、物理可解释的运动。

    arXiv:2403.06828v1 Announce Type: cross  Abstract: Navigating a nonholonomic robot in a cluttered environment requires extremely accurate perception and locomotion for collision avoidance. This paper presents NeuPAN: a real-time, highly-accurate, map-free, robot-agnostic, and environment-invariant robot navigation solution. Leveraging a tightly-coupled perception-locomotion framework, NeuPAN has two key innovations compared to existing approaches: 1) it directly maps raw points to a learned multi-frame distance space, avoiding error propagation from perception to control; 2) it is interpretable from an end-to-end model-based learning perspective, enabling provable convergence. The crux of NeuPAN is to solve a high-dimensional end-to-end mathematical model with various point-level constraints using the plug-and-play (PnP) proximal alternating-minimization network (PAN) with neurons in the loop. This allows NeuPAN to generate real-time, end-to-end, physically-interpretable motions direct
    

