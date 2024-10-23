# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving](https://arxiv.org/abs/2404.01596) | PhysORD是一种神经符号方法，将物理定律融入神经模型中，显著提高了在越野驾驶中的运动预测泛化能力。 |
| [^2] | [TopoNav: Topological Navigation for Efficient Exploration in Sparse Reward Environments](https://arxiv.org/abs/2402.04061) | TopoNav是一种拓扑导航框架，它通过主动拓扑映射、内部奖励机制和层次化目标优先级的组合来实现在稀疏奖励环境中高效探索。 |
| [^3] | [Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning](https://arxiv.org/abs/2402.02500) | 通过广泛实验发现基于点云的方法在机器人学习中表现出更好的性能，特别是在各种预训练和泛化任务中。结果表明，点云观测模态对于复杂机器人任务是有价值的。 |

# 详细

[^1]: PhysORD：一种神经符号方法用于越野驾驶中注入物理学的运动预测

    PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving

    [https://arxiv.org/abs/2404.01596](https://arxiv.org/abs/2404.01596)

    PhysORD是一种神经符号方法，将物理定律融入神经模型中，显著提高了在越野驾驶中的运动预测泛化能力。

    

    运动预测对于自主越野驾驶至关重要，但与在道路上驾驶相比，它面临着更多挑战，主要是由于车辆与地形之间复杂的相互作用。传统的基于物理的方法在准确建模动态系统和外部干扰方面遇到困难。相反，基于数据驱动的神经网络需要大量数据集，并且难以明确捕捉基本的物理定律，这很容易导致泛化能力差。通过融合这两种方法的优势，神经符号方法提出了一个有前途的方向。这些方法将物理定律嵌入神经模型中，可能显著提高泛化能力。然而，以往的研究都没有在现实世界的越野驾驶环境中进行评估。为了弥合这一差距，我们提出 PhysORD，这是一种神经符号方法，集成了守恒定律，即欧拉-拉格朗日方程。

    arXiv:2404.01596v1 Announce Type: cross  Abstract: Motion prediction is critical for autonomous off-road driving, however, it presents significantly more challenges than on-road driving because of the complex interaction between the vehicle and the terrain. Traditional physics-based approaches encounter difficulties in accurately modeling dynamic systems and external disturbance. In contrast, data-driven neural networks require extensive datasets and struggle with explicitly capturing the fundamental physical laws, which can easily lead to poor generalization. By merging the advantages of both methods, neuro-symbolic approaches present a promising direction. These methods embed physical laws into neural models, potentially significantly improving generalization capabilities. However, no prior works were evaluated in real-world settings for off-road driving. To bridge this gap, we present PhysORD, a neural-symbolic approach integrating the conservation law, i.e., the Euler-Lagrange equa
    
[^2]: TopoNav：节约奖励环境中高效探索的拓扑导航

    TopoNav: Topological Navigation for Efficient Exploration in Sparse Reward Environments

    [https://arxiv.org/abs/2402.04061](https://arxiv.org/abs/2402.04061)

    TopoNav是一种拓扑导航框架，它通过主动拓扑映射、内部奖励机制和层次化目标优先级的组合来实现在稀疏奖励环境中高效探索。

    

    自动化机器人在未知区域的探索面临着一个重大挑战——在没有先前地图和有限外部反馈的情况下有效导航。在稀疏奖励环境中，这个挑战更加严峻，传统的探索技术往往失败。本文介绍了TopoNav，一种全新的框架，使机器人能够克服这些限制，实现高效、适应性强且目标导向的探索。TopoNav的基本构建模块是主动拓扑映射、内部奖励机制和层次化目标优先级。在探索过程中，TopoNav构建了动态拓扑地图，捕获关键位置和路径。它利用内部奖励来指导机器人朝着地图中指定的子目标前进，促进在稀疏奖励环境中的结构化探索。为了确保高效导航，TopoNav采用了分层目标驱动的主动拓扑框架，使机器人能够优先考虑最紧急的目标。

    Autonomous robots exploring unknown areas face a significant challenge -- navigating effectively without prior maps and with limited external feedback. This challenge intensifies in sparse reward environments, where traditional exploration techniques often fail. In this paper, we introduce TopoNav, a novel framework that empowers robots to overcome these constraints and achieve efficient, adaptable, and goal-oriented exploration. TopoNav's fundamental building blocks are active topological mapping, intrinsic reward mechanisms, and hierarchical objective prioritization. Throughout its exploration, TopoNav constructs a dynamic topological map that captures key locations and pathways. It utilizes intrinsic rewards to guide the robot towards designated sub-goals within this map, fostering structured exploration even in sparse reward settings. To ensure efficient navigation, TopoNav employs the Hierarchical Objective-Driven Active Topologies framework, enabling the robot to prioritize immed
    
[^3]: 点云问题:重新思考不同观测空间对机器人学习的影响

    Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning

    [https://arxiv.org/abs/2402.02500](https://arxiv.org/abs/2402.02500)

    通过广泛实验发现基于点云的方法在机器人学习中表现出更好的性能，特别是在各种预训练和泛化任务中。结果表明，点云观测模态对于复杂机器人任务是有价值的。

    

    在这项研究中，我们探讨了不同观测空间对机器人学习的影响，重点关注了三种主要模态：RGB，RGB-D和点云。通过在超过17个不同接触丰富的操作任务上进行广泛实验，涉及两个基准和仿真器，我们观察到了一个显著的趋势：基于点云的方法，即使是最简单的设计，通常在性能上超过了其RGB和RGB-D的对应物。这在从头开始训练和利用预训练的两种情况下都是一致的。此外，我们的研究结果表明，点云观测在相机视角、照明条件、噪声水平和背景外观等各种几何和视觉线索方面，都能提高策略零样本泛化能力。研究结果表明，三维点云是复杂机器人任务中有价值的观测模态。我们将公开所有的代码和检查点，希望我们的观点能帮助解决问题。

    In this study, we explore the influence of different observation spaces on robot learning, focusing on three predominant modalities: RGB, RGB-D, and point cloud. Through extensive experimentation on over 17 varied contact-rich manipulation tasks, conducted across two benchmarks and simulators, we have observed a notable trend: point cloud-based methods, even those with the simplest designs, frequently surpass their RGB and RGB-D counterparts in performance. This remains consistent in both scenarios: training from scratch and utilizing pretraining. Furthermore, our findings indicate that point cloud observations lead to improved policy zero-shot generalization in relation to various geometry and visual clues, including camera viewpoints, lighting conditions, noise levels and background appearance. The outcomes suggest that 3D point cloud is a valuable observation modality for intricate robotic tasks. We will open-source all our codes and checkpoints, hoping that our insights can help de
    

