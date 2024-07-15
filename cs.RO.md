# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Boundary State Generation for Testing and Improvement of Autonomous Driving Systems.](http://arxiv.org/abs/2307.10590) | 该论文介绍了一种新的自动驾驶系统测试生成器（GenBo），它通过在无故障环境中变异自动驾驶车辆的驾驶条件来生成边界状态对，以解决现有测试方法中存在的问题。 |
| [^2] | [Kinematic Data-Based Action Segmentation for Surgical Applications.](http://arxiv.org/abs/2303.07814) | 本文提出了两种多阶段体系结构和两种数据增强技术，专门用于基于运动学数据的行动分割。同时，作者在三个手术缝合任务数据集上对模型进行了评估。 |

# 详细

[^1]: 自动驾驶系统测试与改进的边界状态生成

    Boundary State Generation for Testing and Improvement of Autonomous Driving Systems. (arXiv:2307.10590v1 [cs.SE])

    [http://arxiv.org/abs/2307.10590](http://arxiv.org/abs/2307.10590)

    该论文介绍了一种新的自动驾驶系统测试生成器（GenBo），它通过在无故障环境中变异自动驾驶车辆的驾驶条件来生成边界状态对，以解决现有测试方法中存在的问题。

    

    最近深度神经网络（DNN）和传感器技术的进展使得自动驾驶系统（ADS）具有了越来越高的自主性。然而，评估其可靠性仍然是一个关键问题。目前的ADS测试方法修改模拟驾驶环境的可控属性，直到ADS出现问题。这种方法有两个主要缺点：（1）对模拟环境的修改可能不容易转移到实际测试环境（例如改变道路形状）；（2）即使ADS在某些环境中成功，这些环境实例也会被丢弃，尽管它们可能包含ADS可能出现问题的潜在驾驶条件。本文提出了一种新的ADS测试生成器——GenBo（GENerator of BOundary state pairs）。GenBo在一个无故障环境实例中变异自动驾驶车辆的驾驶条件（位置，速度和方向），并有效地生成可边界化的状态对。

    Recent advances in Deep Neural Networks (DNNs) and sensor technologies are enabling autonomous driving systems (ADSs) with an ever-increasing level of autonomy. However, assessing their dependability remains a critical concern. State-of-the-art ADS testing approaches modify the controllable attributes of a simulated driving environment until the ADS misbehaves. Such approaches have two main drawbacks: (1) modifications to the simulated environment might not be easily transferable to the in-field test setting (e.g., changing the road shape); (2) environment instances in which the ADS is successful are discarded, despite the possibility that they could contain hidden driving conditions in which the ADS may misbehave.  In this paper, we present GenBo (GENerator of BOundary state pairs), a novel test generator for ADS testing. GenBo mutates the driving conditions of the ego vehicle (position, velocity and orientation), collected in a failure-free environment instance, and efficiently gener
    
[^2]: 基于运动学数据的手术行为切分

    Kinematic Data-Based Action Segmentation for Surgical Applications. (arXiv:2303.07814v1 [cs.CV])

    [http://arxiv.org/abs/2303.07814](http://arxiv.org/abs/2303.07814)

    本文提出了两种多阶段体系结构和两种数据增强技术，专门用于基于运动学数据的行动分割。同时，作者在三个手术缝合任务数据集上对模型进行了评估。

    

    行动切分是高级流程分析中的一个挑战性任务，通常在视频或从各种传感器获取的运动学数据上执行。在手术过程中，行动切分对于工作流分析算法至关重要。本文提出了两个与运动学数据相关的行动分割方面的贡献。首先，我们介绍了两种多阶段体系结构，MS-TCN-BiLSTM和MS-TCN-BiGRU，专门设计用于运动学数据。 这些体系结构由具有阶内规则化和双向LSTM或GRU的细化阶段的预测生成器组成。其次，我们提出了两种新的数据增强技术，World Frame Rotation和Horizontal-Flip，利用运动学数据的强几何结构来提高算法性能和鲁棒性。我们在三个手术缝合任务数据集上评估了我们的模型：可变组织模拟（VTS）数据集和新推出的肠道修复模拟（BRS）数据集。

    Action segmentation is a challenging task in high-level process analysis, typically performed on video or kinematic data obtained from various sensors. In the context of surgical procedures, action segmentation is critical for workflow analysis algorithms. This work presents two contributions related to action segmentation on kinematic data. Firstly, we introduce two multi-stage architectures, MS-TCN-BiLSTM and MS-TCN-BiGRU, specifically designed for kinematic data. The architectures consist of a prediction generator with intra-stage regularization and Bidirectional LSTM or GRU-based refinement stages. Secondly, we propose two new data augmentation techniques, World Frame Rotation and Horizontal-Flip, which utilize the strong geometric structure of kinematic data to improve algorithm performance and robustness. We evaluate our models on three datasets of surgical suturing tasks: the Variable Tissue Simulation (VTS) Dataset and the newly introduced Bowel Repair Simulation (BRS) Dataset,
    

