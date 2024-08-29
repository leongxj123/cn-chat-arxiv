# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DeepMIF: Deep Monotonic Implicit Fields for Large-Scale LiDAR 3D Mapping](https://arxiv.org/abs/2403.17550) | 提出了DeepMIF，通过设计学习系统集成单调性损失，在大规模3D地图绘制中优化神经单调场，避免了LiDAR测量的嘈杂问题 |
| [^2] | [Scaling Learning based Policy Optimization for Temporal Tasks via Dropout](https://arxiv.org/abs/2403.15826) | 本文介绍了一种基于模型的方法用于训练在高度非线性环境中运行的自主智能体的反馈控制器，通过对任务进行形式化表述，实现对特定任务目标的定量满足语义，并利用前馈神经网络学习反馈控制器。 |
| [^3] | [Vid2Robot: End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers](https://arxiv.org/abs/2403.12943) | Vid2Robot提出了一种新颖的端到端视频条件化策略学习框架，通过交叉注意力机制融合视频特征和机器人状态，直接生成模仿所观察任务的动作。 |

# 详细

[^1]: DeepMIF: 用于大规模LiDAR 3D地图绘制的深度单调隐式场

    DeepMIF: Deep Monotonic Implicit Fields for Large-Scale LiDAR 3D Mapping

    [https://arxiv.org/abs/2403.17550](https://arxiv.org/abs/2403.17550)

    提出了DeepMIF，通过设计学习系统集成单调性损失，在大规模3D地图绘制中优化神经单调场，避免了LiDAR测量的嘈杂问题

    

    近年来，通过使用现代获取设备如LiDAR传感器，在感知真实大规模室外3D环境方面取得了显著进展。然而，它们在生成稠密、完整的3D场景方面存在固有限制。为解决这一问题，最近的基于学习的方法集成了神经隐式表示和可优化特征网格，以逼近3D场景的表面。然而，简单地沿原始LiDAR光线拟合样本会导致由于稀疏、互相矛盾的LiDAR测量的特性而产生嘈杂的3D绘图结果。相反，在这项工作中，我们不再精确拟合LiDAR数据，而是让网络优化在3D空间中定义的非度量单调隐式场。为适应我们的场，我们设计了一个学习系统，集成了一个单调性损失，使得能够优化神经单调场并利用了大规模3D地图绘制的最新进展。我们的算法...

    arXiv:2403.17550v1 Announce Type: cross  Abstract: Recently, significant progress has been achieved in sensing real large-scale outdoor 3D environments, particularly by using modern acquisition equipment such as LiDAR sensors. Unfortunately, they are fundamentally limited in their ability to produce dense, complete 3D scenes. To address this issue, recent learning-based methods integrate neural implicit representations and optimizable feature grids to approximate surfaces of 3D scenes. However, naively fitting samples along raw LiDAR rays leads to noisy 3D mapping results due to the nature of sparse, conflicting LiDAR measurements. Instead, in this work we depart from fitting LiDAR data exactly, instead letting the network optimize a non-metric monotonic implicit field defined in 3D space. To fit our field, we design a learning system integrating a monotonicity loss that enables optimizing neural monotonic fields and leverages recent progress in large-scale 3D mapping. Our algorithm ac
    
[^2]: 通过Dropout对时间任务进行比例学习的策略优化扩展

    Scaling Learning based Policy Optimization for Temporal Tasks via Dropout

    [https://arxiv.org/abs/2403.15826](https://arxiv.org/abs/2403.15826)

    本文介绍了一种基于模型的方法用于训练在高度非线性环境中运行的自主智能体的反馈控制器，通过对任务进行形式化表述，实现对特定任务目标的定量满足语义，并利用前馈神经网络学习反馈控制器。

    

    本文介绍了一种基于模型的方法，用于训练在高度非线性环境中运行的自主智能体的反馈控制器。我们希望经过训练的策略能够确保该智能体满足特定的任务目标，这些目标以离散时间信号时间逻辑（DT-STL）表示。通过将任务重新表述为形式化框架（如DT-STL），一个优势是允许定量满足语义。换句话说，给定一个轨迹和一个DT-STL公式，我们可以计算鲁棒性，这可以解释为轨迹与满足该公式的轨迹集之间的近似有符号距离。我们利用反馈控制器，并假设使用前馈神经网络来学习这些反馈控制器。我们展示了这个学习问题与训练递归神经网络（RNNs）类似的地方，其中递归单元的数量与智能体的时间视野成比例。

    arXiv:2403.15826v1 Announce Type: cross  Abstract: This paper introduces a model-based approach for training feedback controllers for an autonomous agent operating in a highly nonlinear environment. We desire the trained policy to ensure that the agent satisfies specific task objectives, expressed in discrete-time Signal Temporal Logic (DT-STL). One advantage for reformulation of a task via formal frameworks, like DT-STL, is that it permits quantitative satisfaction semantics. In other words, given a trajectory and a DT-STL formula, we can compute the robustness, which can be interpreted as an approximate signed distance between the trajectory and the set of trajectories satisfying the formula. We utilize feedback controllers, and we assume a feed forward neural network for learning these feedback controllers. We show how this learning problem is similar to training recurrent neural networks (RNNs), where the number of recurrent units is proportional to the temporal horizon of the agen
    
[^3]: Vid2Robot：基于视频条件化策略学习的端到端交叉注意力变换器

    Vid2Robot: End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers

    [https://arxiv.org/abs/2403.12943](https://arxiv.org/abs/2403.12943)

    Vid2Robot提出了一种新颖的端到端视频条件化策略学习框架，通过交叉注意力机制融合视频特征和机器人状态，直接生成模仿所观察任务的动作。

    

    尽管大规模机器人系统通常依赖文本指令进行任务，但这项工作探索了一种不同的方法：机器人能否直接从观察人类推断任务？这种转变要求机器人能够解码人类意图，并将其转化为可在其物理约束和环境内执行的动作。我们引入了Vid2Robot，这是一种新颖的面向机器人的端到端基于视频的学习框架。给定一个操作任务的视频演示和当前的视觉观察，Vid2Robot直接生成机器人动作。这是通过在大规模人类视频和机器人轨迹数据集上训练的统一表示模型实现的。该模型利用交叉注意力机制来融合提示视频特征与机器人的当前状态，并生成模仿所观察任务的适当动作。为了进一步提高策略性能，我们提出了辅助对比损失，以增强对齐

    arXiv:2403.12943v1 Announce Type: cross  Abstract: While large-scale robotic systems typically rely on textual instructions for tasks, this work explores a different approach: can robots infer the task directly from observing humans? This shift necessitates the robot's ability to decode human intent and translate it into executable actions within its physical constraints and environment. We introduce Vid2Robot, a novel end-to-end video-based learning framework for robots. Given a video demonstration of a manipulation task and current visual observations, Vid2Robot directly produces robot actions. This is achieved through a unified representation model trained on a large dataset of human video and robot trajectory. The model leverages cross-attention mechanisms to fuse prompt video features to the robot's current state and generate appropriate actions that mimic the observed task. To further improve policy performance, we propose auxiliary contrastive losses that enhance the alignment b
    

