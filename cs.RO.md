# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DaCapo: Accelerating Continuous Learning in Autonomous Systems for Video Analytics](https://arxiv.org/abs/2403.14353) | 该论文提出了一种在自主系统中加速视频分析的持续学习方法，通过利用轻量级“学生”模型进行部署推理，利用更大的“教师”模型进行数据标记，实现对不断变化场景的持续自适应。 |
| [^2] | [Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling](https://arxiv.org/abs/2402.10211) | 分层状态空间模型（HiSS）是一种针对连续序列到序列建模的技术，它利用堆叠的结构化状态空间模型来进行预测。 |

# 详细

[^1]: DaCapo：加快自主系统在视频分析中的持续学习

    DaCapo: Accelerating Continuous Learning in Autonomous Systems for Video Analytics

    [https://arxiv.org/abs/2403.14353](https://arxiv.org/abs/2403.14353)

    该论文提出了一种在自主系统中加速视频分析的持续学习方法，通过利用轻量级“学生”模型进行部署推理，利用更大的“教师”模型进行数据标记，实现对不断变化场景的持续自适应。

    

    深度神经网络（DNN）视频分析对于自动驾驶车辆、无人机（UAV）和安防机器人等自主系统至关重要。然而，由于其有限的计算资源和电池功率，实际部署面临挑战。为了解决这些挑战，持续学习利用在部署（推理）中的轻量级“学生”模型，利用更大的“教师”模型对采样数据进行标记（标记），并不断重新训练学生模型以适应不断变化的场景。

    arXiv:2403.14353v1 Announce Type: cross  Abstract: Deep neural network (DNN) video analytics is crucial for autonomous systems such as self-driving vehicles, unmanned aerial vehicles (UAVs), and security robots. However, real-world deployment faces challenges due to their limited computational resources and battery power. To tackle these challenges, continuous learning exploits a lightweight "student" model at deployment (inference), leverages a larger "teacher" model for labeling sampled data (labeling), and continuously retrains the student model to adapt to changing scenarios (retraining). This paper highlights the limitations in state-of-the-art continuous learning systems: (1) they focus on computations for retraining, while overlooking the compute needs for inference and labeling, (2) they rely on power-hungry GPUs, unsuitable for battery-operated autonomous systems, and (3) they are located on a remote centralized server, intended for multi-tenant scenarios, again unsuitable for
    
[^2]: 针对连续序列到序列建模的分层状态空间模型

    Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling

    [https://arxiv.org/abs/2402.10211](https://arxiv.org/abs/2402.10211)

    分层状态空间模型（HiSS）是一种针对连续序列到序列建模的技术，它利用堆叠的结构化状态空间模型来进行预测。

    

    arXiv:2402.10211v1 公告类型：新的 摘要：从原始感知数据的序列推理是从医疗设备到机器人领域中普遍存在的问题。这些问题常常涉及使用长序列的原始传感器数据（例如磁力计，压阻器）来预测理想的物理量序列（例如力量，惯性测量）。虽然经典方法对于局部线性预测问题非常有效，但在使用实际传感器时往往表现不佳。这些传感器通常是非线性的，受到外界变量（例如振动）的影响，并且表现出数据相关漂移。对于许多问题来说，预测任务受到稀缺标记数据集的限制，因为获取地面真实标签需要昂贵的设备。在这项工作中，我们提出了分层状态空间模型（HiSS），这是一种概念上简单、全新的连续顺序预测技术。HiSS将结构化的状态空间模型堆叠在一起，以创建一个暂定的预测模型。

    arXiv:2402.10211v1 Announce Type: new  Abstract: Reasoning from sequences of raw sensory data is a ubiquitous problem across fields ranging from medical devices to robotics. These problems often involve using long sequences of raw sensor data (e.g. magnetometers, piezoresistors) to predict sequences of desirable physical quantities (e.g. force, inertial measurements). While classical approaches are powerful for locally-linear prediction problems, they often fall short when using real-world sensors. These sensors are typically non-linear, are affected by extraneous variables (e.g. vibration), and exhibit data-dependent drift. For many problems, the prediction task is exacerbated by small labeled datasets since obtaining ground-truth labels requires expensive equipment. In this work, we present Hierarchical State-Space Models (HiSS), a conceptually simple, new technique for continuous sequential prediction. HiSS stacks structured state-space models on top of each other to create a tempor
    

