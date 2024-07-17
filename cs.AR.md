# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DaCapo: Accelerating Continuous Learning in Autonomous Systems for Video Analytics](https://arxiv.org/abs/2403.14353) | 该论文提出了一种在自主系统中加速视频分析的持续学习方法，通过利用轻量级“学生”模型进行部署推理，利用更大的“教师”模型进行数据标记，实现对不断变化场景的持续自适应。 |

# 详细

[^1]: DaCapo：加快自主系统在视频分析中的持续学习

    DaCapo: Accelerating Continuous Learning in Autonomous Systems for Video Analytics

    [https://arxiv.org/abs/2403.14353](https://arxiv.org/abs/2403.14353)

    该论文提出了一种在自主系统中加速视频分析的持续学习方法，通过利用轻量级“学生”模型进行部署推理，利用更大的“教师”模型进行数据标记，实现对不断变化场景的持续自适应。

    

    深度神经网络（DNN）视频分析对于自动驾驶车辆、无人机（UAV）和安防机器人等自主系统至关重要。然而，由于其有限的计算资源和电池功率，实际部署面临挑战。为了解决这些挑战，持续学习利用在部署（推理）中的轻量级“学生”模型，利用更大的“教师”模型对采样数据进行标记（标记），并不断重新训练学生模型以适应不断变化的场景。

    arXiv:2403.14353v1 Announce Type: cross  Abstract: Deep neural network (DNN) video analytics is crucial for autonomous systems such as self-driving vehicles, unmanned aerial vehicles (UAVs), and security robots. However, real-world deployment faces challenges due to their limited computational resources and battery power. To tackle these challenges, continuous learning exploits a lightweight "student" model at deployment (inference), leverages a larger "teacher" model for labeling sampled data (labeling), and continuously retrains the student model to adapt to changing scenarios (retraining). This paper highlights the limitations in state-of-the-art continuous learning systems: (1) they focus on computations for retraining, while overlooking the compute needs for inference and labeling, (2) they rely on power-hungry GPUs, unsuitable for battery-operated autonomous systems, and (3) they are located on a remote centralized server, intended for multi-tenant scenarios, again unsuitable for
    

