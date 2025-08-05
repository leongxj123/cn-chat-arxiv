# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EqDrive: Efficient Equivariant Motion Forecasting with Multi-Modality for Autonomous Driving.](http://arxiv.org/abs/2310.17540) | 本研究发展了EqDrive模型，通过使用EqMotion等变粒子和人类预测模型以及多模式预测机制，在自动驾驶中实现了高效的车辆运动预测。该模型在模型容量较低、参数更少、训练时间显著缩短的情况下，取得了业界最先进的性能。 |

# 详细

[^1]: EqDrive: 自动驾驶的高效等变运动预测与多模式处理

    EqDrive: Efficient Equivariant Motion Forecasting with Multi-Modality for Autonomous Driving. (arXiv:2310.17540v1 [cs.RO])

    [http://arxiv.org/abs/2310.17540](http://arxiv.org/abs/2310.17540)

    本研究发展了EqDrive模型，通过使用EqMotion等变粒子和人类预测模型以及多模式预测机制，在自动驾驶中实现了高效的车辆运动预测。该模型在模型容量较低、参数更少、训练时间显著缩短的情况下，取得了业界最先进的性能。

    

    在自动驾驶中预测车辆运动需要对车辆间的相互作用有深入的理解，并保持在欧几里得几何变换下的运动等变性。传统模型往往缺乏处理自动驾驶车辆中复杂动力学和场景中各主体之间交互关系所需的复杂性。因此，这些模型具有较低的模型容量，导致更高的预测误差和较低的训练效率。在我们的研究中，我们使用EqMotion，一个领先的等变粒子和人类预测模型，该模型还考虑到不变的主体间相互作用，用于多代理车辆运动预测任务。此外，我们使用多模式预测机制以概率化方式考虑多个可能的未来路径。通过利用EqMotion，我们的模型在参数更少（120万）和训练时间显著缩短（少于..）的情况下实现了业界最先进的性能。

    Forecasting vehicular motions in autonomous driving requires a deep understanding of agent interactions and the preservation of motion equivariance under Euclidean geometric transformations. Traditional models often lack the sophistication needed to handle the intricate dynamics inherent to autonomous vehicles and the interaction relationships among agents in the scene. As a result, these models have a lower model capacity, which then leads to higher prediction errors and lower training efficiency. In our research, we employ EqMotion, a leading equivariant particle, and human prediction model that also accounts for invariant agent interactions, for the task of multi-agent vehicle motion forecasting. In addition, we use a multi-modal prediction mechanism to account for multiple possible future paths in a probabilistic manner. By leveraging EqMotion, our model achieves state-of-the-art (SOTA) performance with fewer parameters (1.2 million) and a significantly reduced training time (less 
    

