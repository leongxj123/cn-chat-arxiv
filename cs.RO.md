# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transformer-based deep imitation learning for dual-arm robot manipulation](https://arxiv.org/abs/2108.00385) | 使用Transformer的深度模仿学习结构成功解决了双臂机器人操作任务中神经网络性能不佳的问题 |
| [^2] | [Gaze-based dual resolution deep imitation learning for high-precision dexterous robot manipulation](https://arxiv.org/abs/2102.01295) | 基于人类基于凝视的双分辨率视觉运动控制系统的启发，提出了一种利用深度模仿学习解决高精度灵巧机器人操作任务的方法 |

# 详细

[^1]: 基于Transformer的双臂机器人操作的深度模仿学习

    Transformer-based deep imitation learning for dual-arm robot manipulation

    [https://arxiv.org/abs/2108.00385](https://arxiv.org/abs/2108.00385)

    使用Transformer的深度模仿学习结构成功解决了双臂机器人操作任务中神经网络性能不佳的问题

    

    深度模仿学习对解决熟练操作任务具有潜力，因为它不需要环境模型和预编程的机器人行为。然而，将其应用于双臂操作任务仍具有挑战性。在双臂操作设置中，由于附加机器人操作器引起的状态维度增加，导致了神经网络性能不佳。我们通过使用一种自注意力机制来解决这个问题，该机制计算顺序输入中元素之间的依赖关系，并专注于重要元素。Transformer，作为自注意力架构的一种变体，被应用于深度模仿学习中，以解决真实世界中的双臂操作任务。所提出的方法已在真实机器人上的双臂操作任务上进行了测试。实验结果表明，基于Transformer的深度模仿学习架构可以进行关注

    arXiv:2108.00385v2 Announce Type: replace-cross  Abstract: Deep imitation learning is promising for solving dexterous manipulation tasks because it does not require an environment model and pre-programmed robot behavior. However, its application to dual-arm manipulation tasks remains challenging. In a dual-arm manipulation setup, the increased number of state dimensions caused by the additional robot manipulators causes distractions and results in poor performance of the neural networks. We address this issue using a self-attention mechanism that computes dependencies between elements in a sequential input and focuses on important elements. A Transformer, a variant of self-attention architecture, is applied to deep imitation learning to solve dual-arm manipulation tasks in the real world. The proposed method has been tested on dual-arm manipulation tasks using a real robot. The experimental results demonstrated that the Transformer-based deep imitation learning architecture can attend 
    
[^2]: 基于凝视的双分辨率深度模仿学习用于高精度灵巧机器人操作

    Gaze-based dual resolution deep imitation learning for high-precision dexterous robot manipulation

    [https://arxiv.org/abs/2102.01295](https://arxiv.org/abs/2102.01295)

    基于人类基于凝视的双分辨率视觉运动控制系统的启发，提出了一种利用深度模仿学习解决高精度灵巧机器人操作任务的方法

    

    一个高精度操纵任务，如穿针引线，是具有挑战性的。生理学研究提出了将低分辨率外围视觉和快速移动连接起来，将手传送到对象的附近，并使用高分辨率的凹陷视觉来实现手精确对准对象。本研究结果表明，受人类基于凝视的双分辨率视觉运动控制系统的启发，基于深度模仿学习的方法可以解决穿针引线任务。首先，我们记录了远程操作机器人的人类操作员的凝视运动。然后，在靠近目标时，我们仅使用围绕凝视点的高分辨率图像来精确控制线的位置。我们使用低分辨率的外围图像到达目标附近。本研究获得的实验结果表明，所提出的方法实现了精准的操纵

    arXiv:2102.01295v3 Announce Type: replace-cross  Abstract: A high-precision manipulation task, such as needle threading, is challenging. Physiological studies have proposed connecting low-resolution peripheral vision and fast movement to transport the hand into the vicinity of an object, and using high-resolution foveated vision to achieve the accurate homing of the hand to the object. The results of this study demonstrate that a deep imitation learning based method, inspired by the gaze-based dual resolution visuomotor control system in humans, can solve the needle threading task. First, we recorded the gaze movements of a human operator who was teleoperating a robot. Then, we used only a high-resolution image around the gaze to precisely control the thread position when it was close to the target. We used a low-resolution peripheral image to reach the vicinity of the target. The experimental results obtained in this study demonstrate that the proposed method enables precise manipulat
    

