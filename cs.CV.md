# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sine Activated Low-Rank Matrices for Parameter Efficient Learning](https://arxiv.org/abs/2403.19243) | 整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。 |
| [^2] | [A Vision-Guided Robotic System for Grasping Harvested Tomato Trusses in Cluttered Environments.](http://arxiv.org/abs/2309.17170) | 提出了一种用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统。该系统利用基于深度学习的视觉系统来识别穗果并确定适合抓取的位置，通过在线学习来排序抓取姿势，并实现无触觉传感器或几何模型的夹持抓取。实验表明，该系统具有100%的清理率和93%的一次性成功抓取率。 |

# 详细

[^1]: 用正弦激活的低秩矩阵实现参数高效学习

    Sine Activated Low-Rank Matrices for Parameter Efficient Learning

    [https://arxiv.org/abs/2403.19243](https://arxiv.org/abs/2403.19243)

    整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。

    

    低秩分解已经成为在神经网络架构中增强参数效率的重要工具，在机器学习的各种应用中越来越受到关注。这些技术显著降低了参数数量，取得了简洁性和性能之间的平衡。然而，一个常见的挑战是在参数效率和模型准确性之间做出妥协，参数减少往往导致准确性不及完整秩对应模型。在这项工作中，我们提出了一个创新的理论框架，在低秩分解过程中整合了一个正弦函数。这种方法不仅保留了低秩方法的参数效率特性的好处，还增加了分解的秩，从而提高了模型的准确性。我们的方法被证明是现有低秩模型的一种适应性增强，正如其成功证实的那样。

    arXiv:2403.19243v1 Announce Type: new  Abstract: Low-rank decomposition has emerged as a vital tool for enhancing parameter efficiency in neural network architectures, gaining traction across diverse applications in machine learning. These techniques significantly lower the number of parameters, striking a balance between compactness and performance. However, a common challenge has been the compromise between parameter efficiency and the accuracy of the model, where reduced parameters often lead to diminished accuracy compared to their full-rank counterparts. In this work, we propose a novel theoretical framework that integrates a sinusoidal function within the low-rank decomposition process. This approach not only preserves the benefits of the parameter efficiency characteristic of low-rank methods but also increases the decomposition's rank, thereby enhancing model accuracy. Our method proves to be an adaptable enhancement for existing low-rank models, as evidenced by its successful 
    
[^2]: 用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统

    A Vision-Guided Robotic System for Grasping Harvested Tomato Trusses in Cluttered Environments. (arXiv:2309.17170v1 [cs.RO])

    [http://arxiv.org/abs/2309.17170](http://arxiv.org/abs/2309.17170)

    提出了一种用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统。该系统利用基于深度学习的视觉系统来识别穗果并确定适合抓取的位置，通过在线学习来排序抓取姿势，并实现无触觉传感器或几何模型的夹持抓取。实验表明，该系统具有100%的清理率和93%的一次性成功抓取率。

    

    目前，对于西红柿的称重和包装需要大量的人工操作。自动化的主要障碍在于开发一个可靠的用于已采摘的穗果的机器人抓取系统的困难。我们提出了一种方法来抓取堆放在装箱中的穗果，这是它们在采摘后常见的存储和运输方式。该方法包括一个基于深度学习的视觉系统，首先识别出装箱中的单个穗果，然后确定茎部的适合抓取的位置。为此，我们引入了一个具有在线学习能力的抓取姿势排序算法。在选择了最有前景的抓取姿势之后，机器人执行一种无需触觉传感器或几何模型的夹持抓取。实验室实验证明，配备了一个手眼一体的RGB-D相机的机器人操纵器从堆中捡起所有的穗果的清理率达到100%。93%的穗果在第一次尝试时成功抓取。

    Currently, truss tomato weighing and packaging require significant manual work. The main obstacle to automation lies in the difficulty of developing a reliable robotic grasping system for already harvested trusses. We propose a method to grasp trusses that are stacked in a crate with considerable clutter, which is how they are commonly stored and transported after harvest. The method consists of a deep learning-based vision system to first identify the individual trusses in the crate and then determine a suitable grasping location on the stem. To this end, we have introduced a grasp pose ranking algorithm with online learning capabilities. After selecting the most promising grasp pose, the robot executes a pinch grasp without needing touch sensors or geometric models. Lab experiments with a robotic manipulator equipped with an eye-in-hand RGB-D camera showed a 100% clearance rate when tasked to pick all trusses from a pile. 93% of the trusses were successfully grasped on the first try,
    

