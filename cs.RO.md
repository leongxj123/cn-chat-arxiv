# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm](https://arxiv.org/abs/2403.05666) | 通过对ICP算法进行基于深度学习的攻击，在安全关键应用中评估其鲁棒性，重点在于找到可能的最大ICP姿势误差。 |

# 详细

[^1]: 面对最坏情况：一种基于学习的对ICP算法鲁棒性分析的对抗攻击

    Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm

    [https://arxiv.org/abs/2403.05666](https://arxiv.org/abs/2403.05666)

    通过对ICP算法进行基于深度学习的攻击，在安全关键应用中评估其鲁棒性，重点在于找到可能的最大ICP姿势误差。

    

    这篇论文提出了一种通过深度学习攻击激光雷达点云来评估迭代最近点（ICP）算法鲁棒性的新方法。对于像自主导航这样的安全关键应用，确保算法在部署前的鲁棒性至关重要。ICP算法已成为基于激光雷达的定位的标准。然而，它产生的姿势估计可能会受到测量数据的影响。数据的污染可能来自各种场景，如遮挡、恶劣天气或传感器的机械问题。不幸的是，ICP的复杂和迭代特性使得评估其对污染的鲁棒性具有挑战性。虽然已经有人努力创建具有挑战性的数据集和开发仿真来经验性地评估ICP的鲁棒性，但我们的方法侧重于通过基于扰动的对抗攻击找到最大可能的ICP姿势误差。

    arXiv:2403.05666v1 Announce Type: cross  Abstract: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial
    

