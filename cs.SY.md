# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A-KIT: Adaptive Kalman-Informed Transformer.](http://arxiv.org/abs/2401.09987) | 这项研究提出了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习传感器融合中变化的过程噪声协方差。它通过适应实际情况中的过程噪声变化，改进了估计状态的准确性，避免了滤波器发散的问题。 |
| [^2] | [An Efficient Intelligent Semi-Automated Warehouse Inventory Stocktaking System.](http://arxiv.org/abs/2309.12365) | 本研究提出了一个智能库存管理系统，通过结合条码和分布式flutter应用技术，以及大数据分析实现数据驱动的决策，解决了库存管理中的准确性、监测延迟和过度依赖主观经验的挑战。 |

# 详细

[^1]: A-KIT:自适应Kalman-Informed Transformer

    A-KIT: Adaptive Kalman-Informed Transformer. (arXiv:2401.09987v1 [cs.RO])

    [http://arxiv.org/abs/2401.09987](http://arxiv.org/abs/2401.09987)

    这项研究提出了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习传感器融合中变化的过程噪声协方差。它通过适应实际情况中的过程噪声变化，改进了估计状态的准确性，避免了滤波器发散的问题。

    

    扩展卡尔曼滤波器(EKF)是导航应用中广泛采用的传感器融合方法。EKF的一个关键方面是在线确定反映模型不确定性的过程噪声协方差矩阵。尽管常见的EKF实现假设过程噪声是恒定的，但在实际情况中，过程噪声是变化的，导致估计状态的不准确，并可能导致滤波器发散。为了应对这种情况，提出了基于模型的自适应EKF方法，并展示了性能改进，凸显了对稳健自适应方法的需求。在本文中，我们推导并引入了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习变化的过程噪声协方差。A-KIT框架适用于任何类型的传感器融合。我们在这里介绍了基于惯性导航系统和多普勒速度日志的非线性传感器融合方法。通过使用来自自主无人潜水器的真实记录数据，我们验证了A-KIT的有效性。

    The extended Kalman filter (EKF) is a widely adopted method for sensor fusion in navigation applications. A crucial aspect of the EKF is the online determination of the process noise covariance matrix reflecting the model uncertainty. While common EKF implementation assumes a constant process noise, in real-world scenarios, the process noise varies, leading to inaccuracies in the estimated state and potentially causing the filter to diverge. To cope with such situations, model-based adaptive EKF methods were proposed and demonstrated performance improvements, highlighting the need for a robust adaptive approach. In this paper, we derive and introduce A-KIT, an adaptive Kalman-informed transformer to learn the varying process noise covariance online. The A-KIT framework is applicable to any type of sensor fusion. Here, we present our approach to nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log. By employing real recorded data from an autonomous und
    
[^2]: 一个高效的智能半自动仓库库存盘点系统

    An Efficient Intelligent Semi-Automated Warehouse Inventory Stocktaking System. (arXiv:2309.12365v1 [cs.HC])

    [http://arxiv.org/abs/2309.12365](http://arxiv.org/abs/2309.12365)

    本研究提出了一个智能库存管理系统，通过结合条码和分布式flutter应用技术，以及大数据分析实现数据驱动的决策，解决了库存管理中的准确性、监测延迟和过度依赖主观经验的挑战。

    

    在不断发展的供应链管理背景下，高效的库存管理对于企业变得越来越重要。然而，传统的手工和经验驱动的方法往往难以满足现代市场需求的复杂性。本研究引入了一种智能库存管理系统，以解决与数据不准确、监测延迟和过度依赖主观经验的预测相关的挑战。该系统结合了条码和分布式 flutter 应用技术，用于智能感知，并通过全面的大数据分析实现数据驱动的决策。通过仔细的分析、系统设计、关键技术探索和模拟验证，成功展示了所提出系统的有效性。该智能系统实现了二级监测、高频检查和人工智能驱动的预测，从而提高了自动化程度。

    In the context of evolving supply chain management, the significance of efficient inventory management has grown substantially for businesses. However, conventional manual and experience-based approaches often struggle to meet the complexities of modern market demands. This research introduces an intelligent inventory management system to address challenges related to inaccurate data, delayed monitoring, and overreliance on subjective experience in forecasting. The proposed system integrates bar code and distributed flutter application technologies for intelligent perception, alongside comprehensive big data analytics to enable data-driven decision-making. Through meticulous analysis, system design, critical technology exploration, and simulation validation, the effectiveness of the proposed system is successfully demonstrated. The intelligent system facilitates second-level monitoring, high-frequency checks, and artificial intelligence-driven forecasting, consequently enhancing the au
    

