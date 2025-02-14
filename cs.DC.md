# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rendering Wireless Environments Useful for Gradient Estimators: A Zero-Order Stochastic Federated Learning Method](https://arxiv.org/abs/2401.17460) | 提出了一种新颖的零阶随机联邦学习方法，通过利用无线通信通道的特性，在学习算法中考虑了无线通道，避免了资源的浪费和分析难度。 |

# 详细

[^1]: 使无线环境对梯度估计器有用：一种零阶随机联邦学习方法

    Rendering Wireless Environments Useful for Gradient Estimators: A Zero-Order Stochastic Federated Learning Method

    [https://arxiv.org/abs/2401.17460](https://arxiv.org/abs/2401.17460)

    提出了一种新颖的零阶随机联邦学习方法，通过利用无线通信通道的特性，在学习算法中考虑了无线通道，避免了资源的浪费和分析难度。

    

    联邦学习（FL）是一种新颖的机器学习方法，允许多个边缘设备协同训练模型，而无需公开原始数据。然而，当设备和服务器通过无线信道通信时，该方法面临着通信和计算瓶颈。通过利用一个通信高效的框架，我们提出了一种新颖的零阶（ZO）方法，采用一点梯度估计器，利用无线通信通道的特性，而无需知道通道状态系数。这是第一种将无线通道包含在学习算法本身中的方法，而不是浪费资源来分析和消除其影响。这项工作的两个主要困难是，在FL中，目标函数通常不是凸的，这使得将FL扩展到ZO方法具有挑战性，以及包括影响的难度。

    Federated learning (FL) is a novel approach to machine learning that allows multiple edge devices to collaboratively train a model without disclosing their raw data. However, several challenges hinder the practical implementation of this approach, especially when devices and the server communicate over wireless channels, as it suffers from communication and computation bottlenecks in this case. By utilizing a communication-efficient framework, we propose a novel zero-order (ZO) method with a one-point gradient estimator that harnesses the nature of the wireless communication channel without requiring the knowledge of the channel state coefficient. It is the first method that includes the wireless channel in the learning algorithm itself instead of wasting resources to analyze it and remove its impact. The two main difficulties of this work are that in FL, the objective function is usually not convex, which makes the extension of FL to ZO methods challenging, and that including the impa
    

