# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Explainable Machine Learning-Based Security and Privacy Protection Framework for Internet of Medical Things Systems](https://arxiv.org/abs/2403.09752) | 该论文提出了面向互联网医疗物联网系统的可解释机器学习安全与隐私保护框架，旨在解决IoMT系统面临的安全挑战，包括数据敏感性、恶意攻击和异常检测。 |
| [^2] | [Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization.](http://arxiv.org/abs/2307.10053) | 本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。 |

# 详细

[^1]: 面向IoMT系统的可解释机器学习安全与隐私保护框架

    Explainable Machine Learning-Based Security and Privacy Protection Framework for Internet of Medical Things Systems

    [https://arxiv.org/abs/2403.09752](https://arxiv.org/abs/2403.09752)

    该论文提出了面向互联网医疗物联网系统的可解释机器学习安全与隐私保护框架，旨在解决IoMT系统面临的安全挑战，包括数据敏感性、恶意攻击和异常检测。

    

    互联网医疗物联网（IoMT）跨越了传统医疗边界，实现了从被动治疗向主动预防的过渡。这种创新方法通过实时健康数据收集实现早期疾病检测和个性化护理，特别在慢性病管理方面，IoMT可以自动化治疗。然而，由于处理数据的敏感性和价值，IoMT面临着严重的安全挑战，这会威胁到其用户的生命，因此吸引了恶意利益。此外，利用无线通信进行数据传输会使医疗数据暴露于被网络犯罪分子截获和篡改的风险之下。此外，由于人为错误、网络干扰或硬件故障，可能会出现异常。在这种背景下，基于机器学习（ML）的异常检测是一个有趣的解决方案，但它再次出现。

    arXiv:2403.09752v1 Announce Type: cross  Abstract: The Internet of Medical Things (IoMT) transcends traditional medical boundaries, enabling a transition from reactive treatment to proactive prevention. This innovative method revolutionizes healthcare by facilitating early disease detection and tailored care, particularly in chronic disease management, where IoMT automates treatments based on real-time health data collection. Nonetheless, its benefits are countered by significant security challenges that endanger the lives of its users due to the sensitivity and value of the processed data, thereby attracting malicious interests. Moreover, the utilization of wireless communication for data transmission exposes medical data to interception and tampering by cybercriminals. Additionally, anomalies may arise due to human errors, network interference, or hardware malfunctions. In this context, anomaly detection based on Machine Learning (ML) is an interesting solution, but it comes up again
    
[^2]: 非平滑非凸优化中随机次梯度方法的收敛性保证

    Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization. (arXiv:2307.10053v1 [math.OC])

    [http://arxiv.org/abs/2307.10053](http://arxiv.org/abs/2307.10053)

    本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。

    

    本文研究了随机梯度下降（SGD）方法及其变种在训练由非平滑激活函数构建的神经网络中的收敛性质。我们提出了一种新颖的框架，为更新动量项和变量的步长分配了不同的时间尺度。在一些温和的条件下，我们证明了我们提出的框架在单时间尺度和双时间尺度情况下的全局收敛性。我们还证明了我们提出的框架包含了很多已知的SGD类型方法，包括heavy-ball SGD、SignSGD、Lion、normalized SGD和clipped SGD。此外，当目标函数采用有限和形式时，我们基于我们提出的框架证明了这些SGD类型方法的收敛性质。特别地，在温和的假设下，我们证明了这些SGD类型方法在随机选择的步长和初始点上能够找到目标函数的Clarke稳定点。

    In this paper, we investigate the convergence properties of the stochastic gradient descent (SGD) method and its variants, especially in training neural networks built from nonsmooth activation functions. We develop a novel framework that assigns different timescales to stepsizes for updating the momentum terms and variables, respectively. Under mild conditions, we prove the global convergence of our proposed framework in both single-timescale and two-timescale cases. We show that our proposed framework encompasses a wide range of well-known SGD-type methods, including heavy-ball SGD, SignSGD, Lion, normalized SGD and clipped SGD. Furthermore, when the objective function adopts a finite-sum formulation, we prove the convergence properties for these SGD-type methods based on our proposed framework. In particular, we prove that these SGD-type methods find the Clarke stationary points of the objective function with randomly chosen stepsizes and initial points under mild assumptions. Preli
    

