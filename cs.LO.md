# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustness Assessment of a Runway Object Classifier for Safe Aircraft Taxiing](https://arxiv.org/abs/2402.00035) | 本文介绍了对航班滑行安全的跑道物体分类器的鲁棒性评估，使用形式方法评估了该分类器对三种常见图像扰动类型的鲁棒性，并提出了一种利用单调性的方法。 |

# 详细

[^1]: 航班滑行安全的跑道物体分类器的鲁棒性评估

    Robustness Assessment of a Runway Object Classifier for Safe Aircraft Taxiing

    [https://arxiv.org/abs/2402.00035](https://arxiv.org/abs/2402.00035)

    本文介绍了对航班滑行安全的跑道物体分类器的鲁棒性评估，使用形式方法评估了该分类器对三种常见图像扰动类型的鲁棒性，并提出了一种利用单调性的方法。

    

    随着深度神经网络(DNNs)在许多计算问题上成为主要解决方案，航空业希望探索它们在减轻飞行员负担和改善运营安全方面的潜力。然而，在这类安全关键应用中使用DNNs需要进行彻底的认证过程。这一需求可以通过形式验证来解决，形式验证提供了严格的保证，例如证明某些误判的不存在。在本文中，我们使用Airbus当前正在开发的图像分类器DNN作为案例研究，旨在在飞机滑行阶段使用。我们使用形式方法来评估这个DNN对三种常见图像扰动类型的鲁棒性：噪声、亮度和对比度，以及它们的部分组合。这个过程涉及多次调用底层验证器，这可能在计算上是昂贵的；因此，我们提出了一种利用单调性的方法。

    As deep neural networks (DNNs) are becoming the prominent solution for many computational problems, the aviation industry seeks to explore their potential in alleviating pilot workload and in improving operational safety. However, the use of DNNs in this type of safety-critical applications requires a thorough certification process. This need can be addressed through formal verification, which provides rigorous assurances -- e.g.,~by proving the absence of certain mispredictions. In this case-study paper, we demonstrate this process using an image-classifier DNN currently under development at Airbus and intended for use during the aircraft taxiing phase. We use formal methods to assess this DNN's robustness to three common image perturbation types: noise, brightness and contrast, and some of their combinations. This process entails multiple invocations of the underlying verifier, which might be computationally expensive; and we therefore propose a method that leverages the monotonicity
    

