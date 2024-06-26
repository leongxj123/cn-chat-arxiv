# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DEM: A Method for Certifying Deep Neural Network Classifier Outputs in Aerospace.](http://arxiv.org/abs/2401.02283) | 这项工作提出了一种新的、以输出为中心的方法，通过统计验证技术来认证深度神经网络(DNN)分类器的输出。该方法能够标记可能不可靠的特定输入，以便后续由人工专家检查。与现有技术相比，该方法主要关注单个输出而不是整个DNN的认证。 |

# 详细

[^1]: DEM: 航空航天中用于认证深度神经网络分类器输出的方法

    DEM: A Method for Certifying Deep Neural Network Classifier Outputs in Aerospace. (arXiv:2401.02283v1 [cs.SE])

    [http://arxiv.org/abs/2401.02283](http://arxiv.org/abs/2401.02283)

    这项工作提出了一种新的、以输出为中心的方法，通过统计验证技术来认证深度神经网络(DNN)分类器的输出。该方法能够标记可能不可靠的特定输入，以便后续由人工专家检查。与现有技术相比，该方法主要关注单个输出而不是整个DNN的认证。

    

    航空航天领域的软件开发要求遵循严格、高质量的标准。尽管在这个领域中存在着商用软件的监管指南（例如ARP-4754和DO-178），但这些指南并不适用于具有深度神经网络（DNN）组件的软件。因此，如何使航空航天系统受益于深度学习的革命尚不清楚。我们的研究旨在通过一种新颖的、以输出为中心的方法来解决这个挑战，用于DNN的认证。我们的方法采用统计验证技术，并具有能够标记DNN输出可能不可靠的特定输入的关键优势，以便后续由专家检查。为了实现这一点，我们的方法对DNN对其他附近输入的预测进行统计分析，以检测不一致性。这与现有技术相反，后者通常试图对整个DNN进行认证，而非单个输出。

    Software development in the aerospace domain requires adhering to strict, high-quality standards. While there exist regulatory guidelines for commercial software in this domain (e.g., ARP-4754 and DO-178), these do not apply to software with deep neural network (DNN) components. Consequently, it is unclear how to allow aerospace systems to benefit from the deep learning revolution. Our work here seeks to address this challenge with a novel, output-centric approach for DNN certification. Our method employs statistical verification techniques, and has the key advantage of being able to flag specific inputs for which the DNN's output may be unreliable - so that they may be later inspected by a human expert. To achieve this, our method conducts a statistical analysis of the DNN's predictions for other, nearby inputs, in order to detect inconsistencies. This is in contrast to existing techniques, which typically attempt to certify the entire DNN, as opposed to individual outputs. Our method
    

