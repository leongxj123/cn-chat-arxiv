# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DEM: A Method for Certifying Deep Neural Network Classifier Outputs in Aerospace.](http://arxiv.org/abs/2401.02283) | 这项工作提出了一种新的、以输出为中心的方法，通过统计验证技术来认证深度神经网络(DNN)分类器的输出。该方法能够标记可能不可靠的特定输入，以便后续由人工专家检查。与现有技术相比，该方法主要关注单个输出而不是整个DNN的认证。 |
| [^2] | [Quality Issues in Machine Learning Software Systems.](http://arxiv.org/abs/2306.15007) | 本文通过采访实践者和进行调查的方式，研究了机器学习软件系统中的质量问题，并确定了一个质量问题目录。 |

# 详细

[^1]: DEM: 航空航天中用于认证深度神经网络分类器输出的方法

    DEM: A Method for Certifying Deep Neural Network Classifier Outputs in Aerospace. (arXiv:2401.02283v1 [cs.SE])

    [http://arxiv.org/abs/2401.02283](http://arxiv.org/abs/2401.02283)

    这项工作提出了一种新的、以输出为中心的方法，通过统计验证技术来认证深度神经网络(DNN)分类器的输出。该方法能够标记可能不可靠的特定输入，以便后续由人工专家检查。与现有技术相比，该方法主要关注单个输出而不是整个DNN的认证。

    

    航空航天领域的软件开发要求遵循严格、高质量的标准。尽管在这个领域中存在着商用软件的监管指南（例如ARP-4754和DO-178），但这些指南并不适用于具有深度神经网络（DNN）组件的软件。因此，如何使航空航天系统受益于深度学习的革命尚不清楚。我们的研究旨在通过一种新颖的、以输出为中心的方法来解决这个挑战，用于DNN的认证。我们的方法采用统计验证技术，并具有能够标记DNN输出可能不可靠的特定输入的关键优势，以便后续由专家检查。为了实现这一点，我们的方法对DNN对其他附近输入的预测进行统计分析，以检测不一致性。这与现有技术相反，后者通常试图对整个DNN进行认证，而非单个输出。

    Software development in the aerospace domain requires adhering to strict, high-quality standards. While there exist regulatory guidelines for commercial software in this domain (e.g., ARP-4754 and DO-178), these do not apply to software with deep neural network (DNN) components. Consequently, it is unclear how to allow aerospace systems to benefit from the deep learning revolution. Our work here seeks to address this challenge with a novel, output-centric approach for DNN certification. Our method employs statistical verification techniques, and has the key advantage of being able to flag specific inputs for which the DNN's output may be unreliable - so that they may be later inspected by a human expert. To achieve this, our method conducts a statistical analysis of the DNN's predictions for other, nearby inputs, in order to detect inconsistencies. This is in contrast to existing techniques, which typically attempt to certify the entire DNN, as opposed to individual outputs. Our method
    
[^2]: 机器学习软件系统中的质量问题

    Quality Issues in Machine Learning Software Systems. (arXiv:2306.15007v1 [cs.SE])

    [http://arxiv.org/abs/2306.15007](http://arxiv.org/abs/2306.15007)

    本文通过采访实践者和进行调查的方式，研究了机器学习软件系统中的质量问题，并确定了一个质量问题目录。

    

    上下文：在各个领域中，越来越多的需求是使用机器学习（ML）来解决复杂的问题。ML模型被实现为软件组件并部署在机器学习软件系统（MLSSs）中。问题：有必要确保MLSSs的服务质量。这些系统的错误或不良决策可能导致其他系统的故障，造成巨大的财务损失，甚至对人类生命构成威胁。 MLSSs的质量保证被认为是一项具有挑战性的任务，目前是一个热门的研究课题。目标：本文旨在从实践者的角度研究MLSSs中真实质量问题的特征。这项实证研究旨在确定MLSSs中的质量问题目录。方法：我们与实践者/专家进行了一系列采访，以获取他们处理质量问题时的经验和做法。我们通过对ML从业者的调查验证了所确定的质量问题。结果

    Context: An increasing demand is observed in various domains to employ Machine Learning (ML) for solving complex problems. ML models are implemented as software components and deployed in Machine Learning Software Systems (MLSSs). Problem: There is a strong need for ensuring the serving quality of MLSSs. False or poor decisions of such systems can lead to malfunction of other systems, significant financial losses, or even threats to human life. The quality assurance of MLSSs is considered a challenging task and currently is a hot research topic. Objective: This paper aims to investigate the characteristics of real quality issues in MLSSs from the viewpoint of practitioners. This empirical study aims to identify a catalog of quality issues in MLSSs. Method: We conduct a set of interviews with practitioners/experts, to gather insights about their experience and practices when dealing with quality issues. We validate the identified quality issues via a survey with ML practitioners. Result
    

