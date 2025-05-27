# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^2] | [Preconditioners for the Stochastic Training of Implicit Neural Representations](https://arxiv.org/abs/2402.08784) | 本论文提出了一种新的随机训练方法，通过使用曲率感知对角预处理器，在不损失准确性的情况下加速了隐式神经表示的训练过程，适用于多个信号模态。 |
| [^3] | [Comprehensive Assessment of the Performance of Deep Learning Classifiers Reveals a Surprising Lack of Robustness.](http://arxiv.org/abs/2308.04137) | 通过综合评估深度学习分类器的性能，发现它们缺乏稳定性和可靠性，并建议采用广泛的数据类型和统一的评估指标进行性能基准测试。 |

# 详细

[^1]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^2]: 隐式神经表示的随机训练的预处理器

    Preconditioners for the Stochastic Training of Implicit Neural Representations

    [https://arxiv.org/abs/2402.08784](https://arxiv.org/abs/2402.08784)

    本论文提出了一种新的随机训练方法，通过使用曲率感知对角预处理器，在不损失准确性的情况下加速了隐式神经表示的训练过程，适用于多个信号模态。

    

    隐式神经表示已经成为一种强大的技术，用于将复杂连续多维信号编码为神经网络，从而实现计算机视觉、机器人学和几何学等广泛应用。尽管Adam由于其随机的高效性而被广泛应用于训练中，但其训练时间往往较长。为了解决这个问题，我们探索了在加速训练的同时不损失准确性的替代优化技术。传统的二阶优化器如L-BFGS在随机环境中效果不佳，因此不适用于大规模数据集。相反，我们提出了使用曲率感知对角预处理器进行随机训练，展示了它们在图像、形状重建和神经辐射场等各种信号模态中的有效性。

    arXiv:2402.08784v1 Announce Type: cross Abstract: Implicit neural representations have emerged as a powerful technique for encoding complex continuous multidimensional signals as neural networks, enabling a wide range of applications in computer vision, robotics, and geometry. While Adam is commonly used for training due to its stochastic proficiency, it entails lengthy training durations. To address this, we explore alternative optimization techniques for accelerated training without sacrificing accuracy. Traditional second-order optimizers like L-BFGS are suboptimal in stochastic settings, making them unsuitable for large-scale data sets. Instead, we propose stochastic training using curvature-aware diagonal preconditioners, showcasing their effectiveness across various signal modalities such as images, shape reconstruction, and Neural Radiance Fields (NeRF).
    
[^3]: 深度学习分类器性能的综合评估揭示出惊人的缺乏稳定性

    Comprehensive Assessment of the Performance of Deep Learning Classifiers Reveals a Surprising Lack of Robustness. (arXiv:2308.04137v1 [cs.LG])

    [http://arxiv.org/abs/2308.04137](http://arxiv.org/abs/2308.04137)

    通过综合评估深度学习分类器的性能，发现它们缺乏稳定性和可靠性，并建议采用广泛的数据类型和统一的评估指标进行性能基准测试。

    

    可靠而稳健的评估方法是开发本身稳健可靠的机器学习模型的必要第一步。然而，目前用于评估分类器的常规评估协议在综合评估性能方面存在不足，因为它们往往依赖于有限类型的测试数据，忽视其他类型的数据。例如，使用标准测试数据无法评估分类器对于未经训练的类别样本的预测。另一方面，使用包含未知类别样本的数据进行测试无法评估分类器对于已知类别标签的预测能力。本文提倡使用各种不同类型的数据进行性能基准测试，并使用一种可应用于所有这些数据类型的单一指标，以产生一致的性能评估结果。通过这样的基准测试发现，目前的深度神经网络，包括使用认为是全面的方法进行训练的网络，也存在缺乏稳定性的问题。

    Reliable and robust evaluation methods are a necessary first step towards developing machine learning models that are themselves robust and reliable. Unfortunately, current evaluation protocols typically used to assess classifiers fail to comprehensively evaluate performance as they tend to rely on limited types of test data, and ignore others. For example, using the standard test data fails to evaluate the predictions made by the classifier to samples from classes it was not trained on. On the other hand, testing with data containing samples from unknown classes fails to evaluate how well the classifier can predict the labels for known classes. This article advocates bench-marking performance using a wide range of different types of data and using a single metric that can be applied to all such data types to produce a consistent evaluation of performance. Using such a benchmark it is found that current deep neural networks, including those trained with methods that are believed to pro
    

