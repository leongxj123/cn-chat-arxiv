# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Double Machine Learning Approach to Combining Experimental and Observational Data.](http://arxiv.org/abs/2307.01449) | 这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。 |
| [^2] | [Spatiotemporal Besov Priors for Bayesian Inverse Problems.](http://arxiv.org/abs/2306.16378) | 本研究通过将贝索夫过程推广到时空领域，以更好地处理贝叶斯逆问题中的时空重建。通过替换随机系数，该方法能够保持边缘特征并模拟动态变化图像的时空相关性。 |

# 详细

[^1]: 将实验数据与观测数据结合的双机器学习方法

    A Double Machine Learning Approach to Combining Experimental and Observational Data. (arXiv:2307.01449v1 [stat.ME])

    [http://arxiv.org/abs/2307.01449](http://arxiv.org/abs/2307.01449)

    这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。

    

    实验和观测研究通常由于无法测试的假设而缺乏有效性。我们提出了一种双机器学习方法，将实验和观测研究结合起来，使从业人员能够测试假设违反情况并一致估计处理效应。我们的框架在较轻的假设下测试外部效度和可忽视性的违反情况。当只有一个假设被违反时，我们提供半参数高效的处理效应估计器。然而，我们的无免费午餐定理强调了准确识别违反的假设对一致的处理效应估计的必要性。我们通过三个实际案例研究展示了我们方法的适用性，并突出了其在实际环境中的相关性。

    Experimental and observational studies often lack validity due to untestable assumptions. We propose a double machine learning approach to combine experimental and observational studies, allowing practitioners to test for assumption violations and estimate treatment effects consistently. Our framework tests for violations of external validity and ignorability under milder assumptions. When only one assumption is violated, we provide semi-parametrically efficient treatment effect estimators. However, our no-free-lunch theorem highlights the necessity of accurately identifying the violated assumption for consistent treatment effect estimation. We demonstrate the applicability of our approach in three real-world case studies, highlighting its relevance for practical settings.
    
[^2]: 贝索夫先验在贝叶斯逆问题中的时空应用

    Spatiotemporal Besov Priors for Bayesian Inverse Problems. (arXiv:2306.16378v1 [stat.ME])

    [http://arxiv.org/abs/2306.16378](http://arxiv.org/abs/2306.16378)

    本研究通过将贝索夫过程推广到时空领域，以更好地处理贝叶斯逆问题中的时空重建。通过替换随机系数，该方法能够保持边缘特征并模拟动态变化图像的时空相关性。

    

    近年来，科学技术的快速发展促使对捕捉数据特征（如突变或明显对比度）的适当统计工具的需求。许多数据科学应用需要从具有不连续性或奇异性的时间相关对象序列中进行时空重建，如带有边缘的动态计算机断层影像（CT）图像。传统的基于高斯过程（GP）的方法可能无法提供令人满意的解决方案，因为它们往往提供过度平滑的先验候选。最近，通过随机系数的小波展开定义的贝索夫过程（BP）被提出作为这类贝叶斯逆问题的更合适的先验。BP在成像分析中表现出优于GP的性能，能够产生保留边缘特征的重建结果，但没有自动地纳入动态变化图像中的时间相关性。本文将BP推广到时空领域（STBP），通过在小波展开中替换随机系数，实现了时空相关性的建模。

    Fast development in science and technology has driven the need for proper statistical tools to capture special data features such as abrupt changes or sharp contrast. Many applications in the data science seek spatiotemporal reconstruction from a sequence of time-dependent objects with discontinuity or singularity, e.g. dynamic computerized tomography (CT) images with edges. Traditional methods based on Gaussian processes (GP) may not provide satisfactory solutions since they tend to offer over-smooth prior candidates. Recently, Besov process (BP) defined by wavelet expansions with random coefficients has been proposed as a more appropriate prior for this type of Bayesian inverse problems. While BP outperforms GP in imaging analysis to produce edge-preserving reconstructions, it does not automatically incorporate temporal correlation inherited in the dynamically changing images. In this paper, we generalize BP to the spatiotemporal domain (STBP) by replacing the random coefficients in 
    

