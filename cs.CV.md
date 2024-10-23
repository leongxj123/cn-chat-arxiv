# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Auxiliary CycleGAN-guidance for Task-Aware Domain Translation from Duplex to Monoplex IHC Images](https://arxiv.org/abs/2403.07389) | 通过引入新的训练设计，从而利用辅助的免疫荧光图像域，我们提出了一种用于从双向到单向IHC图像的任务感知域翻译的方法，该方法在下游分割任务中表现出比基线方法更好的效果。 |
| [^2] | [Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning](https://arxiv.org/abs/2402.02500) | 通过广泛实验发现基于点云的方法在机器人学习中表现出更好的性能，特别是在各种预训练和泛化任务中。结果表明，点云观测模态对于复杂机器人任务是有价值的。 |
| [^3] | [StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling.](http://arxiv.org/abs/2310.17042) | StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。 |

# 详细

[^1]: 辅助CycleGAN引导下的从双向到单向IHC图像的任务感知域翻译

    Auxiliary CycleGAN-guidance for Task-Aware Domain Translation from Duplex to Monoplex IHC Images

    [https://arxiv.org/abs/2403.07389](https://arxiv.org/abs/2403.07389)

    通过引入新的训练设计，从而利用辅助的免疫荧光图像域，我们提出了一种用于从双向到单向IHC图像的任务感知域翻译的方法，该方法在下游分割任务中表现出比基线方法更好的效果。

    

    生成模型使得从一个源图像域到一个在训练中未见过的目标域的转换成为可能。虽然Cycle生成对抗网络（GANs）已经被广泛应用，但其中的循环一致性约束依赖于两个域之间存在可逆映射的情况，而在染色单向和双向免疫组化（IHC）检测的图像之间的转换不是这样的。针对从后者到前者的转换，我们提出了一种新颖的训练设计，引入了一种新的约束，利用一组免疫荧光（IF）图像作为辅助的不配对图像域。在下游分割任务上的定量和定性结果显示，相比基线方法，所提出的方法带来了显著的好处。

    arXiv:2403.07389v1 Announce Type: cross  Abstract: Generative models enable the translation from a source image domain where readily trained models are available to a target domain unseen during training. While Cycle Generative Adversarial Networks (GANs) are well established, the associated cycle consistency constrain relies on that an invertible mapping exists between the two domains. This is, however, not the case for the translation between images stained with chromogenic monoplex and duplex immunohistochemistry (IHC) assays. Focusing on the translation from the latter to the first, we propose - through the introduction of a novel training design, an alternative constrain leveraging a set of immunofluorescence (IF) images as an auxiliary unpaired image domain. Quantitative and qualitative results on a downstream segmentation task show the benefit of the proposed method in comparison to baseline approaches.
    
[^2]: 点云问题:重新思考不同观测空间对机器人学习的影响

    Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning

    [https://arxiv.org/abs/2402.02500](https://arxiv.org/abs/2402.02500)

    通过广泛实验发现基于点云的方法在机器人学习中表现出更好的性能，特别是在各种预训练和泛化任务中。结果表明，点云观测模态对于复杂机器人任务是有价值的。

    

    在这项研究中，我们探讨了不同观测空间对机器人学习的影响，重点关注了三种主要模态：RGB，RGB-D和点云。通过在超过17个不同接触丰富的操作任务上进行广泛实验，涉及两个基准和仿真器，我们观察到了一个显著的趋势：基于点云的方法，即使是最简单的设计，通常在性能上超过了其RGB和RGB-D的对应物。这在从头开始训练和利用预训练的两种情况下都是一致的。此外，我们的研究结果表明，点云观测在相机视角、照明条件、噪声水平和背景外观等各种几何和视觉线索方面，都能提高策略零样本泛化能力。研究结果表明，三维点云是复杂机器人任务中有价值的观测模态。我们将公开所有的代码和检查点，希望我们的观点能帮助解决问题。

    In this study, we explore the influence of different observation spaces on robot learning, focusing on three predominant modalities: RGB, RGB-D, and point cloud. Through extensive experimentation on over 17 varied contact-rich manipulation tasks, conducted across two benchmarks and simulators, we have observed a notable trend: point cloud-based methods, even those with the simplest designs, frequently surpass their RGB and RGB-D counterparts in performance. This remains consistent in both scenarios: training from scratch and utilizing pretraining. Furthermore, our findings indicate that point cloud observations lead to improved policy zero-shot generalization in relation to various geometry and visual clues, including camera viewpoints, lighting conditions, noise levels and background appearance. The outcomes suggest that 3D point cloud is a valuable observation modality for intricate robotic tasks. We will open-source all our codes and checkpoints, hoping that our insights can help de
    
[^3]: StochGradAdam: 利用随机梯度抽样加速神经网络训练

    StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling. (arXiv:2310.17042v1 [cs.LG])

    [http://arxiv.org/abs/2310.17042](http://arxiv.org/abs/2310.17042)

    StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。

    

    在深度学习优化领域中，本文介绍了StochGradAdam优化器，这是对广受赞誉的Adam算法的新颖改进。StochGradAdam的核心是其梯度抽样技术。该方法不仅确保稳定收敛，而且利用选择性梯度考虑的优势，通过减轻噪声或异常数据的影响和增强损失函数空间的探索，提升了鲁棒训练。在图像分类和分割任务中，StochGradAdam表现出优于传统Adam优化器的性能。通过在每次迭代中精心选择一部分梯度进行抽样，该优化器能够有效应对复杂模型的管理。本文从数学基础到偏差校正策略全面探讨了StochGradAdam的方法，展示了深度学习训练技术的可期进展。

    In the rapidly advancing domain of deep learning optimization, this paper unveils the StochGradAdam optimizer, a novel adaptation of the well-regarded Adam algorithm. Central to StochGradAdam is its gradient sampling technique. This method not only ensures stable convergence but also leverages the advantages of selective gradient consideration, fostering robust training by potentially mitigating the effects of noisy or outlier data and enhancing the exploration of the loss landscape for more dependable convergence. In both image classification and segmentation tasks, StochGradAdam has demonstrated superior performance compared to the traditional Adam optimizer. By judiciously sampling a subset of gradients at each iteration, the optimizer is optimized for managing intricate models. The paper provides a comprehensive exploration of StochGradAdam's methodology, from its mathematical foundations to bias correction strategies, heralding a promising advancement in deep learning training tec
    

