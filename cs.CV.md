# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey](https://arxiv.org/abs/2402.02242) | 本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。 |
| [^2] | [Benchmarking Spiking Neural Network Learning Methods with Varying Locality](https://arxiv.org/abs/2402.01782) | 本研究使用不同局部性对脉冲神经网络学习方法进行基准测试，并发现这些方法在性能和生物学合理性之间存在权衡。此外，研究还探讨了SNN的隐式循环特性。 |
| [^3] | [Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles.](http://arxiv.org/abs/2310.15952) | 本文引入了一种新颖的三阶段方法，通过变换器和条件扩散模型来改善医学图像分类模型对实际应用中常见成像变异性的鲁棒性。 |
| [^4] | [CBAGAN-RRT: Convolutional Block Attention Generative Adversarial Network for Sampling-Based Path Planning.](http://arxiv.org/abs/2305.10442) | 本文介绍了一种基于图像处理学习算法（CBAGAN-RRT）的路径规划方法，使用卷积块注意力生成对抗网络和一种新的损失函数，找到更优的最佳路径并提高算法的收敛速度，与先前的最先进算法相比，在图像质量生成指标和路径规划指标方面都表现更优。 |

# 详细

[^1]: 面向预训练视觉模型的参数高效微调：一项综述

    Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey

    [https://arxiv.org/abs/2402.02242](https://arxiv.org/abs/2402.02242)

    本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。

    

    大规模预训练的视觉模型（PVMs）展示了在各种下游视觉任务中的适应能力潜力。然而，随着最先进的PVMs达到数十亿甚至数万亿个参数，标准的全面微调范式由于高计算和存储需求变得不可持续。作为响应，研究人员正在探索参数高效微调（PEFT），旨在以最小参数修改超越全面微调的性能。本综述提供了视觉PEFT的全面概述和未来方向，对最新进展进行了系统审查。首先，我们提供了PEFT的正式定义，并讨论了模型预训练方法。然后，我们将现有方法分为三类：基于添加的、基于部分的和基于统一的。最后，我们介绍了常用的数据集和应用，并提出了潜在的未来研究挑战。该综述还提供了丰富的资源收藏。

    Large-scale pre-trained vision models (PVMs) have shown great potential for adaptability across various downstream vision tasks. However, with state-of-the-art PVMs growing to billions or even trillions of parameters, the standard full fine-tuning paradigm is becoming unsustainable due to high computational and storage demands. In response, researchers are exploring parameter-efficient fine-tuning (PEFT), which seeks to exceed the performance of full fine-tuning with minimal parameter modifications. This survey provides a comprehensive overview and future directions for visual PEFT, offering a systematic review of the latest advancements. First, we provide a formal definition of PEFT and discuss model pre-training methods. We then categorize existing methods into three categories: addition-based, partial-based, and unified-based. Finally, we introduce the commonly used datasets and applications and suggest potential future research challenges. A comprehensive collection of resources is
    
[^2]: 使用不同局部性对脉冲神经网络学习方法进行基准测试

    Benchmarking Spiking Neural Network Learning Methods with Varying Locality

    [https://arxiv.org/abs/2402.01782](https://arxiv.org/abs/2402.01782)

    本研究使用不同局部性对脉冲神经网络学习方法进行基准测试，并发现这些方法在性能和生物学合理性之间存在权衡。此外，研究还探讨了SNN的隐式循环特性。

    

    脉冲神经网络（SNN）提供更真实的神经动力学，在多个机器学习任务中已经显示出与人工神经网络（ANN）相当的性能。信息在SNN中以脉冲形式进行处理，采用事件驱动机制，显著降低了能源消耗。然而，由于脉冲机制的非可微性，训练SNN具有挑战性。传统方法如时间反向传播（BPTT）已经显示出一定的效果，但在计算和存储成本方面存在问题，并且在生物学上不可行。相反，最近的研究提出了具有不同局部性的替代学习方法，在分类任务中取得了成功。本文表明，这些方法在训练过程中有相似之处，同时在生物学合理性和性能之间存在权衡。此外，本研究还探讨了SNN的隐式循环特性，并进行了调查。

    Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but comes with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, this research examines the implicitly recurrent nature of SNNs and investigat
    
[^3]: 通过潜在引导扩散和嵌套集成改进医学图像分类的鲁棒性和可靠性

    Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles. (arXiv:2310.15952v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.15952](http://arxiv.org/abs/2310.15952)

    本文引入了一种新颖的三阶段方法，通过变换器和条件扩散模型来改善医学图像分类模型对实际应用中常见成像变异性的鲁棒性。

    

    尽管深度学习模型在各种医学图像分析任务中取得了显著的成功，但在真实临床环境中部署这些模型需要它们对所获取的图像的变异性具有鲁棒性。许多方法会对训练数据应用预定义的转换，以增强测试时的鲁棒性，但这些转换可能无法确保模型对患者图像中的多样性变异性具有鲁棒性。在本文中，我们提出了一种基于变换器和条件扩散模型的新型三阶段方法，旨在提高模型对实践中常见的成像变异性的鲁棒性，而无需预先确定的数据增强策略。为了实现这一目标，多个图像编码器首先学习分层特征表示来构建辨别潜在空间。接下来，一个由潜在代码引导的逆扩散过程作用于有信息先验，并提出预测候选。

    While deep learning models have achieved remarkable success across a range of medical image analysis tasks, deployment of these models in real clinical contexts requires that they be robust to variability in the acquired images. While many methods apply predefined transformations to augment the training data to enhance test-time robustness, these transformations may not ensure the model's robustness to the diverse variability seen in patient images. In this paper, we introduce a novel three-stage approach based on transformers coupled with conditional diffusion models, with the goal of improving model robustness to the kinds of imaging variability commonly encountered in practice without the need for pre-determined data augmentation strategies. To this end, multiple image encoders first learn hierarchical feature representations to build discriminative latent spaces. Next, a reverse diffusion process, guided by the latent code, acts on an informative prior and proposes prediction candi
    
[^4]: CBAGAN-RRT: 卷积块注意力生成对抗网络用于基于采样的路径规划

    CBAGAN-RRT: Convolutional Block Attention Generative Adversarial Network for Sampling-Based Path Planning. (arXiv:2305.10442v1 [cs.RO])

    [http://arxiv.org/abs/2305.10442](http://arxiv.org/abs/2305.10442)

    本文介绍了一种基于图像处理学习算法（CBAGAN-RRT）的路径规划方法，使用卷积块注意力生成对抗网络和一种新的损失函数，找到更优的最佳路径并提高算法的收敛速度，与先前的最先进算法相比，在图像质量生成指标和路径规划指标方面都表现更优。

    

    基于采样的路径规划算法在自主机器人中发挥着重要作用。但是，基于RRT算法的一个常见问题是生成的初始路径不是最优的，而且收敛速度过慢，无法应用于实际场景。本文提出了一种使用卷积块注意力生成对抗网络和一种新的损失函数的图像处理学习算法（CBAGAN-RRT），以设计启发式算法，找到更优的最佳路径，并提高算法的收敛速度。我们的GAN模型生成的路径概率分布用于引导RRT算法的采样过程。我们在由 \cite {zhang2021generative} 生成的数据集上进行了网络的训练和测试，并证明了我们的算法在图像质量生成指标（如IOU分数，Dice分数）和路径规划指标（如路径长度和成功率）方面均优于先前的最先进算法。

    Sampling-based path planning algorithms play an important role in autonomous robotics. However, a common problem among the RRT-based algorithms is that the initial path generated is not optimal and the convergence is too slow to be used in real-world applications. In this paper, we propose a novel image-based learning algorithm (CBAGAN-RRT) using a Convolutional Block Attention Generative Adversarial Network with a combination of spatial and channel attention and a novel loss function to design the heuristics, find a better optimal path, and improve the convergence of the algorithm both concerning time and speed. The probability distribution of the paths generated from our GAN model is used to guide the sampling process for the RRT algorithm. We train and test our network on the dataset generated by \cite{zhang2021generative} and demonstrate that our algorithm outperforms the previous state-of-the-art algorithms using both the image quality generation metrics like IOU Score, Dice Score
    

