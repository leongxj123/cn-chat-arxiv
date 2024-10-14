# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [D'OH: Decoder-Only random Hypernetworks for Implicit Neural Representations](https://arxiv.org/abs/2403.19163) | 本文提出使用仅运行时解码器的超网络，不依赖离线数据训练，以更好地模拟跨层参数冗余。 |
| [^2] | [Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It](https://arxiv.org/abs/2403.14715) | LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。 |
| [^3] | [Customizing Segmentation Foundation Model via Prompt Learning for Instance Segmentation](https://arxiv.org/abs/2403.09199) | 提出了一种针对Segment Anything Model (SAM)的新颖方法，通过提示学习定制化实例分割，解决了在应用于定制化实例分割时面临的输入提示模糊性和额外训练需求的挑战 |
| [^4] | [Deep Neural Decision Forest: A Novel Approach for Predicting Recovery or Decease of COVID-19 Patients with Clinical and RT-PCR.](http://arxiv.org/abs/2311.13925) | 该研究介绍了一种利用临床和RT-PCR数据结合深度学习算法来预测COVID-19患者康复或死亡风险的新方法。 |
| [^5] | [PostRainBench: A comprehensive benchmark and a new model for precipitation forecasting.](http://arxiv.org/abs/2310.02676) | PostRainBench是一个全面的降水预测基准，结合AI后处理技术和传统的数值天气预报方法，能够增强准确性并解决复杂的降水预测挑战。 |
| [^6] | [CDAN: Convolutional Dense Attention-guided Network for Low-light Image Enhancement.](http://arxiv.org/abs/2308.12902) | 本研究提出了一种名为CDAN的卷积稠密注意力引导网络，用于低光图像增强。该网络结合了自编码器架构、卷积和稠密块、注意力机制和跳跃连接，通过专门的后处理阶段进一步改善色彩平衡和对比度。与现有方法相比，在低光图像增强方面取得了显著的进展，展示了在各种具有挑战性的场景中的稳健性。 |
| [^7] | [In Defense of Pure 16-bit Floating-Point Neural Networks.](http://arxiv.org/abs/2305.10947) | 本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。 |

# 详细

[^1]: D'OH: 仅解码器随机超网络用于隐式神经表示

    D'OH: Decoder-Only random Hypernetworks for Implicit Neural Representations

    [https://arxiv.org/abs/2403.19163](https://arxiv.org/abs/2403.19163)

    本文提出使用仅运行时解码器的超网络，不依赖离线数据训练，以更好地模拟跨层参数冗余。

    

    深度隐式函数被发现是一种有效的工具，可以高效地编码各种自然信号。它们的吸引力在于能够紧凑地表示信号，几乎不需要离线训练数据。相反，它们利用深度网络的隐式偏差来解耦信号中的隐藏冗余。在本文中，我们探讨了这样一个假设：通过利用层之间存在的冗余可以实现更好的压缩。我们提出使用一种新颖的仅运行时解码器的超网络 - 它不使用离线训练数据 - 来更好地建模跨层参数冗余。先前在深度隐式函数中应用超网络的应用都采用了依赖大量离线数据集的前馈编码器/解码器框架，这些数据集无法泛化到训练信号之外。相反，我们提出一种用于初始化运行时深度隐式函数的策略

    arXiv:2403.19163v1 Announce Type: new  Abstract: Deep implicit functions have been found to be an effective tool for efficiently encoding all manner of natural signals. Their attractiveness stems from their ability to compactly represent signals with little to no off-line training data. Instead, they leverage the implicit bias of deep networks to decouple hidden redundancies within the signal. In this paper, we explore the hypothesis that additional compression can be achieved by leveraging the redundancies that exist between layers. We propose to use a novel run-time decoder-only hypernetwork - that uses no offline training data - to better model this cross-layer parameter redundancy. Previous applications of hyper-networks with deep implicit functions have applied feed-forward encoder/decoder frameworks that rely on large offline datasets that do not generalize beyond the signals they were trained on. We instead present a strategy for the initialization of run-time deep implicit func
    
[^2]: 理解为何标签平滑会降低选择性分类的效果以及如何解决这个问题

    Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It

    [https://arxiv.org/abs/2403.14715](https://arxiv.org/abs/2403.14715)

    LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。

    

    标签平滑（LS）是一种流行的深度神经网络分类器训练的正则化方法，因为它在提高测试准确性方面效果显著，并且实现简单。"硬"的one-hot标签通过将概率质量均匀分配给其他类别来进行"平滑化"，从而减少过度拟合。在这项工作中，我们揭示了LS如何负面影响选择性分类（SC）- 其目标是利用模型的预测不确定性来拒绝错误分类。我们首先在一系列任务和架构中从经验上证明LS会导致SC的一致性降级。然后，我们通过分析logit级别的梯度来解释这一点，表明LS通过在错误概率低时更加正则化最大logit，而在错误概率高时更少正则化，加剧了过度自信和低自信。这阐明了以前报道的强分类器在SC中性能不佳的实验结果。

    arXiv:2403.14715v1 Announce Type: cross  Abstract: Label smoothing (LS) is a popular regularisation method for training deep neural network classifiers due to its effectiveness in improving test accuracy and its simplicity in implementation. "Hard" one-hot labels are "smoothed" by uniformly distributing probability mass to other classes, reducing overfitting. In this work, we reveal that LS negatively affects selective classification (SC) - where the aim is to reject misclassifications using a model's predictive uncertainty. We first demonstrate empirically across a range of tasks and architectures that LS leads to a consistent degradation in SC. We then explain this by analysing logit-level gradients, showing that LS exacerbates overconfidence and underconfidence by regularising the max logit more when the probability of error is low, and less when the probability of error is high. This elucidates previously reported experimental results where strong classifiers underperform in SC. We
    
[^3]: 通过提示学习定制化分割基础模型进行实例分割

    Customizing Segmentation Foundation Model via Prompt Learning for Instance Segmentation

    [https://arxiv.org/abs/2403.09199](https://arxiv.org/abs/2403.09199)

    提出了一种针对Segment Anything Model (SAM)的新颖方法，通过提示学习定制化实例分割，解决了在应用于定制化实例分割时面临的输入提示模糊性和额外训练需求的挑战

    

    最近，通过大规模数据集训练的基础模型吸引了相当多的关注，并在计算机视觉领域得到积极探讨。在这些模型中，Segment Anything Model (SAM)因其在图像分割任务中的泛化能力和灵活性而脱颖而出，通过基于提示的对象掩模生成取得了显著进展。然而，尽管SAM具有这些优势，在应用于定制化实例分割时（对特定对象或在训练数据中通常不存在的独特环境中进行分割），SAM面临两个关键限制：1）输入提示中固有的模糊性，2）为实现最佳分割需要大量额外训练。为解决这些挑战，我们提出了一种新颖的方法，即通过提示学习定制化实例分割，针对SAM进行了定制。我们的方法包含一个提示学习模块（PLM），可以调整输入。

    arXiv:2403.09199v1 Announce Type: cross  Abstract: Recently, foundation models trained on massive datasets to adapt to a wide range of domains have attracted considerable attention and are actively being explored within the computer vision community. Among these, the Segment Anything Model (SAM) stands out for its remarkable progress in generalizability and flexibility for image segmentation tasks, achieved through prompt-based object mask generation. However, despite its strength, SAM faces two key limitations when applied to customized instance segmentation that segments specific objects or those in unique environments not typically present in the training data: 1) the ambiguity inherent in input prompts and 2) the necessity for extensive additional training to achieve optimal segmentation. To address these challenges, we propose a novel method, customized instance segmentation via prompt learning tailored to SAM. Our method involves a prompt learning module (PLM), which adjusts inpu
    
[^4]: 深度神经决策森林：一种用于预测COVID-19患者康复或死亡的新方法，结合临床和RT-PCR数据

    Deep Neural Decision Forest: A Novel Approach for Predicting Recovery or Decease of COVID-19 Patients with Clinical and RT-PCR. (arXiv:2311.13925v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2311.13925](http://arxiv.org/abs/2311.13925)

    该研究介绍了一种利用临床和RT-PCR数据结合深度学习算法来预测COVID-19患者康复或死亡风险的新方法。

    

    尽管世界卫生组织宣布大流行已经结束，但COVID-19仍然被视为一种地方性疾病。这次大流行以前所未有的方式打乱了人们的生活并导致广泛的发病率和死亡率。因此，紧急医生有必要确定高风险死亡患者，以便优先考虑医院设备的分配，尤其是在医疗资源有限的地区。尽管存在哪种数据最准确的预测的问题，但患者收集到的数据对于预测COVID-19病例的结果是有益的。因此，本研究旨在实现两个主要目标。首先，我们想要检查深度学习算法是否能够预测患者的死亡率。其次，我们研究了临床和RT-PCR对预测的影响，以确定哪个更可靠。我们定义了四个不同特征集的阶段，并使用可解释的深度学习方法构建了相应的模型。

    COVID-19 continues to be considered an endemic disease in spite of the World Health Organization's declaration that the pandemic is over. This pandemic has disrupted people's lives in unprecedented ways and caused widespread morbidity and mortality. As a result, it is important for emergency physicians to identify patients with a higher mortality risk in order to prioritize hospital equipment, especially in areas with limited medical services. The collected data from patients is beneficial to predict the outcome of COVID-19 cases, although there is a question about which data makes the most accurate predictions. Therefore, this study aims to accomplish two main objectives. First, we want to examine whether deep learning algorithms can predict a patient's morality. Second, we investigated the impact of Clinical and RT-PCR on prediction to determine which one is more reliable. We defined four stages with different feature sets and used interpretable deep learning methods to build appropr
    
[^5]: PostRainBench: 一种全面的降水预测基准和新模型

    PostRainBench: A comprehensive benchmark and a new model for precipitation forecasting. (arXiv:2310.02676v1 [cs.LG])

    [http://arxiv.org/abs/2310.02676](http://arxiv.org/abs/2310.02676)

    PostRainBench是一个全面的降水预测基准，结合AI后处理技术和传统的数值天气预报方法，能够增强准确性并解决复杂的降水预测挑战。

    

    准确的降水预测是一项具有科学和社会重要性的重大挑战。数据驱动方法已经成为解决这个挑战的广泛采用的解决方案。然而，仅依赖数据驱动方法在模拟基础物理过程方面有限，使得准确预测困难。将基于人工智能的后处理技术与传统的数值天气预报（NWP）方法相结合，为提高预测准确性提供了更有效的解决方案。尽管之前进行过后处理的尝试，但由于不同位置的降水数据失衡和多个气象变量之间的复杂关系，准确预测大雨仍然具有挑战性。为了解决这些限制，我们提出了PostRainBench，这是一个全面的多变量NWP后处理基准，包括三个用于NWP后处理降水预测的数据集。我们提出了一种简单而有效的渠道注意力模型CAMT。

    Accurate precipitation forecasting is a vital challenge of both scientific and societal importance. Data-driven approaches have emerged as a widely used solution for addressing this challenge. However, solely relying on data-driven approaches has limitations in modeling the underlying physics, making accurate predictions difficult. Coupling AI-based post-processing techniques with traditional Numerical Weather Prediction (NWP) methods offers a more effective solution for improving forecasting accuracy. Despite previous post-processing efforts, accurately predicting heavy rainfall remains challenging due to the imbalanced precipitation data across locations and complex relationships between multiple meteorological variables. To address these limitations, we introduce the PostRainBench, a comprehensive multi-variable NWP post-processing benchmark consisting of three datasets for NWP post-processing-based precipitation forecasting. We propose CAMT, a simple yet effective Channel Attention
    
[^6]: CDAN: 用于低光图像增强的卷积稠密注意力引导网络

    CDAN: Convolutional Dense Attention-guided Network for Low-light Image Enhancement. (arXiv:2308.12902v1 [cs.CV])

    [http://arxiv.org/abs/2308.12902](http://arxiv.org/abs/2308.12902)

    本研究提出了一种名为CDAN的卷积稠密注意力引导网络，用于低光图像增强。该网络结合了自编码器架构、卷积和稠密块、注意力机制和跳跃连接，通过专门的后处理阶段进一步改善色彩平衡和对比度。与现有方法相比，在低光图像增强方面取得了显著的进展，展示了在各种具有挑战性的场景中的稳健性。

    

    低光图像以不足的照明为特征，面临清晰度减弱、颜色暗淡和细节减少的挑战。低光图像增强是计算机视觉中的一个重要任务，旨在通过改善亮度、对比度和整体感知质量来纠正这些问题，从而促进准确的分析和解释。本文介绍了一种新颖的解决方案：卷积稠密注意力引导网络（CDAN），用于增强低光图像。CDAN将自编码器架构与卷积和稠密块相结合，配合注意力机制和跳跃连接。该架构确保了有效的信息传递和特征学习。此外，专门的后处理阶段可以进一步改善色彩平衡和对比度。与低光图像增强领域的最新成果相比，我们的方法取得了显著的进展，并展示了在各种具有挑战性的场景中的稳健性。

    Low-light images, characterized by inadequate illumination, pose challenges of diminished clarity, muted colors, and reduced details. Low-light image enhancement, an essential task in computer vision, aims to rectify these issues by improving brightness, contrast, and overall perceptual quality, thereby facilitating accurate analysis and interpretation. This paper introduces the Convolutional Dense Attention-guided Network (CDAN), a novel solution for enhancing low-light images. CDAN integrates an autoencoder-based architecture with convolutional and dense blocks, complemented by an attention mechanism and skip connections. This architecture ensures efficient information propagation and feature learning. Furthermore, a dedicated post-processing phase refines color balance and contrast. Our approach demonstrates notable progress compared to state-of-the-art results in low-light image enhancement, showcasing its robustness across a wide range of challenging scenarios. Our model performs 
    
[^7]: 关于纯16位浮点神经网络的辩护

    In Defense of Pure 16-bit Floating-Point Neural Networks. (arXiv:2305.10947v1 [cs.LG])

    [http://arxiv.org/abs/2305.10947](http://arxiv.org/abs/2305.10947)

    本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。

    

    减少编码神经网络权重和激活所需的位数是非常可取的，因为它可以加快神经网络的训练和推理时间，同时减少内存消耗。因此，这一领域的研究引起了广泛关注，以开发利用更低精度计算的神经网络，比如混合精度训练。有趣的是，目前不存在纯16位浮点设置的方法。本文揭示了纯16位浮点神经网络被忽视的效率。我们通过提供全面的理论分析来探讨造成16位和32位模型的差异的因素。我们规范化了浮点误差和容忍度的概念，从而可以定量解释16位模型与其32位对应物之间密切逼近结果的条件。这种理论探索提供了新的视角。

    Reducing the number of bits needed to encode the weights and activations of neural networks is highly desirable as it speeds up their training and inference time while reducing memory consumption. For these reasons, research in this area has attracted significant attention toward developing neural networks that leverage lower-precision computing, such as mixed-precision training. Interestingly, none of the existing approaches has investigated pure 16-bit floating-point settings. In this paper, we shed light on the overlooked efficiency of pure 16-bit floating-point neural networks. As such, we provide a comprehensive theoretical analysis to investigate the factors contributing to the differences observed between 16-bit and 32-bit models. We formalize the concepts of floating-point error and tolerance, enabling us to quantitatively explain the conditions under which a 16-bit model can closely approximate the results of its 32-bit counterpart. This theoretical exploration offers perspect
    

