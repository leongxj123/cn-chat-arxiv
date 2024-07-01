# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FishNet: Deep Neural Networks for Low-Cost Fish Stock Estimation](https://arxiv.org/abs/2403.10916) | 提出了FishNet，一个自动化计算机视觉系统，利用低成本数码相机图像执行鱼类分类和大小估算，具有高准确性和精度。 |
| [^2] | [Logical Closed Loop: Uncovering Object Hallucinations in Large Vision-Language Models](https://arxiv.org/abs/2402.11622) | 提出了一种基于逻辑闭环的框架（LogicCheckGPT），利用大型视觉-语言模型本身来检测和减轻对象幻觉。 |
| [^3] | [Robustness Assessment of a Runway Object Classifier for Safe Aircraft Taxiing](https://arxiv.org/abs/2402.00035) | 本文介绍了对航班滑行安全的跑道物体分类器的鲁棒性评估，使用形式方法评估了该分类器对三种常见图像扰动类型的鲁棒性，并提出了一种利用单调性的方法。 |
| [^4] | [Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling.](http://arxiv.org/abs/2310.06389) | 本研究提出了乐高积木，通过集成局部特征丰富和全局内容协调，实现了高效且可自适应的迭代细化扩散建模。这些积木可以堆叠在一起，用于在测试时根据需要进行重构，从而减少采样成本并生成高分辨率图像。 |
| [^5] | [Generative Autoencoding of Dropout Patterns.](http://arxiv.org/abs/2310.01712) | 本论文提出了一种称为解读自编码器的生成模型，通过为训练数据集中的每个数据点分配独特的随机丢弃模式来进行训练，只依靠重构误差来提供更稳定的训练性能，并在CIFAR-10数据集上展示了与DCGAN相媲美的采样质量。 |

# 详细

[^1]: FishNet:用于低成本鱼类存栏估算的深度神经网络

    FishNet: Deep Neural Networks for Low-Cost Fish Stock Estimation

    [https://arxiv.org/abs/2403.10916](https://arxiv.org/abs/2403.10916)

    提出了FishNet，一个自动化计算机视觉系统，利用低成本数码相机图像执行鱼类分类和大小估算，具有高准确性和精度。

    

    鱼类库存评估通常需要由分类专家进行手工鱼类计数，这既耗时又昂贵。我们提出了一个自动化的计算机视觉系统，可以从使用低成本数码相机拍摄的图像中执行分类和鱼类大小估算。该系统首先利用Mask R-CNN执行目标检测和分割，以识别包含多条鱼的图像中的单条鱼，这些鱼可能由不同物种组成。然后，每个鱼类被分类并使用单独的机器学习模型预测长度。这些模型训练于包含50,000张手工注释图像的数据集，其中包含163种不同长度从10厘米到250厘米的鱼类。在保留的测试数据上评估，我们的系统在鱼类分割任务上达到了92%的交并比，单一鱼类分类准确率为89%，平均误差为2.3厘米。

    arXiv:2403.10916v1 Announce Type: cross  Abstract: Fish stock assessment often involves manual fish counting by taxonomy specialists, which is both time-consuming and costly. We propose an automated computer vision system that performs both taxonomic classification and fish size estimation from images taken with a low-cost digital camera. The system first performs object detection and segmentation using a Mask R-CNN to identify individual fish from images containing multiple fish, possibly consisting of different species. Then each fish species is classified and the predicted length using separate machine learning models. These models are trained on a dataset of 50,000 hand-annotated images containing 163 different fish species, ranging in length from 10cm to 250cm. Evaluated on held-out test data, our system achieves a $92\%$ intersection over union on the fish segmentation task, a $89\%$ top-1 classification accuracy on single fish species classification, and a $2.3$~cm mean error on
    
[^2]: 逻辑闭环：揭示大型视觉-语言模型中的对象幻觉

    Logical Closed Loop: Uncovering Object Hallucinations in Large Vision-Language Models

    [https://arxiv.org/abs/2402.11622](https://arxiv.org/abs/2402.11622)

    提出了一种基于逻辑闭环的框架（LogicCheckGPT），利用大型视觉-语言模型本身来检测和减轻对象幻觉。

    

    对象幻觉一直是阻碍大型视觉-语言模型（LVLMs）更广泛应用的软肋。对象幻觉是指LVLMs在图像中声称不存在的对象的现象。为了减轻对象幻觉，已经提出了指导调整和基于外部模型的检测方法，这两种方法要么需要大规模的计算资源，要么依赖于外部模型的检测结果。然而，仍然存在一个未深入探讨的领域，即利用LVLM本身来减轻对象幻觉。在这项工作中，我们采用了这样的直觉，即LVLM倾向于对存在的对象做出逻辑一致的反应，但对幻觉对象做出不一致的反应。因此，我们提出了基于逻辑闭环的对象幻觉检测和减轻框架，即LogicCheckGPT。具体来说，我们设计了逻辑一致性探测来提出具有逻辑性的问题。

    arXiv:2402.11622v1 Announce Type: cross  Abstract: Object hallucination has been an Achilles' heel which hinders the broader applications of large vision-language models (LVLMs). Object hallucination refers to the phenomenon that the LVLMs claim non-existent objects in the image. To mitigate the object hallucinations, instruction tuning and external model-based detection methods have been proposed, which either require large-scare computational resources or depend on the detection result of external models. However, there remains an under-explored field to utilize the LVLM itself to alleviate object hallucinations. In this work, we adopt the intuition that the LVLM tends to respond logically consistently for existent objects but inconsistently for hallucinated objects. Therefore, we propose a Logical Closed Loop-based framework for Object Hallucination Detection and Mitigation, namely LogicCheckGPT. In specific, we devise logical consistency probing to raise questions with logical corr
    
[^3]: 航班滑行安全的跑道物体分类器的鲁棒性评估

    Robustness Assessment of a Runway Object Classifier for Safe Aircraft Taxiing

    [https://arxiv.org/abs/2402.00035](https://arxiv.org/abs/2402.00035)

    本文介绍了对航班滑行安全的跑道物体分类器的鲁棒性评估，使用形式方法评估了该分类器对三种常见图像扰动类型的鲁棒性，并提出了一种利用单调性的方法。

    

    随着深度神经网络(DNNs)在许多计算问题上成为主要解决方案，航空业希望探索它们在减轻飞行员负担和改善运营安全方面的潜力。然而，在这类安全关键应用中使用DNNs需要进行彻底的认证过程。这一需求可以通过形式验证来解决，形式验证提供了严格的保证，例如证明某些误判的不存在。在本文中，我们使用Airbus当前正在开发的图像分类器DNN作为案例研究，旨在在飞机滑行阶段使用。我们使用形式方法来评估这个DNN对三种常见图像扰动类型的鲁棒性：噪声、亮度和对比度，以及它们的部分组合。这个过程涉及多次调用底层验证器，这可能在计算上是昂贵的；因此，我们提出了一种利用单调性的方法。

    As deep neural networks (DNNs) are becoming the prominent solution for many computational problems, the aviation industry seeks to explore their potential in alleviating pilot workload and in improving operational safety. However, the use of DNNs in this type of safety-critical applications requires a thorough certification process. This need can be addressed through formal verification, which provides rigorous assurances -- e.g.,~by proving the absence of certain mispredictions. In this case-study paper, we demonstrate this process using an image-classifier DNN currently under development at Airbus and intended for use during the aircraft taxiing phase. We use formal methods to assess this DNN's robustness to three common image perturbation types: noise, brightness and contrast, and some of their combinations. This process entails multiple invocations of the underlying verifier, which might be computationally expensive; and we therefore propose a method that leverages the monotonicity
    
[^4]: 学习可堆叠和可跳过的乐高积木以实现高效、可重构和可变分辨率的扩散建模

    Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling. (arXiv:2310.06389v1 [cs.CV])

    [http://arxiv.org/abs/2310.06389](http://arxiv.org/abs/2310.06389)

    本研究提出了乐高积木，通过集成局部特征丰富和全局内容协调，实现了高效且可自适应的迭代细化扩散建模。这些积木可以堆叠在一起，用于在测试时根据需要进行重构，从而减少采样成本并生成高分辨率图像。

    

    扩散模型在生成真实感图像方面表现出色，但在训练和采样方面具有显著的计算成本。尽管有各种技术来解决这些计算挑战，但一个较少探索的问题是设计一个高效且适应性强的网络骨干，用于迭代细化。当前的选项如U-Net和Vision Transformer通常依赖于资源密集型的深度网络，缺乏在变量分辨率下生成图像或使用比训练中更小的网络所需的灵活性。本研究引入了乐高积木，它们无缝集成了局部特征丰富和全局内容协调。这些积木可以堆叠在一起，创建一个测试时可重构的扩散骨干，允许选择性跳过积木以减少采样成本，并生成比训练数据更高分辨率的图像。乐高积木通过MLP对局部区域进行丰富，并使用Transformer块进行变换，同时保持一致的全分辨率

    Diffusion models excel at generating photo-realistic images but come with significant computational costs in both training and sampling. While various techniques address these computational challenges, a less-explored issue is designing an efficient and adaptable network backbone for iterative refinement. Current options like U-Net and Vision Transformer often rely on resource-intensive deep networks and lack the flexibility needed for generating images at variable resolutions or with a smaller network than used in training. This study introduces LEGO bricks, which seamlessly integrate Local-feature Enrichment and Global-content Orchestration. These bricks can be stacked to create a test-time reconfigurable diffusion backbone, allowing selective skipping of bricks to reduce sampling costs and generate higher-resolution images than the training data. LEGO bricks enrich local regions with an MLP and transform them using a Transformer block while maintaining a consistent full-resolution i
    
[^5]: 丢弃模式的生成自编码器

    Generative Autoencoding of Dropout Patterns. (arXiv:2310.01712v1 [cs.LG])

    [http://arxiv.org/abs/2310.01712](http://arxiv.org/abs/2310.01712)

    本论文提出了一种称为解读自编码器的生成模型，通过为训练数据集中的每个数据点分配独特的随机丢弃模式来进行训练，只依靠重构误差来提供更稳定的训练性能，并在CIFAR-10数据集上展示了与DCGAN相媲美的采样质量。

    

    我们提出了一种称为解读自编码器的生成模型。在这个模型中，我们为训练数据集中的每个数据点分配一个唯一的随机丢弃模式，然后使用这个模式作为被编码的信息来训练自编码器来重构相应的数据点。由于解读自编码器的训练仅依赖于重构误差，所以相比其他生成模型，它具有更稳定的训练性能。尽管它很简单，但解读自编码器在CIFAR-10数据集上展现出了与DCGAN相媲美的采样质量。

    We propose a generative model termed Deciphering Autoencoders. In this model, we assign a unique random dropout pattern to each data point in the training dataset and then train an autoencoder to reconstruct the corresponding data point using this pattern as information to be encoded. Since the training of Deciphering Autoencoders relies solely on reconstruction error, it offers more stable training than other generative models. Despite its simplicity, Deciphering Autoencoders show comparable sampling quality to DCGAN on the CIFAR-10 dataset.
    

