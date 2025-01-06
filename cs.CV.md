# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cheating Suffix: Targeted Attack to Text-To-Image Diffusion Models with Multi-Modal Priors](https://rss.arxiv.org/abs/2402.01369) | 本文提出了一种名为MMP-Attack的有针对性攻击方法，通过整合文本和图像特征，该方法能够有效地攻击商业文本到图像模型，并且具有更高的普适性和可转移性。 |
| [^2] | [Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data.](http://arxiv.org/abs/2401.15113) | 本研究提出了一种使用深度学习和开放地球观测数据进行全球冰川制图的方法，通过新的模型和策略，在多种地形和传感器上实现了较高的准确性。通过添加合成孔径雷达数据，并报告冰川范围的校准置信度，提高了预测的可靠性和可解释性。 |
| [^3] | [HPCR: Holistic Proxy-based Contrastive Replay for Online Continual Learning.](http://arxiv.org/abs/2309.15038) | HPCR是一种用于在线连续学习的新方法，该方法综合了基于代理和对比损失的重放方式。通过在对比损失中使用锚点-代理对替换锚点-样本对，HPCR能够减轻遗忘现象，并有效学习更细粒度的语义信息。实验证明，HPCR在多个任务上实现了最先进的性能。 |
| [^4] | [Experimental Security Analysis of DNN-based Adaptive Cruise Control under Context-Aware Perception Attacks.](http://arxiv.org/abs/2307.08939) | 这项研究评估了基于深度神经网络的自适应巡航控制系统在隐蔽感知攻击下的安全性，并提出了一种上下文感知策略和基于优化的图像扰动生成方法。 |

# 详细

[^1]: 使用多模式先验的有针对性攻击文本到图像扩散模型

    Cheating Suffix: Targeted Attack to Text-To-Image Diffusion Models with Multi-Modal Priors

    [https://rss.arxiv.org/abs/2402.01369](https://rss.arxiv.org/abs/2402.01369)

    本文提出了一种名为MMP-Attack的有针对性攻击方法，通过整合文本和图像特征，该方法能够有效地攻击商业文本到图像模型，并且具有更高的普适性和可转移性。

    

    扩散模型已广泛应用于各种图像生成任务中，展现了图像和文本模态之间的卓越联系。然而，它们面临着被恶意利用的挑战，通过在原始提示后附加特定后缀来生成有害或敏感图像。现有作品主要关注使用单模态信息进行攻击，未能利用多模态特征，导致性能不尽如人意。在本工作中，我们提出了一种名为MMP-Attack的有针对性攻击方法，它将多模态先验（MMP）即文本和图像特征进行整合。具体而言，MMP-Attack的目标是在图像内容中添加目标对象的同时，同时移除原始对象。与现有作品相比，MMP-Attack具有更高的普适性和可转移性，在攻击商业文本到图像（T2I）模型（如DALL-E 3）方面表现出明显优势。据我们所知，这标志着当前最佳的技术水平。

    Diffusion models have been widely deployed in various image generation tasks, demonstrating an extraordinary connection between image and text modalities. However, they face challenges of being maliciously exploited to generate harmful or sensitive images by appending a specific suffix to the original prompt. Existing works mainly focus on using single-modal information to conduct attacks, which fails to utilize multi-modal features and results in less than satisfactory performance. Integrating multi-modal priors (MMP), i.e. both text and image features, we propose a targeted attack method named MMP-Attack in this work. Specifically, the goal of MMP-Attack is to add a target object into the image content while simultaneously removing the original object. The MMP-Attack shows a notable advantage over existing works with superior universality and transferability, which can effectively attack commercial text-to-image (T2I) models such as DALL-E 3. To the best of our knowledge, this marks 
    
[^2]: 使用深度学习和开放地球观测数据实现全球冰川制图

    Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data. (arXiv:2401.15113v1 [cs.CV])

    [http://arxiv.org/abs/2401.15113](http://arxiv.org/abs/2401.15113)

    本研究提出了一种使用深度学习和开放地球观测数据进行全球冰川制图的方法，通过新的模型和策略，在多种地形和传感器上实现了较高的准确性。通过添加合成孔径雷达数据，并报告冰川范围的校准置信度，提高了预测的可靠性和可解释性。

    

    准确的全球冰川制图对于理解气候变化的影响至关重要。这个过程受到冰川多样性、难以分类的碎石和大数据处理的挑战。本文提出了Glacier-VisionTransformer-U-Net (GlaViTU)，一个卷积-Transformer深度学习模型，并提出了五种利用开放卫星影像进行多时相全球冰川制图的策略。空间、时间和跨传感器的泛化性能评估表明，我们的最佳策略在大多数情况下实现了IoU（交并比）> 0.85，并且在以冰雪为主的地区增加到了> 0.90，而在高山亚洲等碎石丰富的区域则降至> 0.75。此外，添加合成孔径雷达数据，即回波和干涉相干度，可以提高所有可用地区的准确性。报告冰川范围的校准置信度使预测更可靠和可解释。我们还发布了一个基准数据集。

    Accurate global glacier mapping is critical for understanding climate change impacts. It is challenged by glacier diversity, difficult-to-classify debris and big data processing. Here we propose Glacier-VisionTransformer-U-Net (GlaViTU), a convolutional-transformer deep learning model, and five strategies for multitemporal global-scale glacier mapping using open satellite imagery. Assessing the spatial, temporal and cross-sensor generalisation shows that our best strategy achieves intersection over union >0.85 on previously unobserved images in most cases, which drops to >0.75 for debris-rich areas such as High-Mountain Asia and increases to >0.90 for regions dominated by clean ice. Additionally, adding synthetic aperture radar data, namely, backscatter and interferometric coherence, increases the accuracy in all regions where available. The calibrated confidence for glacier extents is reported making the predictions more reliable and interpretable. We also release a benchmark dataset 
    
[^3]: HPCR: 基于代理的综合对比重放用于在线连续学习

    HPCR: Holistic Proxy-based Contrastive Replay for Online Continual Learning. (arXiv:2309.15038v1 [cs.LG])

    [http://arxiv.org/abs/2309.15038](http://arxiv.org/abs/2309.15038)

    HPCR是一种用于在线连续学习的新方法，该方法综合了基于代理和对比损失的重放方式。通过在对比损失中使用锚点-代理对替换锚点-样本对，HPCR能够减轻遗忘现象，并有效学习更细粒度的语义信息。实验证明，HPCR在多个任务上实现了最先进的性能。

    

    在线连续学习（OCL）旨在通过一次在线数据流传递持续学习新数据。然而，它通常会面临灾难性遗忘问题。现有的基于重放的方法通过以代理为基础或对比为基础的重放方式有效地缓解了这个问题。在本文中，我们对这两种重放方式进行了全面分析，并发现它们可以相互补充。受到这一发现的启发，我们提出了一种新颖的基于重放的方法称为代理对比重放（PCR），它将对比损失中的锚点-样本对替换为锚点-代理对，以减轻遗忘现象。基于PCR，我们进一步开发了一种更高级的方法，称为综合代理对比重放（HPCR），它由三个组件组成。对比组件在PCR的基础上条件性地将锚点-样本对纳入其中，通过大型训练批次学习更细粒度的语义信息。第二个组件是重放组件，它在样本选择上采用了多样性策略，以确保代理数据与当前任务具有更高的关联性。第三个组件是正则化组件，通过缩小样本空间，促进学习模型对任务特定特征的更好表示。实验证明，HPCR方法在多个在线连续学习任务上实现了最先进的性能。

    Online continual learning (OCL) aims to continuously learn new data from a single pass over the online data stream. It generally suffers from the catastrophic forgetting issue. Existing replay-based methods effectively alleviate this issue by replaying part of old data in a proxy-based or contrastive-based replay manner. In this paper, we conduct a comprehensive analysis of these two replay manners and find they can be complementary. Inspired by this finding, we propose a novel replay-based method called proxy-based contrastive replay (PCR), which replaces anchor-to-sample pairs with anchor-to-proxy pairs in the contrastive-based loss to alleviate the phenomenon of forgetting. Based on PCR, we further develop a more advanced method named holistic proxy-based contrastive replay (HPCR), which consists of three components. The contrastive component conditionally incorporates anchor-to-sample pairs to PCR, learning more fine-grained semantic information with a large training batch. The sec
    
[^4]: 基于深度神经网络的自适应巡航控制在上下文感知攻击下的安全性实验分析

    Experimental Security Analysis of DNN-based Adaptive Cruise Control under Context-Aware Perception Attacks. (arXiv:2307.08939v1 [cs.CR])

    [http://arxiv.org/abs/2307.08939](http://arxiv.org/abs/2307.08939)

    这项研究评估了基于深度神经网络的自适应巡航控制系统在隐蔽感知攻击下的安全性，并提出了一种上下文感知策略和基于优化的图像扰动生成方法。

    

    自适应巡航控制（ACC）是一种广泛应用的驾驶员辅助功能，用于保持期望速度和与前方车辆的安全距离。本文评估基于深度神经网络（DNN）的ACC系统在隐蔽感知攻击下的安全性，该攻击会对摄像机数据进行有针对性的扰动，以导致前方碰撞事故。我们提出了一种基于知识和数据驱动的方法，设计了一种上下文感知策略，用于选择触发攻击最关键的时间点，并采用了一种新颖的基于优化的方法，在运行时生成适应性图像扰动。我们使用实际驾驶数据集和逼真的仿真平台评估了所提出攻击的有效性，该仿真平台使用了来自生产ACC系统的控制软件和物理世界驾驶模拟器，并考虑了驾驶员的干预以及自动紧急制动（AEB）和前向碰撞警示（FCW）等安全功能。

    Adaptive Cruise Control (ACC) is a widely used driver assistance feature for maintaining desired speed and safe distance to the leading vehicles. This paper evaluates the security of the deep neural network (DNN) based ACC systems under stealthy perception attacks that strategically inject perturbations into camera data to cause forward collisions. We present a combined knowledge-and-data-driven approach to design a context-aware strategy for the selection of the most critical times for triggering the attacks and a novel optimization-based method for the adaptive generation of image perturbations at run-time. We evaluate the effectiveness of the proposed attack using an actual driving dataset and a realistic simulation platform with the control software from a production ACC system and a physical-world driving simulator while considering interventions by the driver and safety features such as Automatic Emergency Braking (AEB) and Forward Collision Warning (FCW). Experimental results sh
    

