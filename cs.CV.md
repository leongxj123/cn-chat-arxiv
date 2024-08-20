# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection](https://arxiv.org/abs/2403.17465) | LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。 |
| [^2] | [Chain of Compression: A Systematic Approach to Combinationally Compress Convolutional Neural Networks](https://arxiv.org/abs/2403.17447) | 提出了一种名为“压缩链”的系统化方法，通过结合量化、剪枝、提前退出和知识蒸馏等常见技术，实现对卷积神经网络的压缩。 |
| [^3] | [MatchSeg: Towards Better Segmentation via Reference Image Matching](https://arxiv.org/abs/2403.15901) | 通过引入MatchSeg框架，利用对比语言-图像预训练和联合注意力模块增强了医学图像分割，有效实现了支持集和查询集之间的知识转移。 |
| [^4] | [MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](https://arxiv.org/abs/2403.14624) | MathVerse是一个全方位的视觉数学基准测试，旨在公平而深入地评估MLLMs在视觉数学问题解决中的能力。 |
| [^5] | [Tensor network compressibility of convolutional models](https://arxiv.org/abs/2403.14379) | 张量化是将卷积神经网络中的卷积核替换为紧凑分解，并直接训练分解因子以偏向于低秩分解的方法，该研究探讨了张量化如何通过评估截断卷积核来保持准确性。 |
| [^6] | [Source Matters: Source Dataset Impact on Model Robustness in Medical Imaging](https://arxiv.org/abs/2403.04484) | 该研究调查了在医学成像中，源数据集的选择如何影响模型的鲁棒性，指出ImageNet预训练模型更容易过拟合混杂因素，建议研究人员重新评估模型的鲁棒性。 |
| [^7] | [Wilcoxon Nonparametric CFAR Scheme for Ship Detection in SAR Image](https://arxiv.org/abs/2402.18579) | 提出并分析了用于SAR图像中船只检测的Wilcoxon非参数CFAR方案，可以在没有已知杂波分布假设的情况下维持目标检测的恒定虚警率 |
| [^8] | [WeakSAM: Segment Anything Meets Weakly-supervised Instance-level Recognition](https://arxiv.org/abs/2402.14812) | WeakSAM通过利用预先学习的全球知识，解决了弱监督对象检测和分割问题，提出了自适应PGT生成和RoI丢弃正则化，显著超越了先前的最先进方法。 |
| [^9] | [Hybrid Reasoning Based on Large Language Models for Autonomous Car Driving](https://arxiv.org/abs/2402.13602) | 大型语言模型在自动驾驶中的混合推理能力可以通过分析数据、理解规则和法则、提供语境等方式提高自动驶驶的决策能力。 |
| [^10] | [CIC: A framework for Culturally-aware Image Captioning](https://arxiv.org/abs/2402.05374) | CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。 |
| [^11] | [PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems.](http://arxiv.org/abs/2401.08903) | 本文提出了一种名为PPR的新型攻击方法，旨在增强躲避攻击的性能同时避免冒名顶替攻击的降级。该方法利用对抗样本修剪，并通过嵌入对抗扰动来增强对抗人脸样本的躲避性能。 |
| [^12] | [Not all Minorities are Equal: Empty-Class-Aware Distillation for Heterogeneous Federated Learning.](http://arxiv.org/abs/2401.02329) | 本研究提出了一种异质联邦学习方法FedED，通过同时进行空类别蒸馏和逻辑抑制，解决了在联邦学习中尚未充分识别空类别的问题。 |
| [^13] | [On Responsible Machine Learning Datasets with Fairness, Privacy, and Regulatory Norms.](http://arxiv.org/abs/2310.15848) | 这篇论文讨论了负责任的机器学习数据集的重要性，并提出了一个通过负责任评价标准来评估数据集的框架。 |
| [^14] | [CRITERIA: a New Benchmarking Paradigm for Evaluating Trajectory Prediction Models for Autonomous Driving.](http://arxiv.org/abs/2310.07794) | CRITERIA是一种新的基准测试方法，用于评估自动驾驶轨迹预测模型。它通过精细排名预测，提供了关于模型性能在不同情况下的洞察。 |
| [^15] | [Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors.](http://arxiv.org/abs/2303.04238) | 本文提出了一种基于GAN的无梯度物理对抗攻击方法，用于生成自然的对抗补丁，攻击物体检测器，具有实际应用价值。 |
| [^16] | [RGMIM: Region-Guided Masked Image Modeling for COVID-19 Detection.](http://arxiv.org/abs/2211.00313) | 本论文提出了一种针对COVID-19检测的新颖区域引导的掩膜图像建模方法，该方法通过利用肺掩模信息来识别有效区域，以学习更有用的COVID-19检测信息。 |

# 详细

[^1]: LaRE^2: 基于潜在重构误差的扩散生成图像检测方法

    LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection

    [https://arxiv.org/abs/2403.17465](https://arxiv.org/abs/2403.17465)

    LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。

    

    arXiv:2403.17465v1 类型：交叉 摘要：扩散模型的发展显著提高了图像生成质量，使真实图像和生成图像之间的区分变得越来越困难。尽管这一进展令人印象深刻，但也引发了重要的隐私和安全问题。为了解决这一问题，我们提出了一种新颖的基于潜在重构误差引导特征细化方法（LaRE^2）来检测扩散生成的图像。我们提出了潜在重构误差（LaRE），作为潜在空间中生成图像检测的第一个基于重构误差的特征。LaRE在特征提取效率方面超越了现有方法，同时保留了区分真假所需的关键线索。为了利用LaRE，我们提出了一种误差引导特征细化模块（EGRE），它可以通过LaRE引导的方式细化图像特征，以增强特征的区分能力。

    arXiv:2403.17465v1 Announce Type: cross  Abstract: The evolution of Diffusion Models has dramatically improved image generation quality, making it increasingly difficult to differentiate between real and generated images. This development, while impressive, also raises significant privacy and security concerns. In response to this, we propose a novel Latent REconstruction error guided feature REfinement method (LaRE^2) for detecting the diffusion-generated images. We come up with the Latent Reconstruction Error (LaRE), the first reconstruction-error based feature in the latent space for generated image detection. LaRE surpasses existing methods in terms of feature extraction efficiency while preserving crucial cues required to differentiate between the real and the fake. To exploit LaRE, we propose an Error-Guided feature REfinement module (EGRE), which can refine the image feature guided by LaRE to enhance the discriminativeness of the feature. Our EGRE utilizes an align-then-refine m
    
[^2]: 压缩链：一种系统化的组合压缩卷积神经网络方法

    Chain of Compression: A Systematic Approach to Combinationally Compress Convolutional Neural Networks

    [https://arxiv.org/abs/2403.17447](https://arxiv.org/abs/2403.17447)

    提出了一种名为“压缩链”的系统化方法，通过结合量化、剪枝、提前退出和知识蒸馏等常见技术，实现对卷积神经网络的压缩。

    

    卷积神经网络（CNNs）已经取得了显著的流行，但它们在计算和存储方面的密集性给资源有限的计算系统带来了挑战，尤其是在需要实时性能的情况下。为了减轻负担，模型压缩已经成为一个重要的研究重点。许多方法，如量化、剪枝、提前退出和知识蒸馏已经证明了减少神经网络中冗余的效果。通过进一步的研究，可以明显看出，每种方法都利用了其独特的特性来压缩神经网络，并且当它们结合在一起时也可以展现出互补的行为。为了探究这些相互作用，并从互补特性中获益，我们提出了压缩链，它在组合序列上操作，应用这些常见技术来压缩神经网络。

    arXiv:2403.17447v1 Announce Type: new  Abstract: Convolutional neural networks (CNNs) have achieved significant popularity, but their computational and memory intensity poses challenges for resource-constrained computing systems, particularly with the prerequisite of real-time performance. To release this burden, model compression has become an important research focus. Many approaches like quantization, pruning, early exit, and knowledge distillation have demonstrated the effect of reducing redundancy in neural networks. Upon closer examination, it becomes apparent that each approach capitalizes on its unique features to compress the neural network, and they can also exhibit complementary behavior when combined. To explore the interactions and reap the benefits from the complementary features, we propose the Chain of Compression, which works on the combinational sequence to apply these common techniques to compress the neural network. Validated on the image-based regression and classi
    
[^3]: 通过参考图像匹配实现更好的分割：MatchSeg

    MatchSeg: Towards Better Segmentation via Reference Image Matching

    [https://arxiv.org/abs/2403.15901](https://arxiv.org/abs/2403.15901)

    通过引入MatchSeg框架，利用对比语言-图像预训练和联合注意力模块增强了医学图像分割，有效实现了支持集和查询集之间的知识转移。

    

    最近，基于深度学习的自动医学图像分割方法取得了巨大成功。然而，它们严重依赖于大量的标注数据集，而获取这些数据集的成本高昂且耗时。Few-shot learning旨在通过使用一个小型标记数据集（称为支持集）来指导预测新的、未标记图像（称为查询集）的标签，从而克服对标注数据的需求。受到这一范式的启发，我们引入了MatchSeg，这是一个通过战略性参考图像匹配增强医学图像分割的新框架。我们利用对比语言-图像预训练（CLIP）在定义支持集时选择高度相关的样本。此外，我们设计了联合注意力模块来加强支持和查询特征之间的交互，促进更有效的知识转移。我们在四个公共数据集上验证了我们的方法。

    arXiv:2403.15901v1 Announce Type: new  Abstract: Recently, automated medical image segmentation methods based on deep learning have achieved great success. However, they heavily rely on large annotated datasets, which are costly and time-consuming to acquire. Few-shot learning aims to overcome the need for annotated data by using a small labeled dataset, known as a support set, to guide predicting labels for new, unlabeled images, known as the query set. Inspired by this paradigm, we introduce MatchSeg, a novel framework that enhances medical image segmentation through strategic reference image matching. We leverage contrastive language-image pre-training (CLIP) to select highly relevant samples when defining the support set. Additionally, we design a joint attention module to strengthen the interaction between support and query features, facilitating a more effective knowledge transfer between support and query sets. We validated our method across four public datasets. Experimental re
    
[^4]: MathVerse：您的多模式LLM是否真正看到了视觉数学问题中的图表？

    MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?

    [https://arxiv.org/abs/2403.14624](https://arxiv.org/abs/2403.14624)

    MathVerse是一个全方位的视觉数学基准测试，旨在公平而深入地评估MLLMs在视觉数学问题解决中的能力。

    

    多模式大型语言模型（MLLMs）取得了显著进展，在视觉环境中表现出色，然而它们在视觉数学问题解决方面的能力仍未充分评估和理解。本研究调查了当前基准测试，将过多的视觉内容融入文本问题中，这有助于MLLM在不真正解释输入图表的情况下推导答案。为此，我们介绍了MathVerse，这是一个全方位的视觉数学基准测试，旨在公平而深入地评估MLLMs。我们精心收集了2,612个高质量的多学科数学问题，其中包含图表，来源于公开渠道。然后，每个问题由人工注释者转化为六个不同版本，每个版本在多模式中提供不同程度的信息内容，共贡献了15K个测试样本。这种方法使得MathVerse能够同时

    arXiv:2403.14624v1 Announce Type: cross  Abstract: The remarkable progress of Multi-modal Large Language Models (MLLMs) has garnered unparalleled attention, due to their superior performance in visual contexts. However, their capabilities in visual math problem-solving remain insufficiently evaluated and understood. We investigate current benchmarks to incorporate excessive visual content within textual questions, which potentially assist MLLMs in deducing answers without truly interpreting the input diagrams. To this end, we introduce MathVerse, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. We meticulously collect 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources. Each problem is then transformed by human annotators into six distinct versions, each offering varying degrees of information content in multi-modality, contributing to 15K test samples in total. This approach allows MathVerse to co
    
[^5]: 卷积模型的张量网络可压缩性

    Tensor network compressibility of convolutional models

    [https://arxiv.org/abs/2403.14379](https://arxiv.org/abs/2403.14379)

    张量化是将卷积神经网络中的卷积核替换为紧凑分解，并直接训练分解因子以偏向于低秩分解的方法，该研究探讨了张量化如何通过评估截断卷积核来保持准确性。

    

    卷积神经网络（CNNs）代表了最广泛使用的神经网络架构之一，在计算机视觉任务中展示了最先进的性能。尽管一般情况下更大的CNNs通常表现出更高的准确性，但通过“张量化”可以有效地减小它们的大小，同时保持准确性。张量化包括将卷积核替换为如Tucker、Canonical Polyadic分解或受量子启发的分解（如矩阵乘积状态）等紧凑的分解，并直接训练分解中的因子，以偏向于低秩分解。但为什么张量化似乎对准确性没有不利影响？我们通过评估截断密集（非张量化）CNNs的卷积核对其准确性的影响来探讨这一点。

    arXiv:2403.14379v1 Announce Type: cross  Abstract: Convolutional neural networks (CNNs) represent one of the most widely used neural network architectures, showcasing state-of-the-art performance in computer vision tasks. Although larger CNNs generally exhibit higher accuracy, their size can be effectively reduced by "tensorization" while maintaining accuracy. Tensorization consists of replacing the convolution kernels with compact decompositions such as Tucker, Canonical Polyadic decompositions, or quantum-inspired decompositions such as matrix product states, and directly training the factors in the decompositions to bias the learning towards low-rank decompositions. But why doesn't tensorization seem to impact the accuracy adversely? We explore this by assessing how truncating the convolution kernels of dense (untensorized) CNNs impact their accuracy. Specifically, we truncated the kernels of (i) a vanilla four-layer CNN and (ii) ResNet-50 pre-trained for image classification on CIF
    
[^6]: 来源至关重要：医学成像中数据集对模型鲁棒性的影响

    Source Matters: Source Dataset Impact on Model Robustness in Medical Imaging

    [https://arxiv.org/abs/2403.04484](https://arxiv.org/abs/2403.04484)

    该研究调查了在医学成像中，源数据集的选择如何影响模型的鲁棒性，指出ImageNet预训练模型更容易过拟合混杂因素，建议研究人员重新评估模型的鲁棒性。

    

    迁移学习已成为医学成像分类算法中不可或缺的一部分，通常利用ImageNet权重。然而，从自然到医学图像的领域转变促使了诸如RadImageNet 等替代方案的出现，往往展示出可比的分类性能。然而，目前尚不清楚迁移学习中性能提升是来自于改善的泛化还是快捷学习。为了解决这个问题，我们研究了两个公开的胸部X光片和CT数据集之间的潜在混杂因素--无论是合成的还是从数据中抽取的。我们发现ImageNet 和 RadImageNet 实现了可比的分类性能，然而 ImageNet 更容易过拟合混杂因素。我们建议使用ImageNet预训练模型的研究人员通过开展类似实验来重新审视模型的鲁棒性。我们的代码和实验可在https://github.com/DovileDo/source-mat 获取。

    arXiv:2403.04484v1 Announce Type: cross  Abstract: Transfer learning has become an essential part of medical imaging classification algorithms, often leveraging ImageNet weights. However, the domain shift from natural to medical images has prompted alternatives such as RadImageNet, often demonstrating comparable classification performance. However, it remains unclear whether the performance gains from transfer learning stem from improved generalization or shortcut learning. To address this, we investigate potential confounders -- whether synthetic or sampled from the data -- across two publicly available chest X-ray and CT datasets. We show that ImageNet and RadImageNet achieve comparable classification performance, yet ImageNet is much more prone to overfitting to confounders. We recommend that researchers using ImageNet-pretrained models reexamine their model robustness by conducting similar experiments. Our code and experiments are available at https://github.com/DovileDo/source-mat
    
[^7]: SAR图像中用于船只检测的Wilcoxon非参数化CFAR方案

    Wilcoxon Nonparametric CFAR Scheme for Ship Detection in SAR Image

    [https://arxiv.org/abs/2402.18579](https://arxiv.org/abs/2402.18579)

    提出并分析了用于SAR图像中船只检测的Wilcoxon非参数CFAR方案，可以在没有已知杂波分布假设的情况下维持目标检测的恒定虚警率

    

    常数虚警率（CFAR）检测算法广泛应用于目前SAR图像中检测船只目标，这些算法基于各种统计分布，如高斯分布、Gamma分布、Weibull分布、对数正态分布、G0分布、alpha稳定分布等。然而，SAR图像中的杂散背景复杂多变。当实际杂散背景偏离假定的统计分布时，参数化CFAR检测器的性能将下降。除了参数化CFAR方案，还有另一类非参数化CFAR检测器，可以在没有已知杂波分布的假设情况下保持目标检测的恒定虚警率。在这项工作中，提出并分析了用于SAR图像中船只检测的Wilcoxon非参数化CFAR方案，并推导了Wilcoxon非参数检测器的虚警率的封闭形式以确定

    arXiv:2402.18579v1 Announce Type: cross  Abstract: The parametric constant false alarm rate (CFAR) detection algorithms which are based on various statistical distributions, such as Gaussian, Gamma, Weibull, log-normal, G0 distribution, alpha-stable distribution, etc, are most widely used to detect the ship targets in SAR image at present. However, the clutter background in SAR images is complicated and variable. When the actual clutter background deviates from the assumed statistical distribution, the performance of the parametric CFAR detector will deteriorate. In addition to the parametric CFAR schemes, there is another class of nonparametric CFAR detectors which can maintain a constant false alarm rate for the target detection without the assumption of a known clutter distribution. In this work, the Wilcoxon nonparametric CFAR scheme for ship detection in SAR image is proposed and analyzed, and a closed form of the false alarm rate for the Wilcoxon nonparametric detector to determi
    
[^8]: WeakSAM: 任意分割遇上弱监督实例级别识别

    WeakSAM: Segment Anything Meets Weakly-supervised Instance-level Recognition

    [https://arxiv.org/abs/2402.14812](https://arxiv.org/abs/2402.14812)

    WeakSAM通过利用预先学习的全球知识，解决了弱监督对象检测和分割问题，提出了自适应PGT生成和RoI丢弃正则化，显著超越了先前的最先进方法。

    

    弱监督的视觉识别使用不精确的监督是一个关键但具有挑战性的学习问题。它显著降低了人工标注成本，并且传统上依赖多实例学习和伪标签。本文介绍了WeakSAM，并通过利用包含在视觉基础模型中的预先学习的全球知识，即Segment Anything Model (SAM)，来解决弱监督物体检测（WSOD）和分割。WeakSAM通过自适应PGT生成和感兴趣区域（RoI）丢弃正则化，解决了传统WSOD重新训练中的两个关键限制，即伪标准地面真相（PGT）的不完整性和具有嘈杂PGT实例。它还解决了SAM在自动对象检测和分割时需要提示和类别无感知性的问题。我们的结果表明，WeakSAM在WSOD和WSIS基准测试中显著超越了先前的最先进方法。

    arXiv:2402.14812v1 Announce Type: cross  Abstract: Weakly supervised visual recognition using inexact supervision is a critical yet challenging learning problem. It significantly reduces human labeling costs and traditionally relies on multi-instance learning and pseudo-labeling. This paper introduces WeakSAM and solves the weakly-supervised object detection (WSOD) and segmentation by utilizing the pre-learned world knowledge contained in a vision foundation model, i.e., the Segment Anything Model (SAM). WeakSAM addresses two critical limitations in traditional WSOD retraining, i.e., pseudo ground truth (PGT) incompleteness and noisy PGT instances, through adaptive PGT generation and Region of Interest (RoI) drop regularization. It also addresses the SAM's problems of requiring prompts and category unawareness for automatic object detection and segmentation. Our results indicate that WeakSAM significantly surpasses previous state-of-the-art methods in WSOD and WSIS benchmarks with larg
    
[^9]: 基于大型语言模型的混合推理在自动驾驶中的应用

    Hybrid Reasoning Based on Large Language Models for Autonomous Car Driving

    [https://arxiv.org/abs/2402.13602](https://arxiv.org/abs/2402.13602)

    大型语言模型在自动驾驶中的混合推理能力可以通过分析数据、理解规则和法则、提供语境等方式提高自动驶驶的决策能力。

    

    大型语言模型（LLMs）因其理解文本和图像、生成类人文本以及执行复杂推理任务的能力而受到广泛关注。然而，它们将这种高级推理与自然语言文本相结合以用于动态情况下的决策的泛化能力需要进一步探索。本研究探讨了LLMs在混合算术和常识推理方面的适应能力和应用能力，特别是在自动驾驶场景中。我们假设LLMs的混合推理能力可以通过使它们分析检测到的物体和传感器数据、理解驾驶规定和物理法则，并提供额外的语境来改善自动驾驶。这解决了复杂情景，如低能见度（由于天气条件）下的决策，传统方法可能不足以胜任。我们通过准确性评估基于大型语言模型（LLMs）的这种能力。

    arXiv:2402.13602v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have garnered significant attention for their ability to understand text and images, generate human-like text, and perform complex reasoning tasks. However, their ability to generalize this advanced reasoning with a combination of natural language text for decision-making in dynamic situations requires further exploration. In this study, we investigate how well LLMs can adapt and apply a combination of arithmetic and common-sense reasoning, particularly in autonomous driving scenarios. We hypothesize that LLMs hybrid reasoning abilities can improve autonomous driving by enabling them to analyze detected object and sensor data, understand driving regulations and physical laws, and offer additional context. This addresses complex scenarios, like decisions in low visibility (due to weather conditions), where traditional methods might fall short. We evaluated Large Language Models (LLMs) based on accuracy by co
    
[^10]: CIC：一种面向文化感知图像字幕的框架

    CIC: A framework for Culturally-aware Image Captioning

    [https://arxiv.org/abs/2402.05374](https://arxiv.org/abs/2402.05374)

    CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。

    

    图像字幕通过使用视觉-语言预训练模型（VLPs）如BLIP从图像生成描述性句子，这种方法已经取得了很大的改进。然而，当前的方法缺乏对图像中所描绘的文化元素（例如亚洲文化群体的传统服装）生成详细描述性字幕的能力。在本文中，我们提出了一种新的框架，\textbf{面向文化感知图像字幕（CIC）}，该框架能够从代表不同文化的图像中生成字幕并描述文化元素。受到将视觉模态和大型语言模型（LLMs）通过适当的提示进行组合的方法的启发，我们的框架（1）根据图像中的文化类别生成问题，（2）利用生成的问题从视觉问答（VQA）中提取文化视觉元素，（3）使用带有提示的LLMs生成文化感知字幕。我们在4个不同大学的45名参与者上进行了人工评估。

    Image Captioning generates descriptive sentences from images using Vision-Language Pre-trained models (VLPs) such as BLIP, which has improved greatly. However, current methods lack the generation of detailed descriptive captions for the cultural elements depicted in the images, such as the traditional clothing worn by people from Asian cultural groups. In this paper, we propose a new framework, \textbf{Culturally-aware Image Captioning (CIC)}, that generates captions and describes cultural elements extracted from cultural visual elements in images representing cultures. Inspired by methods combining visual modality and Large Language Models (LLMs) through appropriate prompts, our framework (1) generates questions based on cultural categories from images, (2) extracts cultural visual elements from Visual Question Answering (VQA) using generated questions, and (3) generates culturally-aware captions using LLMs with the prompts. Our human evaluation conducted on 45 participants from 4 dif
    
[^11]: PPR: 在维持冒名顶替攻击的同时增强躲避攻击对人脸识别系统的影响

    PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems. (arXiv:2401.08903v1 [cs.CV])

    [http://arxiv.org/abs/2401.08903](http://arxiv.org/abs/2401.08903)

    本文提出了一种名为PPR的新型攻击方法，旨在增强躲避攻击的性能同时避免冒名顶替攻击的降级。该方法利用对抗样本修剪，并通过嵌入对抗扰动来增强对抗人脸样本的躲避性能。

    

    人脸识别系统上的对抗攻击可以分为两种类型：冒名顶替攻击和躲避攻击。我们观察到，在黑盒设置中成功进行冒名顶替攻击并不一定能保证在人脸识别系统上成功进行躲避攻击。本文提出了一种名为预训练修剪恢复攻击（PPR）的新型攻击方法，旨在增强躲避攻击的性能同时避免冒名顶替攻击的降级。我们的方法利用对抗样本修剪，可以将一部分对抗扰动设为零，并倾向于保持攻击性能。通过利用对抗样本修剪，我们可以修剪预训练的对抗样本，并有选择性地释放某些对抗扰动。然后，我们将对抗扰动嵌入到修剪区域，从而增强对抗人脸样本的躲避性能。通过实验证明了我们提出的攻击方法的有效性。

    Adversarial Attacks on Face Recognition (FR) encompass two types: impersonation attacks and evasion attacks. We observe that achieving a successful impersonation attack on FR does not necessarily ensure a successful dodging attack on FR in the black-box setting. Introducing a novel attack method named Pre-training Pruning Restoration Attack (PPR), we aim to enhance the performance of dodging attacks whilst avoiding the degradation of impersonation attacks. Our method employs adversarial example pruning, enabling a portion of adversarial perturbations to be set to zero, while tending to maintain the attack performance. By utilizing adversarial example pruning, we can prune the pre-trained adversarial examples and selectively free up certain adversarial perturbations. Thereafter, we embed adversarial perturbations in the pruned area, which enhances the dodging performance of the adversarial face examples. The effectiveness of our proposed attack method is demonstrated through our experim
    
[^12]: 不是所有的少数群体都是平等的: 空类别感知的异质联邦学习方法

    Not all Minorities are Equal: Empty-Class-Aware Distillation for Heterogeneous Federated Learning. (arXiv:2401.02329v1 [cs.LG])

    [http://arxiv.org/abs/2401.02329](http://arxiv.org/abs/2401.02329)

    本研究提出了一种异质联邦学习方法FedED，通过同时进行空类别蒸馏和逻辑抑制，解决了在联邦学习中尚未充分识别空类别的问题。

    

    数据异质性是联邦学习中的一个重大挑战，表现为客户端之间本地数据分布的差异。现有方法常常在本地训练过程中采用类别平衡的技术来解决本地类别分布的异质性问题。然而，在少数类别中由于过拟合本地不平衡数据而导致准确性较差的问题仍然存在。本文提出了FedED，这是一种新颖的异质联邦学习方法，同时整合了空类别蒸馏和逻辑抑制。具体而言，空类别蒸馏利用知识蒸馏的方法在每个客户端的本地训练中保留了与空类别相关的重要信息。此外，逻辑抑制直接阻断了预测结果中对空类别的输出。

    Data heterogeneity, characterized by disparities in local data distribution across clients, poses a significant challenge in federated learning. Substantial efforts have been devoted to addressing the heterogeneity in local label distribution. As minority classes suffer from worse accuracy due to overfitting on local imbalanced data, prior methods often incorporate class-balanced learning techniques during local training. Despite the improved mean accuracy across all classes, we observe that empty classes-referring to categories absent from a client's data distribution-are still not well recognized. This paper introduces FedED, a novel approach in heterogeneous federated learning that integrates both empty-class distillation and logit suppression simultaneously. Specifically, empty-class distillation leverages knowledge distillation during local training on each client to retain essential information related to empty classes from the global model. Moreover, logit suppression directly p
    
[^13]: 关于负责任的机器学习数据集和公平性、隐私和法规准则的论文

    On Responsible Machine Learning Datasets with Fairness, Privacy, and Regulatory Norms. (arXiv:2310.15848v1 [cs.LG])

    [http://arxiv.org/abs/2310.15848](http://arxiv.org/abs/2310.15848)

    这篇论文讨论了负责任的机器学习数据集的重要性，并提出了一个通过负责任评价标准来评估数据集的框架。

    

    人工智能已经进入各个科学领域，在各种任务上比现有算法有了惊人的改进。近年来，人们对人工智能技术的可信性存在严重担忧。科学界致力于开发可信的人工智能算法。然而，目前在人工智能社区中流行的机器学习和深度学习算法在其开发过程中严重依赖使用的数据。这些学习算法通过识别数据中的模式来学习行为目标。数据中的任何缺陷都有可能直接转化为算法的缺陷。在这项研究中，我们讨论了负责任的机器学习数据集的重要性，并提出了一个通过负责任评价标准来评估数据集的框架。现有的工作侧重于对算法的后期评估以确保其可信性，而我们提供了一个框架，单独考虑数据组件以理解其特性。

    Artificial Intelligence (AI) has made its way into various scientific fields, providing astonishing improvements over existing algorithms for a wide variety of tasks. In recent years, there have been severe concerns over the trustworthiness of AI technologies. The scientific community has focused on the development of trustworthy AI algorithms. However, machine and deep learning algorithms, popular in the AI community today, depend heavily on the data used during their development. These learning algorithms identify patterns in the data, learning the behavioral objective. Any flaws in the data have the potential to translate directly into algorithms. In this study, we discuss the importance of Responsible Machine Learning Datasets and propose a framework to evaluate the datasets through a responsible rubric. While existing work focuses on the post-hoc evaluation of algorithms for their trustworthiness, we provide a framework that considers the data component separately to understand it
    
[^14]: CRITERIA：一种评估自动驾驶轨迹预测模型的新基准方法

    CRITERIA: a New Benchmarking Paradigm for Evaluating Trajectory Prediction Models for Autonomous Driving. (arXiv:2310.07794v1 [cs.CV])

    [http://arxiv.org/abs/2310.07794](http://arxiv.org/abs/2310.07794)

    CRITERIA是一种新的基准测试方法，用于评估自动驾驶轨迹预测模型。它通过精细排名预测，提供了关于模型性能在不同情况下的洞察。

    

    基准测试是评估自动驾驶轨迹预测模型常用的方法。现有的基准测试依赖于数据集，这些数据集对于较常见的情况（如巡航）存在偏差，并通过对所有情况进行平均计算的基于距离的度量。这种方法很少能提供有关模型性能的洞察，无论是在不同情况下它们能否良好处理，还是它们的输出是否允许和多样化。虽然存在一些用于衡量轨迹可允许性和多样性的补充指标，但它们受到偏见的影响，如轨迹长度。在本文中，我们提出了一种新的基准测试方法（CRITERIA），用于评估轨迹预测方法。特别地，我们提出了一种根据道路结构、模型性能和数据特性提取驾驶场景的方法，以进行精细排名预测。

    Benchmarking is a common method for evaluating trajectory prediction models for autonomous driving. Existing benchmarks rely on datasets, which are biased towards more common scenarios, such as cruising, and distance-based metrics that are computed by averaging over all scenarios. Following such a regiment provides a little insight into the properties of the models both in terms of how well they can handle different scenarios and how admissible and diverse their outputs are. There exist a number of complementary metrics designed to measure the admissibility and diversity of trajectories, however, they suffer from biases, such as length of trajectories.  In this paper, we propose a new benChmarking paRadIgm for evaluaTing trajEctoRy predIction Approaches (CRITERIA). Particularly, we propose 1) a method for extracting driving scenarios at varying levels of specificity according to the structure of the roads, models' performance, and data properties for fine-grained ranking of prediction 
    
[^15]: 区域隐形补丁：基于生成对抗网络的物理对抗攻击物体检测器

    Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors. (arXiv:2303.04238v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.04238](http://arxiv.org/abs/2303.04238)

    本文提出了一种基于GAN的无梯度物理对抗攻击方法，用于生成自然的对抗补丁，攻击物体检测器，具有实际应用价值。

    

    近年来，深度学习模型的对抗攻击越来越引起关注。这一领域的研究大多集中在基于梯度的技术，即所谓的白盒攻击，在其中攻击者可以访问目标模型的内部参数。然而，这种假设在实际世界中通常是不现实的。相对地，我们提出了一种在无需使用梯度的情况下，利用预先训练的生成对抗网络（GAN）的学习图像流形来生成自然的物理对抗补丁，用于物体检测器的攻击方法。我们展示了我们提出的方法在数字和物理层面上均可行。

    Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called white-box attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. We show that our proposed method works both digitally and physically.
    
[^16]: RGMIM: 区域引导的掩膜图像建模用于COVID-19检测。

    RGMIM: Region-Guided Masked Image Modeling for COVID-19 Detection. (arXiv:2211.00313v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.00313](http://arxiv.org/abs/2211.00313)

    本论文提出了一种针对COVID-19检测的新颖区域引导的掩膜图像建模方法，该方法通过利用肺掩模信息来识别有效区域，以学习更有用的COVID-19检测信息。

    

    目的：自监督学习正在快速推进医学领域的计算机辅助诊断。掩膜图像建模（MIM）是一种自监督学习方法，它掩盖了一组输入像素并试图预测遮盖的像素。传统的MIM方法通常采用随机掩膜策略。与普通图像相比，医学图像往往具有用于疾病检测的小区域。因此，我们在本文中专注于解决这个问题，在自动COVID-19识别方面进行评估。方法：本文提出了一种新颖的区域引导的掩膜图像建模方法（RGMIM）用于COVID-19检测。在我们的方法中，我们设计了一种新的掩膜策略，利用肺掩模信息来识别有效区域，以学习更有用的COVID-19检测信息。我们将所提出的方法与五种自监督学习技术（MAE，SKD，Cross，BYOL和SimSiam）进行对比。我们提出了定量评估。

    Purpose: Self-supervised learning is rapidly advancing computer-aided diagnosis in the medical field. Masked image modeling (MIM) is one of the self-supervised learning methods that masks a subset of input pixels and attempts to predict the masked pixels. Traditional MIM methods often employ a random masking strategy. In comparison to ordinary images, medical images often have a small region of interest for disease detection. Consequently, we focus on fixing the problem in this work, which is evaluated by automatic COVID-19 identification. Methods: In this study, we propose a novel region-guided masked image modeling method (RGMIM) for COVID-19 detection in this paper. In our method, we devise a new masking strategy that employed lung mask information to identify valid regions to learn more useful information for COVID-19 detection. The proposed method was contrasted with five self-supervised learning techniques (MAE, SKD, Cross, BYOL, and, SimSiam). We present a quantitative evaluatio
    

