# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AIC-UNet: Anatomy-informed Cascaded UNet for Robust Multi-Organ Segmentation](https://arxiv.org/abs/2403.18878) | 引入了一种新方法，在任何现有编码-解码分割模型上，通过条件化模型预测与可学习的解剖先验来施加解剖约束。 |
| [^2] | [Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures](https://arxiv.org/abs/2403.14772) | 通过稀疏编码层设计新网络架构以提高对模型逆推攻击的鲁棒性。 |
| [^3] | [EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks](https://arxiv.org/abs/2403.12574) | 提出了一种利用循环脉冲神经网络的自适应采样模块，通过将脉冲神经元的神经动力学与理想的时间事件采样器的行为相结合，实现了端到端可学习的事件检测框架 |
| [^4] | [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029) | 引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。 |
| [^5] | [PALM: Pushing Adaptive Learning Rate Mechanisms for Continual Test-Time Adaptation](https://arxiv.org/abs/2403.10650) | 本研究通过对模型预测不确定性的量化来选择需要进一步适应的层，从而克服了持续测试时间自适应方法中由于伪标签引起的不准确性困扰。 |
| [^6] | [AFBT GAN: enhanced explainability and diagnostic performance for cognitive decline by counterfactual generative adversarial network](https://arxiv.org/abs/2403.01758) | 利用反事实推理构建的 AFBT GAN 增强了对认知衰退的可解释性和诊断性能 |
| [^7] | [LLMs Meet Long Video: Advancing Long Video Comprehension with An Interactive Visual Adapter in LLMs](https://arxiv.org/abs/2402.13546) | 介绍了一个交互式视觉适配器（IVA），用于在LLMs中增强对细粒度视觉元素的交互，并解决了长视频理解中的计算成本高、视觉清晰度降低和无关视觉令牌带来的挑战。 |
| [^8] | [Be Persistent: Towards a Unified Solution for Mitigating Shortcuts in Deep Learning](https://arxiv.org/abs/2402.11237) | 本文旨在通过利用拓扑数据分析提出一个统一的解决方案，检测深度学习中的快捷学习问题。 |
| [^9] | [Continual Adversarial Defense](https://arxiv.org/abs/2312.09481) | 提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。 |
| [^10] | [Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models](https://arxiv.org/abs/2311.06607) | Monkey通过提高图像分辨率和采用多级描述生成方法来增强大型多模态模型(LMMs)的能力，从而实现更详细的视觉捕捉和更有效的学习。 |
| [^11] | [Generalized Categories Discovery for Long-tailed Recognition.](http://arxiv.org/abs/2401.05352) | 长尾识别中的通用类别发现方法(GCD)的重大限制是假设未标记数据中的类别分布是均衡的，而事实上自然环境中的视觉类别通常呈现长尾分布。本文提出了一种针对长尾通用类别发现（Long-tailed GCD）的方法，通过两个策略性正则化实现了对较少出现的尾部类别的重要性的增强。 |
| [^12] | [Bringing Back the Context: Camera Trap Species Identification as Link Prediction on Multimodal Knowledge Graphs.](http://arxiv.org/abs/2401.00608) | 本研究利用相机陷阱图像的结构化上下文，提高其在物种识别任务中的泛化能力，并解决了数据稀缺和泛化能力增强的问题。 |
| [^13] | [Text2Seg: Remote Sensing Image Semantic Segmentation via Text-Guided Visual Foundation Models.](http://arxiv.org/abs/2304.10597) | 本文介绍了一种名为 Text2Seg 的遥感图像语义分割流程，利用多个基础模型和文本引导，取得了初步成果。 |

# 详细

[^1]: AIC-UNet: 用于健壮多器官分割的解剖信息驱动级联UNet

    AIC-UNet: Anatomy-informed Cascaded UNet for Robust Multi-Organ Segmentation

    [https://arxiv.org/abs/2403.18878](https://arxiv.org/abs/2403.18878)

    引入了一种新方法，在任何现有编码-解码分割模型上，通过条件化模型预测与可学习的解剖先验来施加解剖约束。

    

    强加关键解剖特征，例如器官数量、形状、大小和相对位置，对于构建健壮的多器官分割模型至关重要。我们通过在现有的编码-解码分割模型上引入可学习的解剖先验，来实施解剖约束的新方法。具体来说，给定腹部扫描时，编码器的一部分通过薄板样条（TPS）网格插值将可学习的先验空间对准给定的输入扫描。然后在解码阶段整合变形的先验以指导模型。

    arXiv:2403.18878v1 Announce Type: cross  Abstract: Imposing key anatomical features, such as the number of organs, their shapes, sizes, and relative positions, is crucial for building a robust multi-organ segmentation model. Current attempts to incorporate anatomical features include broadening effective receptive fields (ERF) size with resource- and data-intensive modules such as self-attention or introducing organ-specific topology regularizers, which may not scale to multi-organ segmentation problems where inter-organ relation also plays a huge role. We introduce a new approach to impose anatomical constraints on any existing encoder-decoder segmentation model by conditioning model prediction with learnable anatomy prior. More specifically, given an abdominal scan, a part of the encoder spatially warps a learnable prior to align with the given input scan using thin plate spline (TPS) grid interpolation. The warped prior is then integrated during the decoding phase to guide the model
    
[^2]: 通过稀疏编码架构提高模型逆推攻击的鲁棒性

    Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures

    [https://arxiv.org/abs/2403.14772](https://arxiv.org/abs/2403.14772)

    通过稀疏编码层设计新网络架构以提高对模型逆推攻击的鲁棒性。

    

    最近的模型逆推攻击算法允许对手通过反复查询神经网络并检查其输出来重建网络的私有训练数据。 在这项工作中，我们开发了一种新颖的网络架构，利用稀疏编码层来获得对这类攻击的卓越鲁棒性。 三十年来，计算机科学研究已经研究了稀疏编码在图像去噪，目标识别和对抗性误分设置中的作用，但据我们所知，其与最先进的隐私漏洞之间的联系尚未被研究。然而，稀疏编码架构提供了一种有利的手段来抵御模型逆推攻击，因为它们允许我们控制编码在网络的中间表示中的无关私人信息的数量，而这种方式可以在训练过程中高效计算，并且众所周知只有较小的影响。

    arXiv:2403.14772v1 Announce Type: cross  Abstract: Recent model inversion attack algorithms permit adversaries to reconstruct a neural network's private training data just by repeatedly querying the network and inspecting its outputs. In this work, we develop a novel network architecture that leverages sparse-coding layers to obtain superior robustness to this class of attacks. Three decades of computer science research has studied sparse coding in the context of image denoising, object recognition, and adversarial misclassification settings, but to the best of our knowledge, its connection to state-of-the-art privacy vulnerabilities remains unstudied. However, sparse coding architectures suggest an advantageous means to defend against model inversion attacks because they allow us to control the amount of irrelevant private information encoded in a network's intermediate representations in a manner that can be computed efficiently during training and that is known to have little effect
    
[^3]: EAS-SNN：端到端自适应采样和表示，用于循环脉冲神经网络的事件检测

    EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks

    [https://arxiv.org/abs/2403.12574](https://arxiv.org/abs/2403.12574)

    提出了一种利用循环脉冲神经网络的自适应采样模块，通过将脉冲神经元的神经动力学与理想的时间事件采样器的行为相结合，实现了端到端可学习的事件检测框架

    

    事件摄像头以其高动态范围和时间分辨率，特别适用于物体检测，尤其是在存在动态模糊和具有挑战性的光照条件的情况下。然而，大多数现有方法更注重优化具有先进检测骨干和早期聚合功能的时空表示，而自适应事件采样的关键问题仍未得到解决。脉冲神经网络（SNN），通过稀疏脉冲通信运行的事件驱动范式，成为解决这一挑战的天然选择。在这项研究中，我们发现脉冲神经元的神经动力学与理想的时间事件采样器的行为密切相符。在这一启发下，我们提出了一个新颖的自适应采样模块，利用具有时间记忆的循环卷积SNN增强，为基于事件检测的完全端到端可学习框架提供支持。

    arXiv:2403.12574v1 Announce Type: cross  Abstract: Event cameras, with their high dynamic range and temporal resolution, are ideally suited for object detection, especially under scenarios with motion blur and challenging lighting conditions. However, while most existing approaches prioritize optimizing spatiotemporal representations with advanced detection backbones and early aggregation functions, the crucial issue of adaptive event sampling remains largely unaddressed. Spiking Neural Networks (SNNs), which operate on an event-driven paradigm through sparse spike communication, emerge as a natural fit for addressing this challenge. In this study, we discover that the neural dynamics of spiking neurons align closely with the behavior of an ideal temporal event sampler. Motivated by this insight, we propose a novel adaptive sampling module that leverages recurrent convolutional SNNs enhanced with temporal memory, facilitating a fully end-to-end learnable framework for event-based detec
    
[^4]: 对齐与提炼：统一和改进领域自适应目标检测

    Align and Distill: Unifying and Improving Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.12029](https://arxiv.org/abs/2403.12029)

    引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。

    

    目标检测器通常表现不佳于与其训练集不同的数据。最近，领域自适应目标检测（DAOD）方法已经展示了在应对这一挑战上的强大结果。遗憾的是，我们发现了系统化的基准测试陷阱，这些陷阱对过去的结果提出质疑并阻碍了进一步的进展：（a）由于基线不足导致性能高估，（b）不一致的实现实践阻止了方法的透明比较，（c）由于过时的骨干和基准测试缺乏多样性，导致缺乏普遍性。我们通过引入以下问题来解决这些问题：（1）一个统一的基准测试和实现框架，Align and Distill（ALDI），支持DAOD方法的比较并支持未来发展，（2）一个公平且现代的DAOD训练和评估协议，解决了基准测试的陷阱，（3）一个新的DAOD基准数据集，CFC-DAOD，能够在多样化的真实环境中进行评估。

    arXiv:2403.12029v1 Announce Type: cross  Abstract: Object detectors often perform poorly on data that differs from their training set. Domain adaptive object detection (DAOD) methods have recently demonstrated strong results on addressing this challenge. Unfortunately, we identify systemic benchmarking pitfalls that call past results into question and hamper further progress: (a) Overestimation of performance due to underpowered baselines, (b) Inconsistent implementation practices preventing transparent comparisons of methods, and (c) Lack of generality due to outdated backbones and lack of diversity in benchmarks. We address these problems by introducing: (1) A unified benchmarking and implementation framework, Align and Distill (ALDI), enabling comparison of DAOD methods and supporting future development, (2) A fair and modern training and evaluation protocol for DAOD that addresses benchmarking pitfalls, (3) A new DAOD benchmark dataset, CFC-DAOD, enabling evaluation on diverse real
    
[^5]: PALM：推进用于持续测试时间自适应的自适应学习率机制

    PALM: Pushing Adaptive Learning Rate Mechanisms for Continual Test-Time Adaptation

    [https://arxiv.org/abs/2403.10650](https://arxiv.org/abs/2403.10650)

    本研究通过对模型预测不确定性的量化来选择需要进一步适应的层，从而克服了持续测试时间自适应方法中由于伪标签引起的不准确性困扰。

    

    实际环境中的视觉模型面临领域分布的快速转变，导致识别性能下降。持续测试时间自适应（CTTA）直接根据测试数据调整预训练的源判别模型以适应这些不断变化的领域。一种高度有效的CTTA方法涉及应用逐层自适应学习率，并选择性地调整预训练层。然而，它受到领域转移估计不准确和由伪标签引起的不准确性所困扰。在这项工作中，我们旨在通过识别层来克服这些限制，通过对模型预测不确定性的量化来选择层，而无须依赖伪标签。我们利用梯度的大小作为一个度量标准，通过反向传播softmax输出与均匀分布之间的KL散度来计算，以选择需要进一步适应的层。随后，仅属于这些层的参数将被进一步适应。

    arXiv:2403.10650v1 Announce Type: cross  Abstract: Real-world vision models in dynamic environments face rapid shifts in domain distributions, leading to decreased recognition performance. Continual test-time adaptation (CTTA) directly adjusts a pre-trained source discriminative model to these changing domains using test data. A highly effective CTTA method involves applying layer-wise adaptive learning rates, and selectively adapting pre-trained layers. However, it suffers from the poor estimation of domain shift and the inaccuracies arising from the pseudo-labels. In this work, we aim to overcome these limitations by identifying layers through the quantification of model prediction uncertainty without relying on pseudo-labels. We utilize the magnitude of gradients as a metric, calculated by backpropagating the KL divergence between the softmax output and a uniform distribution, to select layers for further adaptation. Subsequently, for the parameters exclusively belonging to these se
    
[^6]: AFBT GAN: 通过反事实生成对抗网络增强对认知衰退的可解释性和诊断性能

    AFBT GAN: enhanced explainability and diagnostic performance for cognitive decline by counterfactual generative adversarial network

    [https://arxiv.org/abs/2403.01758](https://arxiv.org/abs/2403.01758)

    利用反事实推理构建的 AFBT GAN 增强了对认知衰退的可解释性和诊断性能

    

    现有的功能连接（FC）的解释结果通常是通过使用分类结果标签和诸如Pearson相关性或梯度反推等相关分析方法生成的。然而，诊断模型仍然是在黑盒模型上训练的，在训练过程中可能缺乏对重要区域FC的关注。为了增强可解释性和提高诊断性能，在诊断模型中提供关于神经退行性相关区域的先验知识，特别是当健康受试者（HC）发展为主观认知衰退（SCD）和轻度认知障碍（MCI）时，这是一个关键步骤。为了更好地确定神经退行性相关区域，我们采用反事实推理来生成源标签FC派生的目标标签FC矩阵，然后将源标签FC减去目标标签FC。自适应前向和后向转换构成了反事实推理架构。

    arXiv:2403.01758v1 Announce Type: cross  Abstract: Existing explanation results of functional connectivity (FC) are normally generated by using classification result labels and correlation analysis methods such as Pearson's correlation or gradient backward. However, the diagnostic model is still trained on the black box model and might lack the attention of FCs in important regions during the training. To enhance the explainability and improve diagnostic performance, providing prior knowledge on neurodegeneration-related regions when healthy subjects (HC) develop into subject cognitive decline (SCD) and mild cognitive impairment (MCI) for the diagnostic model is a key step. To better determine the neurodegeneration-related regions, we employ counterfactual reasoning to generate the target label FC matrices derived from source label FC and then subtract source label FC with target label FC. The counterfactual reasoning architecture is constructed by adaptive forward and backward transfo
    
[^7]: LLMs与长视频相遇：在LLMs中利用互动式视觉适配器推进长视频理解

    LLMs Meet Long Video: Advancing Long Video Comprehension with An Interactive Visual Adapter in LLMs

    [https://arxiv.org/abs/2402.13546](https://arxiv.org/abs/2402.13546)

    介绍了一个交互式视觉适配器（IVA），用于在LLMs中增强对细粒度视觉元素的交互，并解决了长视频理解中的计算成本高、视觉清晰度降低和无关视觉令牌带来的挑战。

    

    长视频理解是多媒体和人工智能交叉领域中一项重要且持续挑战。利用大型语言模型(LLMs)来理解视频成为一种新兴且有前景的方法。然而，由于视频令牌数量庞大，这种方法导致计算成本高，视觉清晰度降低，还面临着在回答视频相关问题时出现无关视觉令牌所带来的挑战。为了缓解这些问题，我们在LLMs中提出了一个交互式视觉适配器(IVA)，旨在增强与细粒度视觉元素的交互。具体来说，我们首先通过利用视觉编码器和预训练因果变换器将长视频转换为时间视频令牌，然后将它们与视频说明一起输入LLMs。随后，我们集成了IVA，其中包含一个轻量级的时间帧选择器

    arXiv:2402.13546v1 Announce Type: new  Abstract: Long video understanding is a significant and ongoing challenge in the intersection of multimedia and artificial intelligence. Employing large language models (LLMs) for comprehending video becomes an emerging and promising method. However, this approach incurs high computational costs due to the extensive array of video tokens, experiences reduced visual clarity as a consequence of token aggregation, and confronts challenges arising from irrelevant visual tokens while answering video-related questions. To alleviate these issues, we present an Interactive Visual Adapter (IVA) within LLMs, designed to enhance interaction with fine-grained visual elements. Specifically, we first transform long videos into temporal video tokens via leveraging a visual encoder alongside a pretrained causal transformer, then feed them into LLMs with the video instructions. Subsequently, we integrated IVA, which contains a lightweight temporal frame selector a
    
[^8]: 对抗深度学习中快捷方式的统一解决方案

    Be Persistent: Towards a Unified Solution for Mitigating Shortcuts in Deep Learning

    [https://arxiv.org/abs/2402.11237](https://arxiv.org/abs/2402.11237)

    本文旨在通过利用拓扑数据分析提出一个统一的解决方案，检测深度学习中的快捷学习问题。

    

    深度神经网络(DNNs)容易受到快捷学习的影响：它们倾向于建立输入和输出之间无关的关系，而不是学习预期的任务。快捷学习在神经网络许多失败案例中普遍存在，这一现象的痕迹可见于其泛化问题、领域转移、对抗性脆弱性，甚至对多数群体的偏见。本文认为，各种DNN问题的共同原因为我们提供了一个重要机会，应该利用这一点找到对抗快捷学习的统一解决方案。为此，我们概述了拓扑数据分析(TDA)特别是持续同调(PH)方面的最新进展，为探测深度学习中快捷方式勾画了统一的路线图。我们通过研究DNNs中计算图的拓扑特征，使用无法学习的示例和偏见为两种情况，来证明我们的论点。

    arXiv:2402.11237v1 Announce Type: new  Abstract: Deep neural networks (DNNs) are vulnerable to shortcut learning: rather than learning the intended task, they tend to draw inconclusive relationships between their inputs and outputs. Shortcut learning is ubiquitous among many failure cases of neural networks, and traces of this phenomenon can be seen in their generalizability issues, domain shift, adversarial vulnerability, and even bias towards majority groups. In this paper, we argue that this commonality in the cause of various DNN issues creates a significant opportunity that should be leveraged to find a unified solution for shortcut learning. To this end, we outline the recent advances in topological data analysis~(TDA), and persistent homology~(PH) in particular, to sketch a unified roadmap for detecting shortcuts in deep learning. We demonstrate our arguments by investigating the topological features of computational graphs in DNNs using two cases of unlearnable examples and bia
    
[^9]: 持续不断的对抗性防御

    Continual Adversarial Defense

    [https://arxiv.org/abs/2312.09481](https://arxiv.org/abs/2312.09481)

    提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。

    

    针对每月针对视觉分类器的对抗性攻击快速演变的特性，人们提出了许多防御方法，旨在尽可能通用化以抵御尽可能多的已知攻击。然而，设计一个能够对抗所有类型攻击的防御方法并不现实，因为防御系统运行的环境是动态的，包含随着时间出现的各种独特攻击。防御系统必须收集在线少样本对抗反馈以迅速增强自身，充分利用内存。因此，我们提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架，其中各种攻击逐个阶段出现。在实践中，CAD基于四项原则进行建模：(1) 持续适应新攻击而无灾难性遗忘，(2) 少样本适应，(3) 内存高效适应，以及(4) 高准确性

    arXiv:2312.09481v2 Announce Type: replace-cross  Abstract: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. The defense system must gather online few-shot defense feedback to promptly enhance itself, leveraging efficient memory utilization. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accur
    
[^10]: Monkey: 大型多模态模型中图像分辨率和文本标签的重要性

    Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models

    [https://arxiv.org/abs/2311.06607](https://arxiv.org/abs/2311.06607)

    Monkey通过提高图像分辨率和采用多级描述生成方法来增强大型多模态模型(LMMs)的能力，从而实现更详细的视觉捕捉和更有效的学习。

    

    大型多模态模型(LMMs)在视觉语言任务中表现出了潜力，但在高分辨率输入和详细场景理解方面表现不佳。为了解决这些挑战，我们引入了Monkey来增强LMM的能力。首先，Monkey通过将输入图像划分为统一的补丁来处理图像，每个补丁的大小与原来训练良好的视觉编码器使用的大小(例如448x448)相匹配。配备了每个补丁的适配器，Monkey可以处理高达1344x896像素的更高分辨率，实现对复杂视觉信息的详细捕捉。其次，它采用多级描述生成方法，丰富了场景-对象关联的上下文。这种两部分策略确保了从生成数据中更有效的学习：更高的分辨率允许对视觉进行更详细的捕捉，从而增强了全面描述的效果。广泛的实验证明...

    arXiv:2311.06607v3 Announce Type: replace-cross  Abstract: Large Multimodal Models (LMMs) have shown promise in vision-language tasks but struggle with high-resolution input and detailed scene understanding. Addressing these challenges, we introduce Monkey to enhance LMM capabilities. Firstly, Monkey processes input images by dividing them into uniform patches, each matching the size (e.g., 448x448) used in the original training of the well-trained vision encoder. Equipped with individual adapter for each patch, Monkey can handle higher resolutions up to 1344x896 pixels, enabling the detailed capture of complex visual information. Secondly, it employs a multi-level description generation method, enriching the context for scene-object associations. This two-part strategy ensures more effective learning from generated data: the higher resolution allows for a more detailed capture of visuals, which in turn enhances the effectiveness of comprehensive descriptions. Extensive ablative result
    
[^11]: 长尾识别中的通用类别发现

    Generalized Categories Discovery for Long-tailed Recognition. (arXiv:2401.05352v1 [cs.CV])

    [http://arxiv.org/abs/2401.05352](http://arxiv.org/abs/2401.05352)

    长尾识别中的通用类别发现方法(GCD)的重大限制是假设未标记数据中的类别分布是均衡的，而事实上自然环境中的视觉类别通常呈现长尾分布。本文提出了一种针对长尾通用类别发现（Long-tailed GCD）的方法，通过两个策略性正则化实现了对较少出现的尾部类别的重要性的增强。

    

    通用类别发现（GCD）在从未标记的数据集中识别已知和未知类别方面起着至关重要的作用，它利用了通过已标记类别集合获取的洞察力。现有的GCD方法的一个显著限制是它们假设未标记数据中的类别分布是均衡的。与这一假设相反，自然环境中的视觉类别通常表现出长尾分布，已知或普遍的类别比罕见的类别更频繁地出现。我们的研究致力于弥合这种差距，着重于长尾通用类别发现（Long-tailed GCD）范式，该范式反映了现实世界未标记数据集的固有不平衡性。针对长尾GCD所带来的独特挑战，我们提出了一种基于两个策略性正则化的强大方法:（i）一种加权机制，增强了较少出现的尾部类别的重要性。

    Generalized Class Discovery (GCD) plays a pivotal role in discerning both known and unknown categories from unlabeled datasets by harnessing the insights derived from a labeled set comprising recognized classes. A significant limitation in prevailing GCD methods is their presumption of an equitably distributed category occurrence in unlabeled data. Contrary to this assumption, visual classes in natural environments typically exhibit a long-tailed distribution, with known or prevalent categories surfacing more frequently than their rarer counterparts. Our research endeavors to bridge this disconnect by focusing on the long-tailed Generalized Category Discovery (Long-tailed GCD) paradigm, which echoes the innate imbalances of real-world unlabeled datasets. In response to the unique challenges posed by Long-tailed GCD, we present a robust methodology anchored in two strategic regularizations: (i) a reweighting mechanism that bolsters the prominence of less-represented, tail-end categories
    
[^12]: 将上下文带回来：多模态知识图谱上的相机陷阱物种识别作为链接预测

    Bringing Back the Context: Camera Trap Species Identification as Link Prediction on Multimodal Knowledge Graphs. (arXiv:2401.00608v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2401.00608](http://arxiv.org/abs/2401.00608)

    本研究利用相机陷阱图像的结构化上下文，提高其在物种识别任务中的泛化能力，并解决了数据稀缺和泛化能力增强的问题。

    

    相机陷阱在动物生态学中是宝贵的工具，用于生物多样性监测和保护。然而，挑战如在新的未知位置部署时的糟糕泛化限制了它们的实际应用。图像自然与可能在不同模态下的异质上下文相关联。在这项工作中，我们利用与相机陷阱图像相关联的结构化上下文，改善在相机陷阱中物种识别这个任务的超出分布的泛化能力。例如，一张野生动物的照片可能与拍摄地点和时间以及关于动物物种的结构化生物学知识相关联。虽然现有的工作通常忽视这一点，但将这样的上下文带回来可以带来一些潜在的好处，如解决数据稀缺和增强泛化能力。然而，有效地将这样的异质上下文整合到视觉领域是一个具有挑战性的问题。

    Camera traps are valuable tools in animal ecology for biodiversity monitoring and conservation. However, challenges like poor generalization to deployment at new unseen locations limit their practical application. Images are naturally associated with heterogeneous forms of context possibly in different modalities. In this work, we leverage the structured context associated with the camera trap images to improve out-of-distribution generalization for the task of species identification in camera traps. For example, a photo of a wild animal may be associated with information about where and when it was taken, as well as structured biology knowledge about the animal species. While typically overlooked by existing work, bringing back such context offers several potential benefits for better image understanding, such as addressing data scarcity and enhancing generalization. However, effectively integrating such heterogeneous context into the visual domain is a challenging problem. To address
    
[^13]: Text2Seg: 通过文本引导视觉基础模型的遥感图像语义分割

    Text2Seg: Remote Sensing Image Semantic Segmentation via Text-Guided Visual Foundation Models. (arXiv:2304.10597v1 [cs.CV])

    [http://arxiv.org/abs/2304.10597](http://arxiv.org/abs/2304.10597)

    本文介绍了一种名为 Text2Seg 的遥感图像语义分割流程，利用多个基础模型和文本引导，取得了初步成果。

    

    最近，基础模型（FMs），如 GPT-4 和 LLaMA，在零样本学习方案中表现出色，吸引了大量关注。类似地，在视觉学习领域，Grounding DINO 和 Segment Anything Model（SAM）等模型在开放式检测和实例分割任务中展现了显著的进步。本研究专注于遥感领域，其中图片与传统场景中的图片明显不同。我们开发了一个流程，利用多个 FMs，以文本提示为指导，促进遥感图像语义分割任务，我们将其称为 Text2Seg 。该管道在多个广泛使用的遥感数据集上进行基准测试，并提供初步结果以证明其有效性。

    Recent advancements in foundation models (FMs), such as GPT-4 and LLaMA, have attracted significant attention due to their exceptional performance in zero-shot learning scenarios. Similarly, in the field of visual learning, models like Grounding DINO and the Segment Anything Model (SAM) have exhibited remarkable progress in open-set detection and instance segmentation tasks. It is undeniable that these FMs will profoundly impact a wide range of real-world visual learning tasks, ushering in a new paradigm shift for developing such models. In this study, we concentrate on the remote sensing domain, where the images are notably dissimilar from those in conventional scenarios. We developed a pipeline that leverages multiple FMs to facilitate remote sensing image semantic segmentation tasks guided by text prompt, which we denote as Text2Seg. The pipeline is benchmarked on several widely-used remote sensing datasets, and we present preliminary results to demonstrate its effectiveness. Throug
    

