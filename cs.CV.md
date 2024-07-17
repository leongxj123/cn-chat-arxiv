# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298) | RoPE在视觉变压器中展现出令人印象深刻的外推性能，提高了ImageNet-1k、COCO检测和ADE-20k分割的性能。 |
| [^2] | [Neuro-Symbolic Video Search](https://arxiv.org/abs/2403.11021) | 提出了一种神经网络符号视频搜索系统，该系统利用视觉-语言模型进行语义理解，并通过状态机和时间逻辑公式对事件的长期演变进行推理，从而实现高效的场景识别。 |
| [^3] | [Unsupervised Concept Discovery Mitigates Spurious Correlations](https://arxiv.org/abs/2402.13368) | 该论文介绍了一种无监督概念发现方法，通过发现共享的离散概念来减轻虚假相关性，而无需人工标记子组，有效提高模型的鲁棒性和消除偏见。 |
| [^4] | [Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector](https://arxiv.org/abs/2402.03094) | 本文提出了一种跨领域少样本目标检测器，通过增强的开集目标检测方法来解决跨领域数据差异带来的性能下降问题。 |
| [^5] | [LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model](https://arxiv.org/abs/2402.02544) | LHRS-Bot 是一个利用自愿地理信息(VGI)增强的大型多模态语言模型，旨在解决近期MLLM在遥感领域中未对多样的地理景观和物体进行充分考虑的问题。通过引入多层次视觉-语言对齐策略和课程学习方法，LHRS-Bot展现出对RS图像的深刻理解以及在RS领域内进行细致推理的能力。 |
| [^6] | [ConTextual: Evaluating Context-Sensitive Text-Rich Visual Reasoning in Large Multimodal Models.](http://arxiv.org/abs/2401.13311) | 本文介绍了一个新颖的基准ConTextual，用于评估能够进行上下文敏感的文本富有视觉推理的大型多模态模型。研究发现，目前最好的模型GPT-4V在抽象类别表现出色，但在整体性能上仍然落后于人类，存在改进的空间。 |
| [^7] | [A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting.](http://arxiv.org/abs/2401.10227) | 该论文提出了一种基于稳定扩散的潜在扩散方法，用于全景分割和遮罩修复，通过简化架构来避免复杂性，实现了生成模型解锁遮罩修复功能，具有应用于交互式分割的潜力。 |
| [^8] | [VMAF Re-implementation on PyTorch: Some Experimental Results.](http://arxiv.org/abs/2310.15578) | 这项研究重新在PyTorch上实现了VMAF，与标准实现进行比较，结果显示在VMAF单位上的差异小于$10^{-2}$。同时，研究了在使用VMAF作为目标函数时的梯度计算，并证明使用该函数进行训练不会导致梯度不良。 |
| [^9] | [Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding.](http://arxiv.org/abs/2309.04561) | 提出了一个稠密三维引用网络ConcreteNet，包含三个新模块，旨在改善具有相同语义类别干扰因素的重复实例的引用性能。 |
| [^10] | [MOCA: Self-supervised Representation Learning by Predicting Masked Online Codebook Assignments.](http://arxiv.org/abs/2307.09361) | MOCA是一种自监督学习方法，通过预测掩码式在线码本分配来实现表示学习。它同时具备良好的语境推理属性和对图像扰动的不变性，并在低样本设置和各种评估协议中取得了最新的最先进结果，训练速度比之前的方法快3倍以上。 |
| [^11] | [Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models.](http://arxiv.org/abs/2306.12941) | 本文提出了针对语义分割模型的解决方案，使得可以对其进行攻击并提供了更好的评估协议。同时，通过微调鲁棒的主干，可以有限的计算代价训练对抗性鲁棒的分割模型。 |
| [^12] | [CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion.](http://arxiv.org/abs/2303.11916) | CompoDiff 是一种多功能的组合图像检索模型，通过接受各种条件，具有潜在扩散的能力，并在 FashionIQ 上实现了新的零样本最新技术水平。其特征位于完整的 CLIP 嵌入空间中，可以直接用于所有利用 CLIP 空间的模型。 |

# 详细

[^1]: 视觉变压器的旋转位置嵌入

    Rotary Position Embedding for Vision Transformer

    [https://arxiv.org/abs/2403.13298](https://arxiv.org/abs/2403.13298)

    RoPE在视觉变压器中展现出令人印象深刻的外推性能，提高了ImageNet-1k、COCO检测和ADE-20k分割的性能。

    

    旋转位置嵌入（RoPE）在语言模型上表现出色，特别适用于Transformer的长度外推。然而，RoPE对计算机视觉领域的影响尚未被充分探讨，尽管RoPE似乎能够像语言领域一样增强视觉变压器（ViT）的性能。本研究对将RoPE应用于ViT时进行了全面分析，利用RoPE在2D视觉数据上的实际实现。分析显示，RoPE展示出令人印象深刻的外推性能，即在推断时在增加图像分辨率的同时保持精度。最终导致了ImageNet-1k、COCO检测和ADE-20k分割的性能提升。我们相信本研究提供了将RoPE应用于ViT的详尽指导，承诺通过最小的额外计算开销提高骨干性能。我们的代码和预训练模型可在网址https://找到。

    arXiv:2403.13298v1 Announce Type: cross  Abstract: Rotary Position Embedding (RoPE) performs remarkably on language models, especially for length extrapolation of Transformers. However, the impacts of RoPE on computer vision domains have been underexplored, even though RoPE appears capable of enhancing Vision Transformer (ViT) performance in a way similar to the language domain. This study provides a comprehensive analysis of RoPE when applied to ViTs, utilizing practical implementations of RoPE for 2D vision data. The analysis reveals that RoPE demonstrates impressive extrapolation performance, i.e., maintaining precision while increasing image resolution at inference. It eventually leads to performance improvement for ImageNet-1k, COCO detection, and ADE-20k segmentation. We believe this study provides thorough guidelines to apply RoPE into ViT, promising improved backbone performance with minimal extra computational overhead. Our code and pre-trained models are available at https://
    
[^2]: 神经符号视频搜索

    Neuro-Symbolic Video Search

    [https://arxiv.org/abs/2403.11021](https://arxiv.org/abs/2403.11021)

    提出了一种神经网络符号视频搜索系统，该系统利用视觉-语言模型进行语义理解，并通过状态机和时间逻辑公式对事件的长期演变进行推理，从而实现高效的场景识别。

    

    近年来视频数据生产的空前激增需求高效的工具，以从视频中提取有意义的帧供下游任务使用。 长期时间推理是帧检索系统的一个关键要求。 虽然 VideoLLaMA 和 ViCLIP 等最先进的基础模型在短期语义理解方面表现优异，但它们在跨帧的长期推理方面却令人惊讶地失败。 这种失败的一个关键原因是它们将逐帧感知和时间推理交织成单个深度网络。 因此，解耦但共同设计语义理解和时间推理对于高效的场景识别是至关重要的。 我们提出了一种系统，利用视觉-语言模型对单个帧进行语义理解，但有效地通过使用状态机和时间逻辑（TL）公式对事件的长期演变进行推理，这些公式在本质上捕捉了记忆。

    arXiv:2403.11021v1 Announce Type: cross  Abstract: The unprecedented surge in video data production in recent years necessitates efficient tools to extract meaningful frames from videos for downstream tasks. Long-term temporal reasoning is a key desideratum for frame retrieval systems. While state-of-the-art foundation models, like VideoLLaMA and ViCLIP, are proficient in short-term semantic understanding, they surprisingly fail at long-term reasoning across frames. A key reason for this failure is that they intertwine per-frame perception and temporal reasoning into a single deep network. Hence, decoupling but co-designing semantic understanding and temporal reasoning is essential for efficient scene identification. We propose a system that leverages vision-language models for semantic understanding of individual frames but effectively reasons about the long-term evolution of events using state machines and temporal logic (TL) formulae that inherently capture memory. Our TL-based reas
    
[^3]: 无监督概念发现减轻虚假相关性

    Unsupervised Concept Discovery Mitigates Spurious Correlations

    [https://arxiv.org/abs/2402.13368](https://arxiv.org/abs/2402.13368)

    该论文介绍了一种无监督概念发现方法，通过发现共享的离散概念来减轻虚假相关性，而无需人工标记子组，有效提高模型的鲁棒性和消除偏见。

    

    在训练数据中容易产生虚假相关性的模型通常会产生脆弱的预测并引入意外的偏见。解决这一挑战通常涉及依赖先验知识和群组注释的方法，以消除虚假相关性，而这些信息在许多应用程序中可能并不容易获得。在本文中，我们建立了无监督物体中心学习与减轻虚假相关性之间的一种新联系。我们的方法不是直接推断与标签具有不同相关性的子组，而是专注于发现概念：在输入样本之间共享的离散思想。借助现有的物体中心表示学习，我们引入了CoBalT：一种概念平衡技术，有效减轻虚假相关性，而无需人类对子组进行标记。在水鸟、CelebA和ImageNet-9基准数据集上针对子群体变化的评估表明了其优越性。

    arXiv:2402.13368v1 Announce Type: new  Abstract: Models prone to spurious correlations in training data often produce brittle predictions and introduce unintended biases. Addressing this challenge typically involves methods relying on prior knowledge and group annotation to remove spurious correlations, which may not be readily available in many applications. In this paper, we establish a novel connection between unsupervised object-centric learning and mitigation of spurious correlations. Instead of directly inferring sub-groups with varying correlations with labels, our approach focuses on discovering concepts: discrete ideas that are shared across input samples. Leveraging existing object-centric representation learning, we introduce CoBalT: a concept balancing technique that effectively mitigates spurious correlations without requiring human labeling of subgroups. Evaluation across the Waterbirds, CelebA and ImageNet-9 benchmark datasets for subpopulation shifts demonstrate superio
    
[^4]: 跨领域少样本目标检测通过增强的开集目标检测器

    Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector

    [https://arxiv.org/abs/2402.03094](https://arxiv.org/abs/2402.03094)

    本文提出了一种跨领域少样本目标检测器，通过增强的开集目标检测方法来解决跨领域数据差异带来的性能下降问题。

    

    本文解决了跨领域少样本目标检测（CD-FSOD）的挑战，旨在开发一个准确的目标检测器，用最少的标记样本来检测新领域的目标。虽然基于转换器的开集检测器（例如DE-ViT）在开放词汇目标检测和传统的少样本目标检测方面表现出色，能够检测到训练过程中没有见过的类别，我们自然会提出两个关键问题：1）这种开集检测方法能否容易地推广到CD-FSOD？2）如果不能，如何在面对显著的领域差异时增强开集方法的结果？为了回答第一个问题，我们引入了几个衡量领域差异的指标，并建立了一个具有多样领域度量值的新的CD-FSOD基准。在这个基准上评估了一些最先进的开集目标检测方法，在域外数据集中观察到明显的性能下降。这表明采用这些方法在CD-FSOD上失败了。

    This paper addresses the challenge of cross-domain few-shot object detection (CD-FSOD), aiming to develop an accurate object detector for novel domains with minimal labeled examples. While transformer-based open-set detectors e.g., DE-ViT~\cite{zhang2023detect} have excelled in both open-vocabulary object detection and traditional few-shot object detection, detecting categories beyond those seen during training, we thus naturally raise two key questions: 1) can such open-set detection methods easily generalize to CD-FSOD? 2) If no, how to enhance the results of open-set methods when faced with significant domain gaps? To address the first question, we introduce several metrics to quantify domain variances and establish a new CD-FSOD benchmark with diverse domain metric values. Some State-Of-The-Art (SOTA) open-set object detection methods are evaluated on this benchmark, with evident performance degradation observed across out-of-domain datasets. This indicates the failure of adopting 
    
[^5]: LHRS-Bot：利用VGI增强的大型多模态语言模型赋能遥感领域

    LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model

    [https://arxiv.org/abs/2402.02544](https://arxiv.org/abs/2402.02544)

    LHRS-Bot 是一个利用自愿地理信息(VGI)增强的大型多模态语言模型，旨在解决近期MLLM在遥感领域中未对多样的地理景观和物体进行充分考虑的问题。通过引入多层次视觉-语言对齐策略和课程学习方法，LHRS-Bot展现出对RS图像的深刻理解以及在RS领域内进行细致推理的能力。

    

    大型语言模型（LLMs）的革命性能力开创了多模态大型语言模型（MLLMs）并促进了在各个专业领域的多样化应用。然而，在遥感（RS）领域中，近期的MLLM努力未能充分考虑到遥感图像中多样的地理景观和物体。为了弥补这一差距，我们构建了一个大规模的RS图像-文本数据集LHRS-Align，以及一个信息丰富的RS特定指导数据集LHRS-Instruct，利用丰富的自愿地理信息（VGI）和全球可用的RS图像。在此基础上，我们引入了LHRS-Bot，一种针对RS图像理解的MLLM，通过一种新颖的多层次视觉-语言对齐策略和课程学习方法。全面的实验证明，LHRS-Bot展现出对RS图像的深刻理解以及在RS领域内进行细致推理的能力。

    The revolutionary capabilities of large language models (LLMs) have paved the way for multimodal large language models (MLLMs) and fostered diverse applications across various specialized domains. In the remote sensing (RS) field, however, the diverse geographical landscapes and varied objects in RS imagery are not adequately considered in recent MLLM endeavors. To bridge this gap, we construct a large-scale RS image-text dataset, LHRS-Align, and an informative RS-specific instruction dataset, LHRS-Instruct, leveraging the extensive volunteered geographic information (VGI) and globally available RS images. Building on this foundation, we introduce LHRS-Bot, an MLLM tailored for RS image understanding through a novel multi-level vision-language alignment strategy and a curriculum learning method. Comprehensive experiments demonstrate that LHRS-Bot exhibits a profound understanding of RS images and the ability to perform nuanced reasoning within the RS domain.
    
[^6]: ConTextual: 在大型多模态模型中评估上下文敏感的文本富有视觉推理

    ConTextual: Evaluating Context-Sensitive Text-Rich Visual Reasoning in Large Multimodal Models. (arXiv:2401.13311v1 [cs.CV])

    [http://arxiv.org/abs/2401.13311](http://arxiv.org/abs/2401.13311)

    本文介绍了一个新颖的基准ConTextual，用于评估能够进行上下文敏感的文本富有视觉推理的大型多模态模型。研究发现，目前最好的模型GPT-4V在抽象类别表现出色，但在整体性能上仍然落后于人类，存在改进的空间。

    

    最近人工智能的进步导致了大型多模态模型（LMMs）的发展，这些模型能够处理涉及文本和图像内容的复杂任务，例如在公共场所导航地图。本文介绍了ConTextual，这是一个新颖的基准，包括专门设计的指令，用于评估LMMs在执行上下文敏感的文本富有视觉推理方面的能力。ConTextual强调了多样的现实世界场景（例如时间阅读、导航、购物等），要求更深入地理解文本和视觉元素之间的相互作用。我们的研究结果显示，最佳表现的LMM，GPT-4V(ision)，与人类能力之间存在30.8%的性能差距，使用人类评估指出在上下文敏感的文本富有视觉推理方面还有很大的改进空间。值得注意的是，虽然GPT-4V在抽象类别（如模因和引文解释）中表现出色，但其整体性能仍然落后于人类。

    Recent advancements in AI have led to the development of large multimodal models (LMMs) capable of processing complex tasks involving joint reasoning over text and visual content in the image (e.g., navigating maps in public places). This paper introduces ConTextual, a novel benchmark comprising instructions designed explicitly to evaluate LMMs' ability to perform context-sensitive text-rich visual reasoning. ConTextual emphasizes diverse real-world scenarios (e.g., time-reading, navigation, shopping and more) demanding a deeper understanding of the interactions between textual and visual elements. Our findings reveal a significant performance gap of 30.8% between the best-performing LMM, GPT-4V(ision), and human capabilities using human evaluation indicating substantial room for improvement in context-sensitive text-rich visual reasoning. Notably, while GPT-4V excelled in abstract categories like meme and quote interpretation, its overall performance still lagged behind humans. In add
    
[^7]: 一个简单的潜在扩散方法应用于全景分割和遮罩修复

    A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting. (arXiv:2401.10227v1 [cs.CV])

    [http://arxiv.org/abs/2401.10227](http://arxiv.org/abs/2401.10227)

    该论文提出了一种基于稳定扩散的潜在扩散方法，用于全景分割和遮罩修复，通过简化架构来避免复杂性，实现了生成模型解锁遮罩修复功能，具有应用于交互式分割的潜力。

    

    全景和实例分割网络通常通过专门的目标检测模块，复杂的损失函数和特殊的后处理步骤来训练，以处理实例遮罩的置换不变性。

    Panoptic and instance segmentation networks are often trained with specialized object detection modules, complex loss functions, and ad-hoc post-processing steps to handle the permutation-invariance of the instance masks. This work builds upon Stable Diffusion and proposes a latent diffusion approach for panoptic segmentation, resulting in a simple architecture which omits these complexities. Our training process consists of two steps: (1) training a shallow autoencoder to project the segmentation masks to latent space; (2) training a diffusion model to allow image-conditioned sampling in latent space. The use of a generative model unlocks the exploration of mask completion or inpainting, which has applications in interactive segmentation. The experimental validation yields promising results for both panoptic segmentation and mask inpainting. While not setting a new state-of-the-art, our model's simplicity, generality, and mask completion capability are desirable properties.
    
[^8]: 在PyTorch上重新实现的VMAF：一些实验结果

    VMAF Re-implementation on PyTorch: Some Experimental Results. (arXiv:2310.15578v1 [cs.LG])

    [http://arxiv.org/abs/2310.15578](http://arxiv.org/abs/2310.15578)

    这项研究重新在PyTorch上实现了VMAF，与标准实现进行比较，结果显示在VMAF单位上的差异小于$10^{-2}$。同时，研究了在使用VMAF作为目标函数时的梯度计算，并证明使用该函数进行训练不会导致梯度不良。

    

    基于标准的VMAF实现，我们提出了使用PyTorch框架实现VMAF的方法。对于这个实现，与标准的(libvmaf)进行比较，VMAF单位上的差异小于$10^{-2}$。我们研究了在使用VMAF作为目标函数时的梯度计算，并证明使用该函数进行训练不会导致梯度不良。

    Based on the standard VMAF implementation we propose an implementation of VMAF using PyTorch framework. For this implementation comparisons with the standard (libvmaf) show the discrepancy $\lesssim 10^{-2}$ in VMAF units. We investigate gradients computation when using VMAF as an objective function and demonstrate that training using this function does not result in ill-behaving gradients.
    
[^9]: 改进稠密三维视觉引用的三种方法

    Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding. (arXiv:2309.04561v1 [cs.CV])

    [http://arxiv.org/abs/2309.04561](http://arxiv.org/abs/2309.04561)

    提出了一个稠密三维引用网络ConcreteNet，包含三个新模块，旨在改善具有相同语义类别干扰因素的重复实例的引用性能。

    

    三维视觉引用是指通过自然语言描述来定位三维场景中被引用的物体的任务。该任务在自主室内机器人到AR/VR等各种应用中广泛应用。目前一种常见的解决方案是通过检测来完成三维视觉引用，即通过边界框来定位。然而，在需要进行物理交互的实际应用中，边界框不足以描述物体的几何属性。因此，我们解决了稠密三维视觉引用的问题，即基于引用的三维实例分割。我们提出了一个稠密三维引用网络ConcreteNet，其中包含三个独立的新模块，旨在改进具有相同语义类别干扰因素的具有挑战性的重复实例的引用性能。首先，我们引入了一个自下而上的注意力融合模块，旨在消除实例间关系线索的歧义性。接下来，我们构造一个cont

    3D visual grounding is the task of localizing the object in a 3D scene which is referred by a description in natural language. With a wide range of applications ranging from autonomous indoor robotics to AR/VR, the task has recently risen in popularity. A common formulation to tackle 3D visual grounding is grounding-by-detection, where localization is done via bounding boxes. However, for real-life applications that require physical interactions, a bounding box insufficiently describes the geometry of an object. We therefore tackle the problem of dense 3D visual grounding, i.e. referral-based 3D instance segmentation. We propose a dense 3D grounding network ConcreteNet, featuring three novel stand-alone modules which aim to improve grounding performance for challenging repetitive instances, i.e. instances with distractors of the same semantic class. First, we introduce a bottom-up attentive fusion module that aims to disambiguate inter-instance relational cues, next we construct a cont
    
[^10]: MOCA: 自监督学习通过预测掩码式在线码本分配实现表示学习

    MOCA: Self-supervised Representation Learning by Predicting Masked Online Codebook Assignments. (arXiv:2307.09361v1 [cs.CV])

    [http://arxiv.org/abs/2307.09361](http://arxiv.org/abs/2307.09361)

    MOCA是一种自监督学习方法，通过预测掩码式在线码本分配来实现表示学习。它同时具备良好的语境推理属性和对图像扰动的不变性，并在低样本设置和各种评估协议中取得了最新的最先进结果，训练速度比之前的方法快3倍以上。

    

    自监督学习可以用于缓解Vision Transformer网络对大型全注释数据集的贪婪需求。不同类别的自监督学习提供了具有良好语境推理属性的表示，例如使用掩码图像建模策略，或者对图像扰动具有不变性的表示，例如使用对比方法。在这项工作中，我们提出了一种单阶段、独立的方法MOCA，使用基于高级特征（而不是像素级细节）定义的新型掩码和预测目标来统一这两种期望的属性。此外，我们展示了如何以协同和计算高效的方式有效地应用这两种学习范式。通过这样做，我们在低样本设置上实现了新的最先进结果，并且在各种评估协议中取得了强大的实验结果，其训练速度至少比之前的方法快3倍。

    Self-supervised learning can be used for mitigating the greedy needs of Vision Transformer networks for very large fully-annotated datasets. Different classes of self-supervised learning offer representations with either good contextual reasoning properties, e.g., using masked image modeling strategies, or invariance to image perturbations, e.g., with contrastive methods. In this work, we propose a single-stage and standalone method, MOCA, which unifies both desired properties using novel mask-and-predict objectives defined with high-level features (instead of pixel-level details). Moreover, we show how to effectively employ both learning paradigms in a synergistic and computation-efficient way. Doing so, we achieve new state-of-the-art results on low-shot settings and strong experimental results in various evaluation protocols with a training that is at least 3 times faster than prior methods.
    
[^11]: 鲁棒语义分割：强鲁棒性攻击和快速训练鲁棒性模型

    Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models. (arXiv:2306.12941v1 [cs.CV])

    [http://arxiv.org/abs/2306.12941](http://arxiv.org/abs/2306.12941)

    本文提出了针对语义分割模型的解决方案，使得可以对其进行攻击并提供了更好的评估协议。同时，通过微调鲁棒的主干，可以有限的计算代价训练对抗性鲁棒的分割模型。

    

    虽然大量的工作已经集中在设计针对图像分类器的对抗性攻击上，但只有少数方法存在用于攻击语义分割模型。我们展示了攻击分割模型的任务特定挑战，并提出了新的解决方案。我们的最终评估协议优于现有方法，并表明这些方法可能高估了模型的鲁棒性。此外，至今最成功的获得鲁棒图像分类器的对抗性训练无法成功应用于语义分割。我们认为这是因为要学习的任务更具挑战性，需要比图像分类更高的计算量。作为解决方法，我们展示了通过利用最近在鲁棒ImageNet分类器方面的进展，可以通过微调鲁棒的主干，以有限的计算代价训练对抗性鲁棒的分割模型。

    While a large amount of work has focused on designing adversarial attacks against image classifiers, only a few methods exist to attack semantic segmentation models. We show that attacking segmentation models presents task-specific challenges, for which we propose novel solutions. Our final evaluation protocol outperforms existing methods, and shows that those can overestimate the robustness of the models. Additionally, so far adversarial training, the most successful way for obtaining robust image classifiers, could not be successfully applied to semantic segmentation. We argue that this is because the task to be learned is more challenging, and requires significantly higher computational effort than for image classification. As a remedy, we show that by taking advantage of recent advances in robust ImageNet classifiers, one can train adversarially robust segmentation models at limited computational cost by fine-tuning robust backbones.
    
[^12]: CompoDiff: 基于潜在扩散的多功能组合图像检索

    CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion. (arXiv:2303.11916v1 [cs.CV])

    [http://arxiv.org/abs/2303.11916](http://arxiv.org/abs/2303.11916)

    CompoDiff 是一种多功能的组合图像检索模型，通过接受各种条件，具有潜在扩散的能力，并在 FashionIQ 上实现了新的零样本最新技术水平。其特征位于完整的 CLIP 嵌入空间中，可以直接用于所有利用 CLIP 空间的模型。

    

    本文提出了一种新颖的基于扩散的模型 CompoDiff，用于解决具有潜在扩散的组合图像检索（CIR）问题，并提供了一个由 1800 万个参考图像、条件和相应的目标图像三元组组成的新数据集，用于训练模型。CompoDiff 不仅在像 FashionIQ 这样的 CIR 基准测试上实现了新的零样本最新技术水平，而且还通过接收各种条件（如负文本和图像遮罩条件），使得 CIR 更加多功能，这是现有 CIR 方法所不具备的。此外，CompoDiff 特征位于完整的 CLIP 嵌入空间中，因此它们可以直接用于利用 CLIP 空间的所有现有模型。训练所使用的代码和数据集，以及预训练权重可在 https://github.com/navervision/CompoDiff 上获得。

    This paper proposes a novel diffusion-based model, CompoDiff, for solving Composed Image Retrieval (CIR) with latent diffusion and presents a newly created dataset of 18 million reference images, conditions, and corresponding target image triplets to train the model. CompoDiff not only achieves a new zero-shot state-of-the-art on a CIR benchmark such as FashionIQ but also enables a more versatile CIR by accepting various conditions, such as negative text and image mask conditions, which are unavailable with existing CIR methods. In addition, the CompoDiff features are on the intact CLIP embedding space so that they can be directly used for all existing models exploiting the CLIP space. The code and dataset used for the training, and the pre-trained weights are available at https://github.com/navervision/CompoDiff
    

