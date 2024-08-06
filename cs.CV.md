# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering](https://arxiv.org/abs/2403.05053) | 本文提出了PrimeComposer，一种更快的逐步组合扩散方式，用于图像合成，主要专注于前景生成，从而解决了合成中的凝聚混乱和外观信息丢失问题，并避免了不必要的背景生成导致的前景生成质量下降。 |
| [^2] | [Enhancing Conceptual Understanding in Multimodal Contrastive Learning through Hard Negative Samples](https://arxiv.org/abs/2403.02875) | 提出了一种通过硬负样本改进多模态对比学习中概念理解的方法，并引入了一个评估视觉-语言模型中颜色、对象和大小细粒度对齐的新数据集。 |
| [^3] | [NiNformer: A Network in Network Transformer with Token Mixing Generated Gating Function](https://arxiv.org/abs/2403.02411) | 提出了一种新的计算块，称为NiNformer，具有令牌混合生成门控功能，以解决注意机制在深度学习中的计算成本高昂和数据集要求大的缺点。 |
| [^4] | [Zero shot VLMs for hate meme detection: Are we there yet?](https://arxiv.org/abs/2402.12198) | 本研究探讨了零-shot分类在处理复杂任务如恶意模因检测中的有效性 |
| [^5] | [AONeuS: A Neural Rendering Framework for Acoustic-Optical Sensor Fusion](https://arxiv.org/abs/2402.03309) | AONeuS是一种基于物理的多模态声光神经表面重建框架，通过融合高分辨率RGB测量和低分辨率深度成像声纳测量，能够在受限基线下实现准确的高分辨率三维表面重建。 |
| [^6] | [Fun with Flags: Robust Principal Directions via Flag Manifolds.](http://arxiv.org/abs/2401.04071) | 本研究提出了一种统一的PCA和其变种框架，该框架基于线性子空间旗帜，并引入了对异常值和数据流形的考虑。通过在旗帜流形上进行优化问题的求解，结合主测地线近似，提出了一系列新的降维算法。 |
| [^7] | [Open Sesame! Universal Black Box Jailbreaking of Large Language Models.](http://arxiv.org/abs/2309.01446) | 本文提出了一种使用遗传算法的新颖方法，可以在无法访问模型架构和参数的情况下操纵大规模语言模型 (LLMs)。通过优化通用对抗提示与用户查询结合，可以扰乱被攻击模型的对齐，导致意外和潜在有害的输出。该方法可以揭示模型的局限性和漏洞，为负责任的AI开发提供了一种诊断工具。 |
| [^8] | [FreeDrag: Point Tracking is Not What You Need for Interactive Point-based Image Editing.](http://arxiv.org/abs/2307.04684) | FreeDrag提出了一种基于特征的方法来解决DragGAN在点追踪方面的困难，通过自适应模板特征、线性搜索和模糊定位技术，实现了稳定和高效的基于点的图像编辑。 |
| [^9] | [Vision Learners Meet Web Image-Text Pairs.](http://arxiv.org/abs/2301.07088) | 本论文提出了一种基于网络数据的新型视觉学习方法MUlti-modal Generator (MUG)。在视觉数据集的转移学习任务上取得了最先进的表现，是之前最佳结果的3.4%和2.2%的提升。 |

# 详细

[^1]: PrimeComposer：用于图像合成的快速逐步组合扩散方法和带有注意力引导的技术

    PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering

    [https://arxiv.org/abs/2403.05053](https://arxiv.org/abs/2403.05053)

    本文提出了PrimeComposer，一种更快的逐步组合扩散方式，用于图像合成，主要专注于前景生成，从而解决了合成中的凝聚混乱和外观信息丢失问题，并避免了不必要的背景生成导致的前景生成质量下降。

    

    图像合成涉及将给定对象无缝地整合到特定的视觉环境中。目前无需训练的方法依赖于从几个采样器中组合注意力权重来引导生成器。然而，由于这些权重来自不同的上下文，它们的组合导致在合成中凝聚混乱和外观信息的丢失。在该任务中，它们过多关注背景生成，即使在这项任务中是不必要的，这些问题恶化。这不仅减慢了推理速度，还损害了前景生成质量。此外，这些方法还在过渡区域引入了不需要的伪影。在本文中，我们将图像合成形式化为一项基于主题的局部编辑任务，仅专注于前景生成。在每一步中，编辑后的前景与噪声背景相结合，以保持场景一致性。为了解决剩下的问题，我们提出了PrimeComposer，一种更快的tr

    arXiv:2403.05053v1 Announce Type: cross  Abstract: Image composition involves seamlessly integrating given objects into a specific visual context. The current training-free methods rely on composing attention weights from several samplers to guide the generator. However, since these weights are derived from disparate contexts, their combination leads to coherence confusion in synthesis and loss of appearance information. These issues worsen with their excessive focus on background generation, even when unnecessary in this task. This not only slows down inference but also compromises foreground generation quality. Moreover, these methods introduce unwanted artifacts in the transition area. In this paper, we formulate image composition as a subject-based local editing task, solely focusing on foreground generation. At each step, the edited foreground is combined with the noisy background to maintain scene consistency. To address the remaining issues, we propose PrimeComposer, a faster tr
    
[^2]: 通过硬负样本增强多模态对比学习中的概念理解

    Enhancing Conceptual Understanding in Multimodal Contrastive Learning through Hard Negative Samples

    [https://arxiv.org/abs/2403.02875](https://arxiv.org/abs/2403.02875)

    提出了一种通过硬负样本改进多模态对比学习中概念理解的方法，并引入了一个评估视觉-语言模型中颜色、对象和大小细粒度对齐的新数据集。

    

    当前利用对比学习的多模态模型在发展精细的概念理解方面通常存在一些限制。在预训练过程中，由于随机负样本，导致几乎只有非常不同的概念进行损失函数比较。因此，模型在处理细粒度语义差异时遇到困难。为了解决这个问题，我们引入了一种新颖的预训练方法，结合了合成的硬负文本示例。这些硬负样本对应于视觉概念的排列，导致更精细的视觉和文本概念对齐。此外，我们引入了InpaintCOCO，一个用于评估视觉-语言模型中颜色、对象和大小细粒度对齐的新挑战性数据集。我们使用从COCO图像生成的信息填充来创建数据集，通过改变视觉概念，使图像不再与其原始标题匹配。我们的结果显示...

    arXiv:2403.02875v1 Announce Type: cross  Abstract: Current multimodal models leveraging contrastive learning often face limitations in developing fine-grained conceptual understanding. This is due to random negative samples during pretraining, causing almost exclusively very dissimilar concepts to be compared in the loss function. Consequently, the models struggle with fine-grained semantic differences. To address this problem, we introduce a novel pretraining method incorporating synthetic hard negative text examples. The hard negatives permute terms corresponding to visual concepts, leading to a more fine-grained visual and textual concept alignment. Further, we introduce InpaintCOCO, a new challenging dataset for assessing the fine-grained alignment of colors, objects, and sizes in vision-language models. We created the dataset using generative inpainting from COCO images by changing the visual concepts so that the images no longer match their original captions. Our results show sig
    
[^3]: NiNformer: 一种具有令牌混合生成门控功能的网络中网络变压器

    NiNformer: A Network in Network Transformer with Token Mixing Generated Gating Function

    [https://arxiv.org/abs/2403.02411](https://arxiv.org/abs/2403.02411)

    提出了一种新的计算块，称为NiNformer，具有令牌混合生成门控功能，以解决注意机制在深度学习中的计算成本高昂和数据集要求大的缺点。

    

    注意机制是Transformer架构的主要组件，自引入以来，在深度学习领域取得了显著进展，跨越了许多领域和多个任务。该机制在计算机视觉中被应用为Vision Transformer ViT，并且其用途已扩展到视觉领域的许多任务，如分类、分割、目标检测和图像生成。尽管该机制非常具有表现力和能力，但其缺点是计算成本高昂，需要大规模数据集来有效优化。为了解决这些缺点，文献中提出了许多设计来减轻计算负担和缓解数据大小要求。在视觉领域的一些尝试的例子包括MLP-Mixer、Conv-Mixer、Perciver-IO等。本文介绍了一种新的计算块，作为一种

    arXiv:2403.02411v1 Announce Type: cross  Abstract: The Attention mechanism is the main component of the Transformer architecture, and since its introduction, it has led to significant advancements in Deep Learning that span many domains and multiple tasks. The Attention Mechanism was utilized in Computer Vision as the Vision Transformer ViT, and its usage has expanded into many tasks in the vision domain, such as classification, segmentation, object detection, and image generation. While this mechanism is very expressive and capable, it comes with the drawback of being computationally expensive and requiring datasets of considerable size for effective optimization. To address these shortcomings, many designs have been proposed in the literature to reduce the computational burden and alleviate the data size requirements. Examples of such attempts in the vision domain are the MLP-Mixer, the Conv-Mixer, the Perciver-IO, and many more. This paper introduces a new computational block as an 
    
[^4]: 零-shot 可见语言模型用于仇恨模因检测：我们已经到达目标了吗？

    Zero shot VLMs for hate meme detection: Are we there yet?

    [https://arxiv.org/abs/2402.12198](https://arxiv.org/abs/2402.12198)

    本研究探讨了零-shot分类在处理复杂任务如恶意模因检测中的有效性

    

    社交媒体上的多媒体内容正在迅速发展，其中模因作为一种独特形式变得日益重要。不幸的是，一些恶意用户利用模因针对个人或易受攻击的社区，因此有必要识别和解决此类恶意模因。已经进行了大量研究来解决这个问题，通过开发仇恨模因检测模型。然而，传统的机器学习/深度学习模型的一个显著局限性是需要带标签的数据集才能进行准确分类。最近，研究界见证了几种可见语言模型的出现，在各种任务中展现出卓越的性能。在这项研究中，我们旨在调查这些可见语言模型在处理诸如仇恨模因检测等复杂任务中的有效性。我们使用各种提示设置来专注于对恶意/有害模因的零-shot 分类。通过我们的分析，我们o

    arXiv:2402.12198v1 Announce Type: new  Abstract: Multimedia content on social media is rapidly evolving, with memes gaining prominence as a distinctive form. Unfortunately, some malicious users exploit memes to target individuals or vulnerable communities, making it imperative to identify and address such instances of hateful memes. Extensive research has been conducted to address this issue by developing hate meme detection models. However, a notable limitation of traditional machine/deep learning models is the requirement for labeled datasets for accurate classification. Recently, the research community has witnessed the emergence of several visual language models that have exhibited outstanding performance across various tasks. In this study, we aim to investigate the efficacy of these visual language models in handling intricate tasks such as hate meme detection. We use various prompt settings to focus on zero-shot classification of hateful/harmful memes. Through our analysis, we o
    
[^5]: AONeuS: 一种用于声光传感器融合的神经渲染框架

    AONeuS: A Neural Rendering Framework for Acoustic-Optical Sensor Fusion

    [https://arxiv.org/abs/2402.03309](https://arxiv.org/abs/2402.03309)

    AONeuS是一种基于物理的多模态声光神经表面重建框架，通过融合高分辨率RGB测量和低分辨率深度成像声纳测量，能够在受限基线下实现准确的高分辨率三维表面重建。

    

    水下感知和三维表面重建是具有广泛应用的挑战性问题，涉及建筑、安全、海洋考古和环境监测等领域。恶劣的操作条件、脆弱的环境和有限的导航控制通常导致水下航行器限制其运动范围和测量基线。在三维场景重建的背景下，我们知道较小的基线会增加重建难度。本文开发了一种基于物理的多模态声光神经表面重建框架（AONeuS），能够有效地将高分辨率RGB测量与低分辨率深度成像声纳测量进行融合。通过融合这些互补的模态，我们的框架可以从在受限基线上捕获的测量中重建出准确的高分辨率三维表面。通过大量的模拟和实验室实验，我们证明了...

    Underwater perception and 3D surface reconstruction are challenging problems with broad applications in construction, security, marine archaeology, and environmental monitoring. Treacherous operating conditions, fragile surroundings, and limited navigation control often dictate that submersibles restrict their range of motion and, thus, the baseline over which they can capture measurements. In the context of 3D scene reconstruction, it is well-known that smaller baselines make reconstruction more challenging. Our work develops a physics-based multimodal acoustic-optical neural surface reconstruction framework (AONeuS) capable of effectively integrating high-resolution RGB measurements with low-resolution depth-resolved imaging sonar measurements. By fusing these complementary modalities, our framework can reconstruct accurate high-resolution 3D surfaces from measurements captured over heavily-restricted baselines. Through extensive simulations and in-lab experiments, we demonstrate tha
    
[^6]: 旗帜游戏：通过旗帜流形来获得鲁棒的主方向

    Fun with Flags: Robust Principal Directions via Flag Manifolds. (arXiv:2401.04071v1 [cs.CV])

    [http://arxiv.org/abs/2401.04071](http://arxiv.org/abs/2401.04071)

    本研究提出了一种统一的PCA和其变种框架，该框架基于线性子空间旗帜，并引入了对异常值和数据流形的考虑。通过在旗帜流形上进行优化问题的求解，结合主测地线近似，提出了一系列新的降维算法。

    

    主成分分析（PCA）及其对流形和异常数据的扩展，在计算机视觉和机器学习中是不可或缺的。本研究提出了PCA及其变种的统一形式，引入了基于线性子空间旗帜的框架，即逐渐增加维度的嵌套线性子空间的层次结构，不仅允许共同实现，还产生了新的未曾探索的变种。我们从广义化传统的PCA方法开始，这些方法要么最大化方差，要么最小化重构误差。我们扩展这些解释，通过考虑异常值和数据流形，开发出了大量新的降维算法。为了设计一种通用的计算方法，我们将鲁棒和对偶形式的PCA重新构建为在旗帜流形上的优化问题。然后，我们将主测地线近似（切线PCA）整合到这个基于旗帜的框架中，创造出一种新的方法。

    Principal component analysis (PCA), along with its extensions to manifolds and outlier contaminated data, have been indispensable in computer vision and machine learning. In this work, we present a unifying formalism for PCA and its variants, and introduce a framework based on the flags of linear subspaces, \ie a hierarchy of nested linear subspaces of increasing dimension, which not only allows for a common implementation but also yields novel variants, not explored previously. We begin by generalizing traditional PCA methods that either maximize variance or minimize reconstruction error. We expand these interpretations to develop a wide array of new dimensionality reduction algorithms by accounting for outliers and the data manifold. To devise a common computational approach, we recast robust and dual forms of PCA as optimization problems on flag manifolds. We then integrate tangent space approximations of principal geodesic analysis (tangent-PCA) into this flag-based framework, crea
    
[^7]: 开门吧！大规模语言模型的通用黑盒破解

    Open Sesame! Universal Black Box Jailbreaking of Large Language Models. (arXiv:2309.01446v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.01446](http://arxiv.org/abs/2309.01446)

    本文提出了一种使用遗传算法的新颖方法，可以在无法访问模型架构和参数的情况下操纵大规模语言模型 (LLMs)。通过优化通用对抗提示与用户查询结合，可以扰乱被攻击模型的对齐，导致意外和潜在有害的输出。该方法可以揭示模型的局限性和漏洞，为负责任的AI开发提供了一种诊断工具。

    

    大规模语言模型（LLMs）旨在提供有帮助和安全的回复，通常依赖于对齐技术与用户意图和社会指南保持一致。然而，这种对齐可能会被恶意行为者利用，以用于意想不到的目的。在本文中，我们引入了一种新颖的方法，利用遗传算法（GA）在模型架构和参数不可访问时操纵LLMs。GA攻击通过优化通用对抗提示与用户查询结合，扰乱被攻击模型的对齐，导致意外和潜在有害的输出。我们的新颖方法通过揭示模型的局限性和漏洞，系统地揭示了其响应与预期行为不符的情况。通过广泛的实验，我们证明了我们的技术的有效性，从而为关于负责任的AI开发的讨论提供了一种诊断工具。

    Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool 
    
[^8]: FreeDrag: 点追踪并不适用于交互式的基于点的图像编辑

    FreeDrag: Point Tracking is Not What You Need for Interactive Point-based Image Editing. (arXiv:2307.04684v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.04684](http://arxiv.org/abs/2307.04684)

    FreeDrag提出了一种基于特征的方法来解决DragGAN在点追踪方面的困难，通过自适应模板特征、线性搜索和模糊定位技术，实现了稳定和高效的基于点的图像编辑。

    

    为了满足图像编辑的复杂和多样化需求，对图像内容的精确和灵活的操纵是不可或缺的。最近，DragGAN通过基于点的操纵实现了令人印象深刻的编辑结果。然而，我们观察到DragGAN在点的追踪上存在困难，包括错误追踪和模糊追踪。为了解决这些问题，我们提出了FreeDrag，它采用了基于特征的方法来减轻DragGAN中点追踪的负担。FreeDrag结合了自适应模板特征、线性搜索和模糊定位技术，实现了稳定和高效的基于点的图像编辑。大量实验表明，我们的方法优于DragGAN，并能在具有相似特征的困难情景下实现稳定的基于点的编辑。

    To serve the intricate and varied demands of image editing, precise and flexible manipulation of image content is indispensable. Recently, DragGAN has achieved impressive editing results through point-based manipulation. However, we have observed that DragGAN struggles with miss tracking, where DragGAN encounters difficulty in effectively tracking the desired handle points, and ambiguous tracking, where the tracked points are situated within other regions that bear resemblance to the handle points. To deal with the above issues, we propose FreeDrag, which adopts a feature-oriented approach to free the burden on point tracking within the point-oriented methodology of DragGAN. The FreeDrag incorporates adaptive template features, line search, and fuzzy localization techniques to perform stable and efficient point-based image editing. Extensive experiments demonstrate that our method is superior to the DragGAN and enables stable point-based editing in challenging scenarios with similar st
    
[^9]: 视觉学习者遇见Web图像-文本对

    Vision Learners Meet Web Image-Text Pairs. (arXiv:2301.07088v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.07088](http://arxiv.org/abs/2301.07088)

    本论文提出了一种基于网络数据的新型视觉学习方法MUlti-modal Generator (MUG)。在视觉数据集的转移学习任务上取得了最先进的表现，是之前最佳结果的3.4%和2.2%的提升。

    

    大多数最新的自监督学习方法都是在维护良好的ImageNet-1K数据集上进行预训练的。在本研究中，考虑到网络数据的出色可伸缩性，我们认为自我监督预训练应该基于嘈杂的网络源图文配对数据。首先，我们在如此设置下，对大规模网络数据上的代表性自监督预训练方法进行了基准研究。我们比较了一系列方法，包括使用被屏蔽的训练目标的单模式方法和使用图像-文本对比训练的多模式方法。我们发现，现有的多模态方法在视觉转移学习任务上并不比单模态方法表现更好。我们提出了一个信息论视角来解释这些基准结果，这提供了如何设计新型视觉学习者的见解。受到这些见解的启发，我们提出了一种新的视觉表示预训练方法——多模式生成器（MUG），它从可伸缩的网络源图文数据中学习。MUG在几个视觉数据集的转移学习任务上取得了最先进的性能，在CIFAR-10上优于之前最佳的结果3.4％，在STL-10上优于之前最佳的结果2.2％。

    Most recent self-supervised learning methods are pre-trained on the well-curated ImageNet-1K dataset. In this work, given the excellent scalability of web data, we consider self-supervised pre-training on noisy web sourced image-text paired data. First, we conduct a benchmark study of representative self-supervised pre-training methods on large-scale web data in a like-for-like setting. We compare a range of methods, including single-modal ones that use masked training objectives and multi-modal ones that use image-text constrastive training. We observe that existing multi-modal methods do not outperform their single-modal counterparts on vision transfer learning tasks. We derive an information-theoretical view to explain these benchmark results, which provides insight into how to design a novel vision learner. Inspired by this insight, we present a new visual representation pre-training method, MUlti-modal Generator~(MUG), that learns from scalable web sourced image-text data. MUG ach
    

