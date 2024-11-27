# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Generalizable Feature Fields for Mobile Manipulation](https://arxiv.org/abs/2403.07563) | 提出了GeFF（通用特征场），作为导航和操作的统一表示，可以实时执行，通过将生成的丰富场景先验与自然语言对齐来提高效果。 |
| [^2] | [Semi-Supervised Semantic Segmentation Based on Pseudo-Labels: A Survey](https://arxiv.org/abs/2403.01909) | 这项综述提供了关于基于伪标签方法在半监督语义分割领域最新研究成果的全面且有组织的概述，探讨了伪标签技术在不同应用领域的具体方法，还研究了其在医学和遥感图像分割中的应用，提出了未来研究方向。 |
| [^3] | [BioNeRF: Biologically Plausible Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2402.07310) | BioNeRF是一种生物合理的架构，通过辐射场对场景进行3D表示并合成新视图。它实现了一种认知启发的机制，提高了存储能力和提取信息的能力，并在真实世界图像和合成数据的两个数据集上超过了以人的感知为基础的质量度量的最新结果。 |
| [^4] | [Direction-Oriented Visual-semantic Embedding Model for Remote Sensing Image-text Retrieval.](http://arxiv.org/abs/2310.08276) | 这篇论文提出了一种面向方向的视觉-语义嵌入模型（DOVE），通过区域导向的注意力模块和轻量级的文字基因辅助模块，解决了遥感图像-文本检索中的视觉-语义不平衡问题，提高了检索准确性。 |
| [^5] | [A Survey on Multimodal Large Language Models.](http://arxiv.org/abs/2306.13549) | 本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。 |

# 详细

[^1]: 学习移动操作的通用特征场

    Learning Generalizable Feature Fields for Mobile Manipulation

    [https://arxiv.org/abs/2403.07563](https://arxiv.org/abs/2403.07563)

    提出了GeFF（通用特征场），作为导航和操作的统一表示，可以实时执行，通过将生成的丰富场景先验与自然语言对齐来提高效果。

    

    移动操作中的一个悬而未决的问题是如何以统一的方式表示物体和场景，使得机器人可以同时用于在环境中导航和操作物体。本工作提出了GeFF（通用特征场），这是一个场景级的通用神经特征场，作为导航和操作的统一表示，可以实时执行。为此，我们将生成新视图合成视为一个预训练任务，然后通过CLIP特征提炼将生成的丰富场景先验与自然语言对齐。

    arXiv:2403.07563v1 Announce Type: cross  Abstract: An open problem in mobile manipulation is how to represent objects and scenes in a unified manner, so that robots can use it both for navigating in the environment and manipulating objects. The latter requires capturing intricate geometry while understanding fine-grained semantics, whereas the former involves capturing the complexity inherit to an expansive physical scale. In this work, we present GeFF (Generalizable Feature Fields), a scene-level generalizable neural feature field that acts as a unified representation for both navigation and manipulation that performs in real-time. To do so, we treat generative novel view synthesis as a pre-training task, and then align the resulting rich scene priors with natural language via CLIP feature distillation. We demonstrate the effectiveness of this approach by deploying GeFF on a quadrupedal robot equipped with a manipulator. We evaluate GeFF's ability to generalize to open-set objects as 
    
[^2]: 基于伪标签的半监督语义分割：综述

    Semi-Supervised Semantic Segmentation Based on Pseudo-Labels: A Survey

    [https://arxiv.org/abs/2403.01909](https://arxiv.org/abs/2403.01909)

    这项综述提供了关于基于伪标签方法在半监督语义分割领域最新研究成果的全面且有组织的概述，探讨了伪标签技术在不同应用领域的具体方法，还研究了其在医学和遥感图像分割中的应用，提出了未来研究方向。

    

    语义分割是计算机视觉中一个重要且热门的研究领域，侧重于基于语义对图像中的像素进行分类。然而，监督学习需要大量数据来训练模型，而逐像素标记图像的过程耗时且繁琐。本综述旨在提供半监督语义分割领域中伪标签方法的最新研究成果的首次综合和有组织的概述，我们从不同角度对其进行分类，并提出了针对特定应用领域的具体方法。此外，我们还探讨了伪标签技术在医学和遥感图像分割中的应用。最后，我们还提出了一些可行的未来研究方向，以解决现有挑战。

    arXiv:2403.01909v1 Announce Type: cross  Abstract: Semantic segmentation is an important and popular research area in computer vision that focuses on classifying pixels in an image based on their semantics. However, supervised deep learning requires large amounts of data to train models and the process of labeling images pixel by pixel is time-consuming and laborious. This review aims to provide a first comprehensive and organized overview of the state-of-the-art research results on pseudo-label methods in the field of semi-supervised semantic segmentation, which we categorize from different perspectives and present specific methods for specific application areas. In addition, we explore the application of pseudo-label technology in medical and remote-sensing image segmentation. Finally, we also propose some feasible future research directions to address the existing challenges.
    
[^3]: BioNeRF: 用于视图合成的生物合理神经辐射场

    BioNeRF: Biologically Plausible Neural Radiance Fields for View Synthesis

    [https://arxiv.org/abs/2402.07310](https://arxiv.org/abs/2402.07310)

    BioNeRF是一种生物合理的架构，通过辐射场对场景进行3D表示并合成新视图。它实现了一种认知启发的机制，提高了存储能力和提取信息的能力，并在真实世界图像和合成数据的两个数据集上超过了以人的感知为基础的质量度量的最新结果。

    

    本文介绍了BioNeRF，一种生物合理的架构，它通过辐射场对场景进行3D表示并合成新视图。由于NeRF依赖于网络权重来存储场景的三维表示，BioNeRF实现了一种受认知启发的机制，将来自多个来源的输入融合成内存类似的结构，提高存储能力并提取更多内在和相关信息。BioNeRF还模仿了金字塔细胞中关于上下文信息的一种行为，其中内存作为上下文提供，并与两个后续神经模型的输入相结合，一个负责生成容积密度，另一个负责渲染场景的颜色。实验结果表明，BioNeRF在两个数据集（真实世界图像和合成数据）上超过了以人的感知为基础的质量度量的最新结果。

    This paper presents BioNeRF, a biologically plausible architecture that models scenes in a 3D representation and synthesizes new views through radiance fields. Since NeRF relies on the network weights to store the scene's 3-dimensional representation, BioNeRF implements a cognitive-inspired mechanism that fuses inputs from multiple sources into a memory-like structure, improving the storing capacity and extracting more intrinsic and correlated information. BioNeRF also mimics a behavior observed in pyramidal cells concerning contextual information, in which the memory is provided as the context and combined with the inputs of two subsequent neural models, one responsible for producing the volumetric densities and the other the colors used to render the scene. Experimental results show that BioNeRF outperforms state-of-the-art results concerning a quality measure that encodes human perception in two datasets: real-world images and synthetic data.
    
[^4]: 面向方向的视觉-语义嵌入模型在遥感图像-文本检索中的应用

    Direction-Oriented Visual-semantic Embedding Model for Remote Sensing Image-text Retrieval. (arXiv:2310.08276v1 [cs.CV])

    [http://arxiv.org/abs/2310.08276](http://arxiv.org/abs/2310.08276)

    这篇论文提出了一种面向方向的视觉-语义嵌入模型（DOVE），通过区域导向的注意力模块和轻量级的文字基因辅助模块，解决了遥感图像-文本检索中的视觉-语义不平衡问题，提高了检索准确性。

    

    图像-文本检索在近年来得到了快速发展，然而在遥感领域仍然存在着视觉-语义不平衡的挑战，这导致了非语义视觉和文本特征的错误匹配。为了解决这个问题，我们提出了一种新颖的面向方向的视觉-语义嵌入模型（DOVE），来挖掘视觉和语言之间的关系。具体而言，通过区域导向的注意力模块（ROAM），在潜在的语义空间中，根据区域视觉特征自适应地调整最终的视觉和文本嵌入之间的距离。同时，设计了一个轻量级的文字基因辅助模块（DTGA），用较少的注意力操作来扩展可处理的文本表示范围，增强全局词级语义连接。最后，我们利用全局视觉-语义约束来减少单一视觉依赖，并为最终的视觉和文本表示提供外部约束。

    Image-text retrieval has developed rapidly in recent years. However, it is still a challenge in remote sensing due to visual-semantic imbalance, which leads to incorrect matching of non-semantic visual and textual features. To solve this problem, we propose a novel Direction-Oriented Visual-semantic Embedding Model (DOVE) to mine the relationship between vision and language. Concretely, a Regional-Oriented Attention Module (ROAM) adaptively adjusts the distance between the final visual and textual embeddings in the latent semantic space, oriented by regional visual features. Meanwhile, a lightweight Digging Text Genome Assistant (DTGA) is designed to expand the range of tractable textual representation and enhance global word-level semantic connections using less attention operations. Ultimately, we exploit a global visual-semantic constraint to reduce single visual dependency and serve as an external constraint for the final visual and textual representations. The effectiveness and su
    
[^5]: 多模态大语言模型综述

    A Survey on Multimodal Large Language Models. (arXiv:2306.13549v1 [cs.CV])

    [http://arxiv.org/abs/2306.13549](http://arxiv.org/abs/2306.13549)

    本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。

    

    多模态大语言模型（MLLM）是一种新兴的研究热点，使用强大的大语言模型作为大脑执行多模态任务。MLLM 的惊人能力，如基于图像编写故事和无OCR数学推理等，在传统方法中很少见，表明了通向人工智能的潜在路径。本文旨在追踪和总结 MLLM 的最新进展。首先，我们介绍了 MLLM 的构成，概述了相关概念。然后，讨论了关键技术和应用，包括多模态指令调整（M-IT）、多模态上下文学习（M-ICL）、多模态思维链（M-CoT）和LLM辅助视觉推理（LAVR）。最后，我们讨论了现有的挑战，并指出了有前途的研究方向。鉴于 MLLM 时代才刚刚开始，我们会不断更新这个综述，并希望能激发更多的研究。

    Multimodal Large Language Model (MLLM) recently has been a new rising research hotspot, which uses powerful Large Language Models (LLMs) as a brain to perform multimodal tasks. The surprising emergent capabilities of MLLM, such as writing stories based on images and OCR-free math reasoning, are rare in traditional methods, suggesting a potential path to artificial general intelligence. In this paper, we aim to trace and summarize the recent progress of MLLM. First of all, we present the formulation of MLLM and delineate its related concepts. Then, we discuss the key techniques and applications, including Multimodal Instruction Tuning (M-IT), Multimodal In-Context Learning (M-ICL), Multimodal Chain of Thought (M-CoT), and LLM-Aided Visual Reasoning (LAVR). Finally, we discuss existing challenges and point out promising research directions. In light of the fact that the era of MLLM has only just begun, we will keep updating this survey and hope it can inspire more research. An associated
    

