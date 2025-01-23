# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Serpent: Scalable and Efficient Image Restoration via Multi-scale Structured State Space Models](https://arxiv.org/abs/2403.17902) | Serpent提出了一种新的图像恢复架构，利用状态空间模型在全局感受野和计算效率之间取得平衡，实现了与最先进技术相当的重建质量，但计算量减少了数个数量级。 |
| [^2] | [UrbanVLP: A Multi-Granularity Vision-Language Pre-Trained Foundation Model for Urban Indicator Prediction](https://arxiv.org/abs/2403.16831) | UrbanVLP是一种多粒度信息集成的视觉语言预训练模型，旨在克服目前城市指标预测中预训练模型的局限性，提高了可解释性和精度 |
| [^3] | [Cartoon Hallucinations Detection: Pose-aware In Context Visual Learning](https://arxiv.org/abs/2403.15048) | 该研究提出了一种用于检测由TTI模型生成的卡通角色图像中视觉幻觉的系统，通过结合姿势感知上下文视觉学习和视觉语言模型，利用RGB图像和姿势信息，实现了更准确的决策，显著提高了视觉幻觉的识别能力，推动了TTI模型在非照片真实领域的发展。 |
| [^4] | [Learning to Mask and Permute Visual Tokens for Vision Transformer Pre-Training.](http://arxiv.org/abs/2306.07346) | 本论文提出了一种新的自监督预训练方法MaPeT，不同于现有的使用掩码图像模型的方法，该方法使用自回归和置换预测来捕获图像块内的依赖关系并减少数据噪声的影响，从而提高了下游任务的一致性。 |

# 详细

[^1]: Serpent：通过多尺度结构化状态空间模型实现可扩展高效的图像恢复

    Serpent: Scalable and Efficient Image Restoration via Multi-scale Structured State Space Models

    [https://arxiv.org/abs/2403.17902](https://arxiv.org/abs/2403.17902)

    Serpent提出了一种新的图像恢复架构，利用状态空间模型在全局感受野和计算效率之间取得平衡，实现了与最先进技术相当的重建质量，但计算量减少了数个数量级。

    

    有效图像恢复架构的计算建筑块领域，主要由卷积处理和各种注意机制的组合所主导。然而，卷积滤波器本质上是局部的，因此在建模图像的长距离依赖性方面存在困难。另一方面，注意机制擅长捕获任意图像区域之间的全局相互作用，但对图像尺寸的二次成本较高。在这项工作中，我们提出了Serpent，这是一种利用最近在状态空间模型（SSMs）方面的进展作为其核心计算模块的架构。SSMs最初用于序列建模，可以通过有利的输入尺寸的线性缩放来维持全局感受野。我们的初步结果表明，Serpent可以实现与最先进技术相当的重建质量，同时需要数量级的计算量较少（在FLOPS上高达150倍的减少）。

    arXiv:2403.17902v1 Announce Type: cross  Abstract: The landscape of computational building blocks of efficient image restoration architectures is dominated by a combination of convolutional processing and various attention mechanisms. However, convolutional filters are inherently local and therefore struggle at modeling long-range dependencies in images. On the other hand, attention excels at capturing global interactions between arbitrary image regions, however at a quadratic cost in image dimension. In this work, we propose Serpent, an architecture that leverages recent advances in state space models (SSMs) in its core computational block. SSMs, originally introduced for sequence modeling, can maintain a global receptive field with a favorable linear scaling in input size. Our preliminary results demonstrate that Serpent can achieve reconstruction quality on par with state-of-the-art techniques, while requiring orders of magnitude less compute (up to $150$ fold reduction in FLOPS) an
    
[^2]: UrbanVLP：用于城市指标预测的多粒度视觉语言预训练基础模型

    UrbanVLP: A Multi-Granularity Vision-Language Pre-Trained Foundation Model for Urban Indicator Prediction

    [https://arxiv.org/abs/2403.16831](https://arxiv.org/abs/2403.16831)

    UrbanVLP是一种多粒度信息集成的视觉语言预训练模型，旨在克服目前城市指标预测中预训练模型的局限性，提高了可解释性和精度

    

    城市指标预测旨在利用数据驱动方法推断不同城市景观中的社会经济指标。然而，目前流行的预训练模型，特别是依赖卫星图像的模型，面临着双重挑战。首先，仅集中在卫星数据中的宏观级别模式可能引入偏见，在微观级别缺乏细致的细节，例如某地的建筑细节。其次，预训练模型缺乏可解释性，限制了它们在提供城市规划透明证据方面的实用性。针对这些问题，本文设计了一种新颖的Vision-Language Pre-Trained Model（UrbanVLP）。我们的UrbanVLP无缝整合来自宏观（卫星）和微观（街景）级别的多粒度信息，克服了先前预训练模型的局限性。此外，它引入了自动生成文本和校准，提高了在下游应用中的可解释性。

    arXiv:2403.16831v1 Announce Type: cross  Abstract: Urban indicator prediction aims to infer socio-economic metrics in diverse urban landscapes using data-driven methods. However, prevalent pre-trained models, particularly those reliant on satellite imagery, face dual challenges. Firstly, concentrating solely on macro-level patterns from satellite data may introduce bias, lacking nuanced details at micro levels, such as architectural details at a place. Secondly, the lack of interpretability in pre-trained models limits their utility in providing transparent evidence for urban planning. In response to these issues, we devise a novel Vision-Language Pre-Trained Model (UrbanVLP) in this paper. Our UrbanVLP seamlessly integrates multi-granularity information from both macro (satellite) and micro (street-view) levels, overcoming the limitations of prior pre-trained models. Moreover, it introduces automatic text generation and calibration, elevating interpretability in downstream application
    
[^3]: 卡通幻觉检测: 姿势感知上下文视觉学习

    Cartoon Hallucinations Detection: Pose-aware In Context Visual Learning

    [https://arxiv.org/abs/2403.15048](https://arxiv.org/abs/2403.15048)

    该研究提出了一种用于检测由TTI模型生成的卡通角色图像中视觉幻觉的系统，通过结合姿势感知上下文视觉学习和视觉语言模型，利用RGB图像和姿势信息，实现了更准确的决策，显著提高了视觉幻觉的识别能力，推动了TTI模型在非照片真实领域的发展。

    

    大规模文本到图像（TTI）模型已经成为各种生成领域中生成训练数据的常见方法。然而，视觉幻觉，尤其是在非照片真实风格如卡通人物中包含了感知上关键的缺陷，依然是一个令人担忧的问题。我们提出了一种新颖的用于检测TTI模型生成的卡通角色图像的视觉幻觉检测系统。我们的方法利用了姿势感知上下文视觉学习（PA-ICVL）与视觉语言模型（VLMs），同时利用RGB图像和姿势信息。通过从一个经过微调的姿势估计器中获得姿势指导，我们使VLM能够做出更准确的决策。实验结果表明，在识别视觉幻觉方面，与仅依赖于RGB图像的基线方法相比，取得了显著的改进。这项研究通过减轻视觉幻觉，推动了TTI模型在非照片真实领域的潜力。

    arXiv:2403.15048v1 Announce Type: cross  Abstract: Large-scale Text-to-Image (TTI) models have become a common approach for generating training data in various generative fields. However, visual hallucinations, which contain perceptually critical defects, remain a concern, especially in non-photorealistic styles like cartoon characters. We propose a novel visual hallucination detection system for cartoon character images generated by TTI models. Our approach leverages pose-aware in-context visual learning (PA-ICVL) with Vision-Language Models (VLMs), utilizing both RGB images and pose information. By incorporating pose guidance from a fine-tuned pose estimator, we enable VLMs to make more accurate decisions. Experimental results demonstrate significant improvements in identifying visual hallucinations compared to baseline methods relying solely on RGB images. This research advances TTI models by mitigating visual hallucinations, expanding their potential in non-photorealistic domains.
    
[^4]: 学习用于视觉Transformer预训练的掩码和置换视觉令牌。

    Learning to Mask and Permute Visual Tokens for Vision Transformer Pre-Training. (arXiv:2306.07346v1 [cs.CV])

    [http://arxiv.org/abs/2306.07346](http://arxiv.org/abs/2306.07346)

    本论文提出了一种新的自监督预训练方法MaPeT，不同于现有的使用掩码图像模型的方法，该方法使用自回归和置换预测来捕获图像块内的依赖关系并减少数据噪声的影响，从而提高了下游任务的一致性。

    

    使用自监督预训练技术已成为提高图像分类等视觉任务性能的有前途的方法。最近的方法使用掩码图像模型范式，通过重构与随机掩码图像块相关联的视觉令牌来预训练骨干网络。然而，这种掩蔽方法会在预训练过程中引入噪声进入输入数据，导致性能下降。此外，输入掩蔽忽略了受损块之间的依赖关系，增加了下游微调任务中观察到的不一致性。为了解决这些问题，我们提出了一种新的自监督预训练方法，名为掩蔽和置换视觉变压器（MaPeT），它使用自回归和置换预测来捕获块内依赖性。此外，MaPeT使用辅助位置信息来减少预训练和微调阶段中的差异性。

    The use of self-supervised pre-training has emerged as a promising approach to enhance the performance of visual tasks such as image classification. In this context, recent approaches have employed the Masked Image Modeling paradigm, which pre-trains a backbone by reconstructing visual tokens associated with randomly masked image patches. This masking approach, however, introduces noise into the input data during pre-training, leading to discrepancies that can impair performance during the fine-tuning phase. Furthermore, input masking neglects the dependencies between corrupted patches, increasing the inconsistencies observed in downstream fine-tuning tasks. To overcome these issues, we propose a new self-supervised pre-training approach, named Masked and Permuted Vision Transformer (MaPeT), that employs autoregressive and permuted predictions to capture intra-patch dependencies. In addition, MaPeT employs auxiliary positional information to reduce the disparity between the pre-trainin
    

