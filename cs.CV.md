# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models](https://arxiv.org/abs/2403.02774) | 通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。 |
| [^3] | [Invariant Test-Time Adaptation for Vision-Language Model Generalization](https://arxiv.org/abs/2403.00376) | 本文提出了一个测试时提示调优范式，通过优化可学习的提示，迫使模型利用真正的因果不变特征，以解决视觉-语言模型在特定任务需求上无法有效利用预训练特征的挑战。 |
| [^4] | [Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words.](http://arxiv.org/abs/2307.09059) | 本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。 |
| [^5] | [PastNet: Introducing Physical Inductive Biases for Spatio-temporal Video Prediction.](http://arxiv.org/abs/2305.11421) | 本文介绍了一种名为PastNet的新颖方法，通过在傅里叶域中引入谱卷积算子，利用内在的物理知识生成高质量的时空视频预测，并通过离散化局部特征降低计算成本。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: 快速、自适应尺度和具有不确定性意识的地球系统模型场降尺度与生成基础模型

    Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models

    [https://arxiv.org/abs/2403.02774](https://arxiv.org/abs/2403.02774)

    通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。

    

    精确和高分辨率的地球系统模型(ESM)模拟对于评估人为气候变化对生态和社会经济影响至关重要，但计算成本过高。最近的机器学习方法在ESM模拟的降尺度中表现出色，优于最先进的统计方法。然而，现有方法对每个ESM都需要计算昂贵的重新训练，并且在训练期间未见过的气候预测效果差。我们通过学习一个一致性模型(CM)，以零样本方式高效准确地降尺度任意ESM模拟来解决这些缺点。我们的基础模型方法以只受观测参考数据限制的分辨率产生概率性降尺度场。我们展示了CM在维持高可控性的同时以较低的计算成本优于最先进的扩散模型。

    arXiv:2403.02774v1 Announce Type: cross  Abstract: Accurate and high-resolution Earth system model (ESM) simulations are essential to assess the ecological and socio-economic impacts of anthropogenic climate change, but are computationally too expensive. Recent machine learning approaches have shown promising results in downscaling ESM simulations, outperforming state-of-the-art statistical approaches. However, existing methods require computationally costly retraining for each ESM and extrapolate poorly to climates unseen during training. We address these shortcomings by learning a consistency model (CM) that efficiently and accurately downscales arbitrary ESM simulations without retraining in a zero-shot manner. Our foundation model approach yields probabilistic downscaled fields at resolution only limited by the observational reference data. We show that the CM outperforms state-of-the-art diffusion models at a fraction of computational cost while maintaining high controllability on
    
[^3]: 视觉-语言模型泛化的不变测试时适应性

    Invariant Test-Time Adaptation for Vision-Language Model Generalization

    [https://arxiv.org/abs/2403.00376](https://arxiv.org/abs/2403.00376)

    本文提出了一个测试时提示调优范式，通过优化可学习的提示，迫使模型利用真正的因果不变特征，以解决视觉-语言模型在特定任务需求上无法有效利用预训练特征的挑战。

    

    arXiv:2403.00376v1 公告类型: 交叉摘要: 视觉-语言基础模型在大量图像-文本配对数据集上的可扩展性使其在众多下游任务中展现出卓越成功。然而，这些模型在应用于长尾任务（如细粒度图像分类）时显示出明显局限，这是由于“决策捷径”导致了它们的泛化能力受限。本文发现CLIP模型具有丰富的特征集，涵盖了既有的\textit{期望不变因果特征}又有的\textit{不希望的决策捷径}。此外，CLIP在下游任务中的表现不佳源自其无法有效利用预训练特征以符合特定任务要求。为解决这一挑战，本文引入一种测试时提示调优范式，优化一个可学习的提示，从而促使模型利用真正的因果不变特征。

    arXiv:2403.00376v1 Announce Type: cross  Abstract: Vision-language foundation models have exhibited remarkable success across a multitude of downstream tasks due to their scalability on extensive image-text paired datasets. However, these models display significant limitations when applied to long-tail tasks, such as fine-grained image classification, as a result of "decision shortcuts" that hinders their generalization capabilities. In this work, we find that the CLIP model possesses a rich set of features, encompassing both \textit{desired invariant causal features} and \textit{undesired decision shortcuts}. Moreover, the underperformance of CLIP on downstream tasks originates from its inability to effectively utilize pre-trained features in accordance with specific task requirements. To address this challenge, this paper introduces a test-time prompt tuning paradigm that optimizes a learnable prompt, thereby compelling the model to exploit genuine causal invariant features while dis
    
[^4]: 文字想象的释放：通过探索文字的力量实现文本到图像的人物检索的新框架

    Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words. (arXiv:2307.09059v1 [cs.CL])

    [http://arxiv.org/abs/2307.09059](http://arxiv.org/abs/2307.09059)

    本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。

    

    文本到图像的人物检索的目标是从大型图库中检索与给定文本描述相匹配的人物图像。这个任务的主要挑战在于视觉和文本模态之间信息表示的显著差异。文本模态通过词汇和语法结构传递抽象和精确的信息，而视觉模态通过图像传递具体和直观的信息。为了充分利用文字表示的表达力，准确地将抽象的文本描述映射到具体图像是至关重要的。为了解决这个问题，我们提出了一个新的框架，通过探索句子中的文字的力量，释放了文本到图像人物检索中的文字想象力。具体来说，该框架使用预训练的全面CLIP模型作为图像和文本的双编码器，利用先前的跨模态对齐知识。

    The goal of Text-to-image person retrieval is to retrieve person images from a large gallery that match the given textual descriptions. The main challenge of this task lies in the significant differences in information representation between the visual and textual modalities. The textual modality conveys abstract and precise information through vocabulary and grammatical structures, while the visual modality conveys concrete and intuitive information through images. To fully leverage the expressive power of textual representations, it is essential to accurately map abstract textual descriptions to specific images.  To address this issue, we propose a novel framework to Unleash the Imagination of Text (UIT) in text-to-image person retrieval, aiming to fully explore the power of words in sentences. Specifically, the framework employs the pre-trained full CLIP model as a dual encoder for the images and texts , taking advantage of prior cross-modal alignment knowledge. The Text-guided Imag
    
[^5]: PastNet：引入物理归纳偏差用于时空视频预测

    PastNet: Introducing Physical Inductive Biases for Spatio-temporal Video Prediction. (arXiv:2305.11421v1 [cs.CV])

    [http://arxiv.org/abs/2305.11421](http://arxiv.org/abs/2305.11421)

    本文介绍了一种名为PastNet的新颖方法，通过在傅里叶域中引入谱卷积算子，利用内在的物理知识生成高质量的时空视频预测，并通过离散化局部特征降低计算成本。

    

    本文研究了时空视频预测的挑战，其中涉及根据历史数据流生成未来视频。现有方法通常利用语义地图等外部信息增强视频预测，但常常忽视视频内固有的物理知识。此外，它们的高计算需求可能会阻碍对高分辨率视频的应用。为解决这些限制，我们引入了一种新颖的方法，称为物理辅助时空网络（PastNet），用于生成高质量的视频预测。我们的PastNet核心在于在傅里叶域中引入谱卷积算子，从而有效地引入基本物理定律的归纳偏差。此外，我们使用一个内在维度估计的存储器库，在处理复杂的时空信号时离散化局部特征，从而降低计算成本。

    In this paper, we investigate the challenge of spatio-temporal video prediction, which involves generating future videos based on historical data streams. Existing approaches typically utilize external information such as semantic maps to enhance video prediction, which often neglect the inherent physical knowledge embedded within videos. Furthermore, their high computational demands could impede their applications for high-resolution videos. To address these constraints, we introduce a novel approach called Physics-assisted Spatio-temporal Network (PastNet) for generating high-quality video predictions. The core of our PastNet lies in incorporating a spectral convolution operator in the Fourier domain, which efficiently introduces inductive biases from the underlying physical laws. Additionally, we employ a memory bank with the estimated intrinsic dimensionality to discretize local features during the processing of complex spatio-temporal signals, thereby reducing computational costs 
    

