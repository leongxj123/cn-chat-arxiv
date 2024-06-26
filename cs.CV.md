# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RESSA: Repair Sparse Vision-Language Models via Sparse Cross-Modality Adaptation](https://arxiv.org/abs/2404.02424) | 通过稀疏跨模态适应修复稀疏视觉-语言模型，探索了VLM修剪中的两个主要问题，提出稀疏比率对性能的影响，展示了修复稀疏VLMs性能所需的专门技术。 |
| [^2] | [MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions](https://arxiv.org/abs/2403.19651) | 本研究提出了MagicLens，一系列支持开放式指令的自监督图像检索模型，核心创新在于利用文本指令使得图像检索可以检索到比视觉相似性更丰富关系的图像。 |
| [^3] | [Large Window-based Mamba UNet for Medical Image Segmentation: Beyond Convolution and Self-attention](https://arxiv.org/abs/2403.07332) | 该论文提出了一个基于大窗口的Mamba UNet用于医学图像分割，相比传统方法，在局部空间建模方面有优势，同时在全局建模方面保持高效率。 |
| [^4] | [Universal Prompt Optimizer for Safe Text-to-Image Generation](https://arxiv.org/abs/2402.10882) | 提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。 |
| [^5] | [Revisiting Active Learning in the Era of Vision Foundation Models.](http://arxiv.org/abs/2401.14555) | 本文评估了基础视觉模型对有效主动学习的三个关键组成部分的影响，并提出了一个新的简单优雅的主动学习策略，该策略通过平衡不确定性估计和样本多样性来实现。 |
| [^6] | [SoK: Facial Deepfake Detectors.](http://arxiv.org/abs/2401.04364) | 本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。 |
| [^7] | [A Probabilistic Fluctuation based Membership Inference Attack for Generative Models.](http://arxiv.org/abs/2308.12143) | 本研究针对生成模型提出了一种概率波动评估成员推断攻击方法(PFAMI)，通过检测概率分布的波动性来推断模型中是否存在某条训练记录的成员身份。 |

# 详细

[^1]: 通过稀疏跨模态适应修复稀疏视觉-语言模型

    RESSA: Repair Sparse Vision-Language Models via Sparse Cross-Modality Adaptation

    [https://arxiv.org/abs/2404.02424](https://arxiv.org/abs/2404.02424)

    通过稀疏跨模态适应修复稀疏视觉-语言模型，探索了VLM修剪中的两个主要问题，提出稀疏比率对性能的影响，展示了修复稀疏VLMs性能所需的专门技术。

    

    视觉-语言模型(VLMs)整合了来自多个模态的不同信息，在各种任务中表现出显著成功。但是，在资源受限的场景中部署包括大规模视觉和语言模型在内的VLMs会带来挑战。尽管修剪后微调提供了一种保持更小模型大小性能的潜在解决方案，但其在VLMs中的应用相对未被探索，这提出了两个主要问题：如何在不同模态特定模型之间分配稀疏性，以及如何修复被修剪稀疏的VLMs的性能。为了回答第一个问题，我们进行了关于VLM修剪的初步研究，发现使用相同稀疏比率修剪视觉模型和语言模型有助于实现接近最佳性能。对于第二个问题，与微调单模稀疏模型不同，稀疏VLMs涉及跨模态交互，需要专门的技术。

    arXiv:2404.02424v1 Announce Type: new  Abstract: Vision-Language Models (VLMs), integrating diverse information from multiple modalities, have shown remarkable success across various tasks. However, deploying VLMs, comprising large-scale vision and language models poses challenges in resource-constrained scenarios. While pruning followed by finetuning offers a potential solution to maintain performance with smaller model sizes, its application to VLMs remains relatively unexplored, presenting two main questions: how to distribute sparsity across different modality-specific models, and how to repair the performance of pruned sparse VLMs. To answer the first question, we conducted preliminary studies on VLM pruning and found that pruning vision models and language models with the same sparsity ratios contribute to nearly optimal performance. For the second question, unlike finetuning unimodal sparse models, sparse VLMs involve cross-modality interactions, requiring specialized techniques
    
[^2]: MagicLens：自监督图像检索与开放式指令

    MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions

    [https://arxiv.org/abs/2403.19651](https://arxiv.org/abs/2403.19651)

    本研究提出了MagicLens，一系列支持开放式指令的自监督图像检索模型，核心创新在于利用文本指令使得图像检索可以检索到比视觉相似性更丰富关系的图像。

    

    图像检索，即根据参考图像查找所需图像，固有地包含难以仅使用基于图像的度量捕捉到的丰富、多方面的搜索意图。最近的工作利用文本指令允许用户更自由地表达他们的搜索意图。然而，现有工作主要集中在那些视觉上相似和/或可以用一小组预定义关系来表征的图像对上。本文的核心论点是文本指令可以使图像检索能够检索到比视觉相似性更丰富关系的图像。为了证明这一点，我们引入了MagicLens，一系列支持开放式指令的自监督图像检索模型。MagicLens建立在一个重要的新颖见解上：自然发生在同一网页上的图像对包含着大量隐式关系（例如，内部视图），我们可以通过综合指令将这些隐式关系变为显式。

    arXiv:2403.19651v1 Announce Type: cross  Abstract: Image retrieval, i.e., finding desired images given a reference image, inherently encompasses rich, multi-faceted search intents that are difficult to capture solely using image-based measures. Recent work leverages text instructions to allow users to more freely express their search intents. However, existing work primarily focuses on image pairs that are visually similar and/or can be characterized by a small set of pre-defined relations. The core thesis of this paper is that text instructions can enable retrieving images with richer relations beyond visual similarity. To show this, we introduce MagicLens, a series of self-supervised image retrieval models that support open-ended instructions. MagicLens is built on a key novel insight: image pairs that naturally occur on the same web pages contain a wide range of implicit relations (e.g., inside view of), and we can bring those implicit relations explicit by synthesizing instructions
    
[^3]: 基于大窗口的Mamba UNet用于医学图像分割：超越卷积和自注意力

    Large Window-based Mamba UNet for Medical Image Segmentation: Beyond Convolution and Self-attention

    [https://arxiv.org/abs/2403.07332](https://arxiv.org/abs/2403.07332)

    该论文提出了一个基于大窗口的Mamba UNet用于医学图像分割，相比传统方法，在局部空间建模方面有优势，同时在全局建模方面保持高效率。

    

    在临床实践中，医学图像分割提供了有关目标器官或组织轮廓和尺寸的有用信息，有助于改进诊断、分析和治疗。最近几年，卷积神经网络（CNN）和Transformer在这一领域占据主导地位，但它们仍然存在一定问题，如有限的感知范围或昂贵的远程建模。Mamba，作为一种具有线性复杂度的长程依赖性建模的状态空间序列模型（SSM），最近出现为一种有前途的范式。在本文中，我们介绍了一种用于2D和3D医学图像分割的基于大窗口的Mamba U-形网络，即LMa-UNet。我们LMa-UNet的一个突出特点是利用大窗口，在局部空间建模方面优于基于小核的CNN和基于小窗口的Transformer，同时与具有二次复杂度的自注意力相比，在全局建模方面保持卓越的效率。

    arXiv:2403.07332v1 Announce Type: cross  Abstract: In clinical practice, medical image segmentation provides useful information on the contours and dimensions of target organs or tissues, facilitating improved diagnosis, analysis, and treatment. In the past few years, convolutional neural networks (CNNs) and Transformers have dominated this area, but they still suffer from either limited receptive fields or costly long-range modeling. Mamba, a State Space Sequence Model (SSM), recently emerged as a promising paradigm for long-range dependency modeling with linear complexity. In this paper, we introduce a Large Window-based Mamba U}-shape Network, or LMa-UNet, for 2D and 3D medical image segmentation. A distinguishing feature of our LMa-UNet is its utilization of large windows, excelling in locally spatial modeling compared to small kernel-based CNNs and small window-based Transformers, while maintaining superior efficiency in global modeling compared to self-attention with quadratic co
    
[^4]: 通用提示优化器用于安全文本到图像生成

    Universal Prompt Optimizer for Safe Text-to-Image Generation

    [https://arxiv.org/abs/2402.10882](https://arxiv.org/abs/2402.10882)

    提出了第一个通用提示优化器，用于在黑盒场景中安全生成文本到图像，通过构建毒素-清洁提示对数据集，设计奖励函数，并通过 Proximal Policy Optimization 训练优化器，成功降低各种 T2I 模型生成不安全内容的可能性。

    

    文本到图像（T2I）模型在根据文字提示生成图像方面表现出色。然而，这些模型容易受到不安全输入的影响，从而生成不安全内容，如色情、骚扰和非法活动图像。基于图像检查器、模型微调和嵌入式阻止的现有研究在真实世界应用中不可行。因此，我们提出了第一个用于黑盒场景中安全 T2I 生成的通用提示优化器。

    arXiv:2402.10882v1 Announce Type: cross  Abstract: Text-to-Image (T2I) models have shown great performance in generating images based on textual prompts. However, these models are vulnerable to unsafe input to generate unsafe content like sexual, harassment and illegal-activity images. Existing studies based on image checker, model fine-tuning and embedding blocking are impractical in real-world applications. Hence, \textit{we propose the first universal prompt optimizer for safe T2I generation in black-box scenario}. We first construct a dataset consisting of toxic-clean prompt pairs by GPT-3.5 Turbo. To guide the optimizer to have the ability of converting toxic prompt to clean prompt while preserving semantic information, we design a novel reward function measuring toxicity and text alignment of generated images and train the optimizer through Proximal Policy Optimization. Experiments show that our approach can effectively reduce the likelihood of various T2I models in generating in
    
[^5]: 在视觉基础模型时代重新审视主动学习

    Revisiting Active Learning in the Era of Vision Foundation Models. (arXiv:2401.14555v1 [cs.CV])

    [http://arxiv.org/abs/2401.14555](http://arxiv.org/abs/2401.14555)

    本文评估了基础视觉模型对有效主动学习的三个关键组成部分的影响，并提出了一个新的简单优雅的主动学习策略，该策略通过平衡不确定性估计和样本多样性来实现。

    

    基础视觉或视觉-语言模型是在大规模无标签或噪声数据上训练的，并学习到可以在各种任务上实现令人印象深刻的零标注或少标注性能的鲁棒表示。鉴于这些特性，它们是主动学习（AL）的自然选择，旨在实现标记效率的最大化，但在低预算条件下，基础模型的全部潜力在AL环境中尚未得到探索。在这项工作中，我们评估了基础模型对有效AL的三个关键组成部分的影响，即1）初始标记样本池的选择，2）确保多样性抽样，以及3）代表性和不确定性抽样之间的权衡。我们系统地研究了基础模型（DINOv2、OpenCLIP）的鲁棒表示如何挑战已有的主动学习结果。我们的观察结果为一个新的简单优雅的AL策略的有原则构建提供了指导，该策略通过使用dropout估计不确定性和样本多样性之间的平衡。

    Foundation vision or vision-language models are trained on large unlabeled or noisy data and learn robust representations that can achieve impressive zeroor few-shot performance on diverse tasks. Given these properties, they are a natural fit for active learning (AL), which aims to maximize labeling efficiency, but the full potential of foundation models has not been explored in the context of AL, specifically in the low-budget regime. In this work, we evaluate how foundation models influence three critical components of effective AL, namely, 1) initial labeled pool selection, 2) ensuring diverse sampling, and 3) the trade-off between representative and uncertainty sampling. We systematically study how the robust representations of foundation models (DINOv2, OpenCLIP) challenge existing findings in active learning. Our observations inform the principled construction of a new simple and elegant AL strategy that balances uncertainty estimated via dropout with sample diversity. We exten
    
[^6]: SoK：面部深度伪造检测器

    SoK: Facial Deepfake Detectors. (arXiv:2401.04364v1 [cs.CV])

    [http://arxiv.org/abs/2401.04364](http://arxiv.org/abs/2401.04364)

    本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。

    

    深度伪造技术迅速成为对社会构成深远和严重威胁的原因之一，主要由于其易于制作和传播。这种情况加速了深度伪造检测技术的发展。然而，许多现有的检测器在验证时 heavily 依赖实验室生成的数据集，这可能无法有效地让它们应对新颖、新兴和实际的深度伪造技术。本文对最新的深度伪造检测器进行广泛全面的回顾和分析，根据几个关键标准对它们进行评估。这些标准将这些检测器分为 4 个高级组别和 13 个细粒度子组别，都遵循一个统一的标准概念框架。这种分类和框架提供了对影响检测器功效的因素的深入和实用的见解。我们对 16 个主要的检测器在各种标准的攻击场景中的普适性进行评估，包括黑盒攻击场景。

    Deepfakes have rapidly emerged as a profound and serious threat to society, primarily due to their ease of creation and dissemination. This situation has triggered an accelerated development of deepfake detection technologies. However, many existing detectors rely heavily on lab-generated datasets for validation, which may not effectively prepare them for novel, emerging, and real-world deepfake techniques. In this paper, we conduct an extensive and comprehensive review and analysis of the latest state-of-the-art deepfake detectors, evaluating them against several critical criteria. These criteria facilitate the categorization of these detectors into 4 high-level groups and 13 fine-grained sub-groups, all aligned with a unified standard conceptual framework. This classification and framework offer deep and practical insights into the factors that affect detector efficacy. We assess the generalizability of 16 leading detectors across various standard attack scenarios, including black-bo
    
[^7]: 一种基于概率波动的生成模型成员推断攻击方法

    A Probabilistic Fluctuation based Membership Inference Attack for Generative Models. (arXiv:2308.12143v1 [cs.LG])

    [http://arxiv.org/abs/2308.12143](http://arxiv.org/abs/2308.12143)

    本研究针对生成模型提出了一种概率波动评估成员推断攻击方法(PFAMI)，通过检测概率分布的波动性来推断模型中是否存在某条训练记录的成员身份。

    

    成员推断攻击(MIA)通过查询模型来识别机器学习模型的训练集中是否存在某条记录。对经典分类模型的MIA已有很多研究，最近的工作开始探索如何将MIA应用到生成模型上。我们的研究表明，现有的面向生成模型的MIA主要依赖于目标模型的过拟合现象。然而，过拟合可以通过采用各种正则化技术来避免，而现有的MIA在实践中表现不佳。与过拟合不同，记忆对于深度学习模型实现最佳性能是至关重要的，使其成为一种更为普遍的现象。生成模型中的记忆导致生成记录的概率分布呈现出增长的趋势。因此，我们提出了一种基于概率波动的成员推断攻击方法(PFAMI)，它是一种黑盒MIA，通过检测概率波动来推断成员身份。

    Membership Inference Attack (MIA) identifies whether a record exists in a machine learning model's training set by querying the model. MIAs on the classic classification models have been well-studied, and recent works have started to explore how to transplant MIA onto generative models. Our investigation indicates that existing MIAs designed for generative models mainly depend on the overfitting in target models. However, overfitting can be avoided by employing various regularization techniques, whereas existing MIAs demonstrate poor performance in practice. Unlike overfitting, memorization is essential for deep learning models to attain optimal performance, making it a more prevalent phenomenon. Memorization in generative models leads to an increasing trend in the probability distribution of generating records around the member record. Therefore, we propose a Probabilistic Fluctuation Assessing Membership Inference Attack (PFAMI), a black-box MIA that infers memberships by detecting t
    

