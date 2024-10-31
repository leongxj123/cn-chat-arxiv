# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Representation Learning via Image Captioning](https://arxiv.org/abs/2403.02506) | 通过图像字幕生成实现了有效的差分隐私表示学习，获得了高质量图像特征，可用于各种视觉和视觉语言任务。 |
| [^2] | [IMITATE: Clinical Prior Guided Hierarchical Vision-Language Pre-training.](http://arxiv.org/abs/2310.07355) | IMITATE是一种临床先验指导的分层视觉语言预训练模型。它利用医学报告的层级结构，从胸部X射线图像中提取多级视觉特征，并与分层医学报告中的描述性和结论性文本进行对齐。 |
| [^3] | [Imprecise Label Learning: A Unified Framework for Learning with Various Imprecise Label Configurations.](http://arxiv.org/abs/2305.12715) | 本文提出了不精确标签学习（ILL）框架，利用期望最大化算法对不精确标签信息进行最大似然估计，为各种不精确标签配置问题提供了统一的解决方案。 |

# 详细

[^1]: 通过图像字幕实现差分隐私表示学习

    Differentially Private Representation Learning via Image Captioning

    [https://arxiv.org/abs/2403.02506](https://arxiv.org/abs/2403.02506)

    通过图像字幕生成实现了有效的差分隐私表示学习，获得了高质量图像特征，可用于各种视觉和视觉语言任务。

    

    差分隐私（DP）机器学习被认为是从敏感数据中训练模型同时保护隐私的黄金标准解决方案。然而，实现这一理想的一个主要障碍是其次优的隐私-准确性权衡，在DP表示学习中特别明显。具体来说，已经证明在适度的隐私预算下，大多数模型学习的表示并不比手工特征显著更好。在这项工作中，我们展示了通过图像字幕和扩展到互联网规模的多模态数据集可以实现有效的DP表示学习。通过一系列工程技巧，我们成功地使用可观的计算量从头开始训练了DP图像字幕生成器（DP-Cap）在来自LAION-2B的233M子集上，并获得了前所未有的高质量图像特征，可用于各种下游视觉和视觉语言任务。

    arXiv:2403.02506v1 Announce Type: cross  Abstract: Differentially private (DP) machine learning is considered the gold-standard solution for training a model from sensitive data while still preserving privacy. However, a major barrier to achieving this ideal is its sub-optimal privacy-accuracy trade-off, which is particularly visible in DP representation learning. Specifically, it has been shown that under modest privacy budgets, most models learn representations that are not significantly better than hand-crafted features. In this work, we show that effective DP representation learning can be done via image captioning and scaling up to internet-scale multimodal datasets. Through a series of engineering tricks, we successfully train a DP image captioner (DP-Cap) on a 233M subset of LAION-2B from scratch using a reasonable amount of computation, and obtaining unprecedented high-quality image features that can be used in a variety of downstream vision and vision-language tasks. For examp
    
[^2]: IMITATE: 临床先验指导的分层视觉语言预训练模型

    IMITATE: Clinical Prior Guided Hierarchical Vision-Language Pre-training. (arXiv:2310.07355v1 [cs.CV])

    [http://arxiv.org/abs/2310.07355](http://arxiv.org/abs/2310.07355)

    IMITATE是一种临床先验指导的分层视觉语言预训练模型。它利用医学报告的层级结构，从胸部X射线图像中提取多级视觉特征，并与分层医学报告中的描述性和结论性文本进行对齐。

    

    在医学视觉语言预训练（VLP）领域，人们致力于从临床报告和相关医学图像中提取文本和图像特征。然而，大多数现有的方法可能忽视了利用临床报告固有的层级结构的机会，这些报告通常被分为描述性内容的“发现”和结论性观察的“印象”。当前的医学VLP方法往往将报告简化为一个统一的实体或分散的标记，而没有利用这种丰富的、结构化的格式。在这项工作中，我们提出了一种新的临床先验指导的VLP框架，名为IMITATE，用于从医学报告中学习结构信息，并使用分层视觉语言对齐。该框架从胸部X射线（CXR）图像中提取多级视觉特征，并将这些特征与分层医学报告中的描述性和结论性文本分别对齐。

    In the field of medical Vision-Language Pre-training (VLP), significant efforts have been devoted to deriving text and image features from both clinical reports and associated medical images. However, most existing methods may have overlooked the opportunity in leveraging the inherent hierarchical structure of clinical reports, which are generally split into `findings' for descriptive content and `impressions' for conclusive observation. Instead of utilizing this rich, structured format, current medical VLP approaches often simplify the report into either a unified entity or fragmented tokens. In this work, we propose a novel clinical prior guided VLP framework named IMITATE to learn the structure information from medical reports with hierarchical vision-language alignment. The framework derives multi-level visual features from the chest X-ray (CXR) images and separately aligns these features with the descriptive and the conclusive text encoded in the hierarchical medical report. Furth
    
[^3]: 不精确标签学习：学习各种不精确标签配置的统一框架

    Imprecise Label Learning: A Unified Framework for Learning with Various Imprecise Label Configurations. (arXiv:2305.12715v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.12715](http://arxiv.org/abs/2305.12715)

    本文提出了不精确标签学习（ILL）框架，利用期望最大化算法对不精确标签信息进行最大似然估计，为各种不精确标签配置问题提供了统一的解决方案。

    

    本文介绍了不精确标签学习（ILL）框架，这是一种处理机器学习任务中普遍存在的各种不精确标签配置的统一方法。ILL利用期望最大化（EM）算法对不精确标签信息进行最大似然估计（MLE），将精确标签视为潜在变量。与以前试图从不精确标签信息中推断正确标签的多功能方法相比，我们的ILL框架考虑了不精确标签信息强加的所有可能标签，允许对任何不精确标签的统一解决方案。通过全面的实验结果，我们展示了ILL可以无缝地适应各种情况，包括部分标签学习、半监督学习、噪声标签学习以及这些配置的混合。值得注意的是，我们的简单方法超过了现有的处理不精确标签的技术，标志着第一个统一解决这个问题的方法。

    In this paper, we introduce the imprecise label learning (ILL) framework, a unified approach to handle various imprecise label configurations, which are commonplace challenges in machine learning tasks. ILL leverages an expectation-maximization (EM) algorithm for the maximum likelihood estimation (MLE) of the imprecise label information, treating the precise labels as latent variables. Compared to previous versatile methods attempting to infer correct labels from the imprecise label information, our ILL framework considers all possible labeling imposed by the imprecise label information, allowing a unified solution to deal with any imprecise labels. With comprehensive experimental results, we demonstrate that ILL can seamlessly adapt to various situations, including partial label learning, semi-supervised learning, noisy label learning, and a mixture of these settings. Notably, our simple method surpasses the existing techniques for handling imprecise labels, marking the first unified 
    

