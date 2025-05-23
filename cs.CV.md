# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Demystifying Variational Diffusion Models.](http://arxiv.org/abs/2401.06281) | 该论文通过使用有向图模型和变分贝叶斯原理，揭示了变分扩散模型的原理和连接，为非专业统计物理领域的读者提供了更容易理解的介绍。 |
| [^2] | [Fast Inference Through The Reuse Of Attention Maps In Diffusion Models.](http://arxiv.org/abs/2401.01008) | 本文提出了一种无需训练的方法，通过重用注意力映射来实现Text-to-image diffusion models中的快速推理，以提高效率。 |
| [^3] | [Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination.](http://arxiv.org/abs/2311.02960) | 本文通过研究中间特征的结构，揭示了深度网络在层级特征学习过程中的演化模式。研究发现线性层在特征学习中起到了与深层非线性网络类似的作用。 |
| [^4] | [SegMatch: A semi-supervised learning method for surgical instrument segmentation.](http://arxiv.org/abs/2308.05232) | SegMatch是一种用于手术器械分割的半监督学习方法，通过结合一致性正则化和伪标签来减少昂贵的注释需求，并通过弱增强和生成伪标签来实现无监督损失的施加。 |
| [^5] | [Diagnostic test accuracy (DTA) of artificial intelligence in digital pathology: a systematic review, meta-analysis and quality assessment.](http://arxiv.org/abs/2306.07999) | 本文进行了数字病理图像中应用人工智能的所有病理学领域的诊断准确度的系统综述和Meta分析。结果表明，人工智能在数字病理学中取得了高度的准确度，是可行的辅助诊断工具。 |

# 详细

[^1]: 揭秘变分扩散模型

    Demystifying Variational Diffusion Models. (arXiv:2401.06281v1 [cs.LG])

    [http://arxiv.org/abs/2401.06281](http://arxiv.org/abs/2401.06281)

    该论文通过使用有向图模型和变分贝叶斯原理，揭示了变分扩散模型的原理和连接，为非专业统计物理领域的读者提供了更容易理解的介绍。

    

    尽管扩散模型越来越受欢迎，但对于非平衡统计物理领域的初学者来说，对该模型类的深入理解仍然有些困难。考虑到这一点，我们通过使用有向图模型和变分贝叶斯原理，提供了一个我们认为更简单易懂的扩散模型介绍，这对于一般读者来说需要的先决条件相对较少。我们的阐述构成了一个全面的技术综述，从深度潜变量模型等基本概念到连续时间扩散模型的最新进展，突出了模型类之间的理论联系。我们尽可能地提供了在初始工作中被省略的额外数学洞察，以帮助理解，同时避免引入新的符号表示。我们希望这篇文章对于该领域的研究人员和实践者来说，能作为一个有用的教育补充材料。

    Despite the growing popularity of diffusion models, gaining a deep understanding of the model class remains somewhat elusive for the uninitiated in non-equilibrium statistical physics. With that in mind, we present what we believe is a more straightforward introduction to diffusion models using directed graphical modelling and variational Bayesian principles, which imposes relatively fewer prerequisites on the average reader. Our exposition constitutes a comprehensive technical review spanning from foundational concepts like deep latent variable models to recent advances in continuous-time diffusion-based modelling, highlighting theoretical connections between model classes along the way. We provide additional mathematical insights that were omitted in the seminal works whenever possible to aid in understanding, while avoiding the introduction of new notation. We envision this article serving as a useful educational supplement for both researchers and practitioners in the area, and we 
    
[^2]: Text-to-image diffusion models中通过重用注意力映射实现快速推理

    Fast Inference Through The Reuse Of Attention Maps In Diffusion Models. (arXiv:2401.01008v1 [cs.CV])

    [http://arxiv.org/abs/2401.01008](http://arxiv.org/abs/2401.01008)

    本文提出了一种无需训练的方法，通过重用注意力映射来实现Text-to-image diffusion models中的快速推理，以提高效率。

    

    文字到图像扩散模型在灵活和逼真的图像合成方面展示了前所未有的能力。然而，生成单个图像所需的迭代过程既昂贵又具有较高的延迟，促使研究人员进一步研究其效率。我们提出了一种无需调整采样步长的无需训练的方法。具体地说，我们发现重复计算注意力映射既耗时又冗余，因此我们建议在采样过程中结构化地重用注意力映射。我们的初步重用策略受到初级ODE理论的启发，该理论认为在采样过程的后期重用最合适。在注意到这种理论方法的一些局限性后，我们通过实验证明了一种更好的方法。

    Text-to-image diffusion models have demonstrated unprecedented abilities at flexible and realistic image synthesis. However, the iterative process required to produce a single image is costly and incurs a high latency, prompting researchers to further investigate its efficiency. Typically, improvements in latency have been achieved in two ways: (1) training smaller models through knowledge distillation (KD); and (2) adopting techniques from ODE-theory to facilitate larger step sizes. In contrast, we propose a training-free approach that does not alter the step-size of the sampler. Specifically, we find the repeated calculation of attention maps to be both costly and redundant; therefore, we propose a structured reuse of attention maps during sampling. Our initial reuse policy is motivated by rudimentary ODE-theory, which suggests that reuse is most suitable late in the sampling procedure. After noting a number of limitations in this theoretical approach, we empirically search for a bet
    
[^3]: 通过层间特征压缩和差别性学习理解深度表示学习

    Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination. (arXiv:2311.02960v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.02960](http://arxiv.org/abs/2311.02960)

    本文通过研究中间特征的结构，揭示了深度网络在层级特征学习过程中的演化模式。研究发现线性层在特征学习中起到了与深层非线性网络类似的作用。

    

    在过去的十年中，深度学习已经证明是从原始数据中学习有意义特征的一种高效工具。然而，深度网络如何在不同层级上进行等级特征学习仍然是一个开放问题。在这项工作中，我们试图通过研究中间特征的结构揭示这个谜团。受到我们实证发现的线性层在特征学习中模仿非线性网络中深层的角色的启发，我们研究了深度线性网络如何将输入数据转化为输出，通过研究训练后的每个层的输出（即特征）在多类分类问题的背景下。为了实现这个目标，我们首先定义了衡量中间特征的类内压缩和类间差别性的度量标准。通过对这两个度量标准的理论分析，我们展示了特征从浅层到深层的演变遵循着一种简单而量化的模式，前提是输入数据是

    Over the past decade, deep learning has proven to be a highly effective tool for learning meaningful features from raw data. However, it remains an open question how deep networks perform hierarchical feature learning across layers. In this work, we attempt to unveil this mystery by investigating the structures of intermediate features. Motivated by our empirical findings that linear layers mimic the roles of deep layers in nonlinear networks for feature learning, we explore how deep linear networks transform input data into output by investigating the output (i.e., features) of each layer after training in the context of multi-class classification problems. Toward this goal, we first define metrics to measure within-class compression and between-class discrimination of intermediate features, respectively. Through theoretical analysis of these two metrics, we show that the evolution of features follows a simple and quantitative pattern from shallow to deep layers when the input data is
    
[^4]: SegMatch: 一种用于手术器械分割的半监督学习方法

    SegMatch: A semi-supervised learning method for surgical instrument segmentation. (arXiv:2308.05232v1 [cs.CV])

    [http://arxiv.org/abs/2308.05232](http://arxiv.org/abs/2308.05232)

    SegMatch是一种用于手术器械分割的半监督学习方法，通过结合一致性正则化和伪标签来减少昂贵的注释需求，并通过弱增强和生成伪标签来实现无监督损失的施加。

    

    手术器械分割被认为是提供先进手术辅助和改善计算机辅助干预的关键手段。在这项工作中，我们提出了SegMatch，一种用于减少昂贵注释对腹腔镜和机器人手术图像的需求的半监督学习方法。SegMatch基于FixMatch，一种广泛采用一致性正则化和伪标签的半监督分类流程，并将其调整为分割任务。在我们提出的SegMatch中，未标记的图像进行弱增强，并通过分割模型生成伪标签，以对高置信度像素上的对抗增强图像的模型输出施加无监督损失。我们针对分割任务的调整还包括仔细考虑所依赖的增强函数的等变性和不变性属性，为增强的相关性增加。

    Surgical instrument segmentation is recognised as a key enabler to provide advanced surgical assistance and improve computer assisted interventions. In this work, we propose SegMatch, a semi supervised learning method to reduce the need for expensive annotation for laparoscopic and robotic surgical images. SegMatch builds on FixMatch, a widespread semi supervised classification pipeline combining consistency regularization and pseudo labelling, and adapts it for the purpose of segmentation. In our proposed SegMatch, the unlabelled images are weakly augmented and fed into the segmentation model to generate a pseudo-label to enforce the unsupervised loss against the output of the model for the adversarial augmented image on the pixels with a high confidence score. Our adaptation for segmentation tasks includes carefully considering the equivariance and invariance properties of the augmentation functions we rely on. To increase the relevance of our augmentations, we depart from using only
    
[^5]: 数字病理学中人工智能的诊断测试准确度：系统综述、Meta分析和质量评估

    Diagnostic test accuracy (DTA) of artificial intelligence in digital pathology: a systematic review, meta-analysis and quality assessment. (arXiv:2306.07999v1 [physics.med-ph])

    [http://arxiv.org/abs/2306.07999](http://arxiv.org/abs/2306.07999)

    本文进行了数字病理图像中应用人工智能的所有病理学领域的诊断准确度的系统综述和Meta分析。结果表明，人工智能在数字病理学中取得了高度的准确度，是可行的辅助诊断工具。

    

    确保临床使用之前AI模型的诊断表现是关键，以确保这些技术的安全和成功的采用。近年来，报道应用于数字病理学图像进行诊断目的的AI研究数量迅速增加。本研究旨在提供数字病理学中AI的诊断准确度的概述，涵盖了所有病理学领域。这项系统性综述和Meta分析包括使用任何类型的人工智能应用于任何疾病类型的WSI图像的诊断准确性研究。参考标准是通过组织病理学评估和/或免疫组化诊断。搜索在2022年6月在PubMed、EMBASE和CENTRAL中进行。在2976项研究中，有100项纳入综述，48项纳入完整的Meta分析。使用QUADAS-2工具评估了偏倚风险和适用性的关注点。数据提取由两个调查员进行，并进行了Meta分析。

    Ensuring diagnostic performance of AI models before clinical use is key to the safe and successful adoption of these technologies. Studies reporting AI applied to digital pathology images for diagnostic purposes have rapidly increased in number in recent years. The aim of this work is to provide an overview of the diagnostic accuracy of AI in digital pathology images from all areas of pathology. This systematic review and meta-analysis included diagnostic accuracy studies using any type of artificial intelligence applied to whole slide images (WSIs) in any disease type. The reference standard was diagnosis through histopathological assessment and / or immunohistochemistry. Searches were conducted in PubMed, EMBASE and CENTRAL in June 2022. We identified 2976 studies, of which 100 were included in the review and 48 in the full meta-analysis. Risk of bias and concerns of applicability were assessed using the QUADAS-2 tool. Data extraction was conducted by two investigators and meta-analy
    

