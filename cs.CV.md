# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalized Consistency Trajectory Models for Image Manipulation](https://arxiv.org/abs/2403.12510) | 本研究提出了广义一致性轨迹模型（GCTMs），能够在任何噪声分布和数据分布之间实现转换。 |
| [^2] | [RACR-MIL: Weakly Supervised Skin Cancer Grading using Rank-Aware Contextual Reasoning on Whole Slide Images.](http://arxiv.org/abs/2308.15618) | RACR-MIL是一个自动化的弱监督的皮肤癌分级方法，可以使用整张切片图像进行训练，无需细粒度的肿瘤注释，通过在切片图像中的瓦片章节中使用注意力机制的多实例学习，可以为切片图像分配分级。该方法主要创新包括使用空间和语义接近性定义切片图像图像以编码肿瘤区域的局部和非局部依赖关系以及使用序数排名约束保证注意力网络的性能 |
| [^3] | [PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing.](http://arxiv.org/abs/2306.16894) | PFB-Diff 是一个通过渐进特征混合的方法，用于文本驱动的图像编辑。该方法解决了扩散模型在像素级混合中产生的伪影问题，并通过多级特征混合和注意力屏蔽机制确保了编辑图像的语义连贯性和高质量。 |
| [^4] | [End-to-End Augmentation Hyperparameter Tuning for Self-Supervised Anomaly Detection.](http://arxiv.org/abs/2306.12033) | 这项研究提出了一种名为ST-SSAD的新方法，可以系统地调整数据增强的超参数，从而有助于提高自我监督异常检测（SSAD）的性能。 |
| [^5] | [CBAGAN-RRT: Convolutional Block Attention Generative Adversarial Network for Sampling-Based Path Planning.](http://arxiv.org/abs/2305.10442) | 本文介绍了一种基于图像处理学习算法（CBAGAN-RRT）的路径规划方法，使用卷积块注意力生成对抗网络和一种新的损失函数，找到更优的最佳路径并提高算法的收敛速度，与先前的最先进算法相比，在图像质量生成指标和路径规划指标方面都表现更优。 |

# 详细

[^1]: 图像操作的广义一致性轨迹模型

    Generalized Consistency Trajectory Models for Image Manipulation

    [https://arxiv.org/abs/2403.12510](https://arxiv.org/abs/2403.12510)

    本研究提出了广义一致性轨迹模型（GCTMs），能够在任何噪声分布和数据分布之间实现转换。

    

    基于扩散的生成模型在无条件生成以及图像编辑和恢复等应用任务中表现出色。扩散模型的成功在于扩散的迭代性质：扩散将将噪声到数据的复杂映射过程分解为一系列简单的去噪任务。此外，通过在每个去噪步骤中注入引导项，我们能够对生成过程进行精细控制。然而，迭代过程也常常计算密集，通常需要进行数十次甚至数千次函数评估。虽然一致性轨迹模型（CTMs）可以在概率流ODE（PFODE）上任意时间点之间进行遍历，并且通过单次函数评估进行得分推导，但CTMs仅允许从高斯噪声转换为数据。因此，本文旨在通过提出广义CTMs（GCTMs）来发挥CTMs的全部潜力，实现在任何噪声分布和数据分布之间进行转换。

    arXiv:2403.12510v1 Announce Type: cross  Abstract: Diffusion-based generative models excel in unconditional generation, as well as on applied tasks such as image editing and restoration. The success of diffusion models lies in the iterative nature of diffusion: diffusion breaks down the complex process of mapping noise to data into a sequence of simple denoising tasks. Moreover, we are able to exert fine-grained control over the generation process by injecting guidance terms into each denoising step. However, the iterative process is also computationally intensive, often taking from tens up to thousands of function evaluations. Although consistency trajectory models (CTMs) enable traversal between any time points along the probability flow ODE (PFODE) and score inference with a single function evaluation, CTMs only allow translation from Gaussian noise to data. Thus, this work aims to unlock the full potential of CTMs by proposing generalized CTMs (GCTMs), which translate between arbit
    
[^2]: RACR-MIL：使用基于排名感知背景推理的整张切片图像进行弱监督的皮肤癌分级

    RACR-MIL: Weakly Supervised Skin Cancer Grading using Rank-Aware Contextual Reasoning on Whole Slide Images. (arXiv:2308.15618v1 [cs.CV])

    [http://arxiv.org/abs/2308.15618](http://arxiv.org/abs/2308.15618)

    RACR-MIL是一个自动化的弱监督的皮肤癌分级方法，可以使用整张切片图像进行训练，无需细粒度的肿瘤注释，通过在切片图像中的瓦片章节中使用注意力机制的多实例学习，可以为切片图像分配分级。该方法主要创新包括使用空间和语义接近性定义切片图像图像以编码肿瘤区域的局部和非局部依赖关系以及使用序数排名约束保证注意力网络的性能

    

    Cutaneous squamous cell cancer (cSCC) is the second most common skin cancer in the US. It is diagnosed by manual multi-class tumor grading using a tissue whole slide image (WSI), which is subjective and suffers from inter-pathologist variability. We propose an automated weakly-supervised grading approach for cSCC WSIs that is trained using WSI-level grade and does not require fine-grained tumor annotations. The proposed model, RACR-MIL, transforms each WSI into a bag of tiled patches and leverages attention-based multiple-instance learning to assign a WSI-level grade. We propose three key innovations to address general as well as cSCC-specific challenges in tumor grading. First, we leverage spatial and semantic proximity to define a WSI graph that encodes both local and non-local dependencies between tumor regions and leverage graph attention convolution to derive contextual patch features. Second, we introduce a novel ordinal ranking constraint on the patch attention network to ensure

    Cutaneous squamous cell cancer (cSCC) is the second most common skin cancer in the US. It is diagnosed by manual multi-class tumor grading using a tissue whole slide image (WSI), which is subjective and suffers from inter-pathologist variability. We propose an automated weakly-supervised grading approach for cSCC WSIs that is trained using WSI-level grade and does not require fine-grained tumor annotations. The proposed model, RACR-MIL, transforms each WSI into a bag of tiled patches and leverages attention-based multiple-instance learning to assign a WSI-level grade. We propose three key innovations to address general as well as cSCC-specific challenges in tumor grading. First, we leverage spatial and semantic proximity to define a WSI graph that encodes both local and non-local dependencies between tumor regions and leverage graph attention convolution to derive contextual patch features. Second, we introduce a novel ordinal ranking constraint on the patch attention network to ensure
    
[^3]: PFB-Diff: 渐进特征混合扩散用于文本驱动的图像编辑

    PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing. (arXiv:2306.16894v1 [cs.CV])

    [http://arxiv.org/abs/2306.16894](http://arxiv.org/abs/2306.16894)

    PFB-Diff 是一个通过渐进特征混合的方法，用于文本驱动的图像编辑。该方法解决了扩散模型在像素级混合中产生的伪影问题，并通过多级特征混合和注意力屏蔽机制确保了编辑图像的语义连贯性和高质量。

    

    扩散模型展示了其合成多样性和高质量图像的卓越能力，引起了人们对将其应用于实际图像编辑的兴趣。然而，现有的基于扩散的局部图像编辑方法常常因为目标图像和扩散潜在变量的像素级混合而产生不期望的伪影，缺乏维持图像一致性所必需的语义。为了解决这些问题，我们提出了PFB-Diff，一种逐步特征混合的方法，用于基于扩散的图像编辑。与以往方法不同，PFB-Diff通过多级特征混合将文本引导生成的内容与目标图像无缝集成在一起。深层特征中编码的丰富语义和从高到低级别的渐进混合方案确保了编辑图像的语义连贯性和高质量。此外，我们在交叉注意力层中引入了一个注意力屏蔽机制，以限制特定词语对编辑图像的影响。

    Diffusion models have showcased their remarkable capability to synthesize diverse and high-quality images, sparking interest in their application for real image editing. However, existing diffusion-based approaches for local image editing often suffer from undesired artifacts due to the pixel-level blending of the noised target images and diffusion latent variables, which lack the necessary semantics for maintaining image consistency. To address these issues, we propose PFB-Diff, a Progressive Feature Blending method for Diffusion-based image editing. Unlike previous methods, PFB-Diff seamlessly integrates text-guided generated content into the target image through multi-level feature blending. The rich semantics encoded in deep features and the progressive blending scheme from high to low levels ensure semantic coherence and high quality in edited images. Additionally, we introduce an attention masking mechanism in the cross-attention layers to confine the impact of specific words to 
    
[^4]: 自我监督异常检测的端到端增强超参数调整

    End-to-End Augmentation Hyperparameter Tuning for Self-Supervised Anomaly Detection. (arXiv:2306.12033v1 [cs.LG])

    [http://arxiv.org/abs/2306.12033](http://arxiv.org/abs/2306.12033)

    这项研究提出了一种名为ST-SSAD的新方法，可以系统地调整数据增强的超参数，从而有助于提高自我监督异常检测（SSAD）的性能。

    

    自我监督学习（SSL）已经成为一个有前途的范例，它为现实问题提供自产生的监督信号，避免了繁琐的手动标注工作。SSL对于无监督任务，如异常检测尤其具有吸引力，因为标记的异常通常不存在或难以获得。虽然自我监督异常检测（SSAD）近年来受到了广泛关注，但文献却未将数据增强视为超参数。同时，最近的研究表明，增强选择对检测性能有重要影响。在本文中，我们介绍了ST-SSAD（自我调整自我监督异常检测），这是一种关于严格调整增强的SSAD的第一个系统方法。为此，我们的工作提出了两个关键贡献。第一是一种新的无监督验证损失函数，量化增强训练数据与（无标签）测试数据之间的对齐程度。在原则上，我们采用了最近高效的有监督学习方法借鉴的无监督验证方案和增强数据搜索策略，并将其适应于SSAD。我们进一步提出了一种新的增强搜索方法，通过贝叶斯优化的形式，将轻量级数据增强搜索器的简单集成。在各种异常检测基准数据集上的实验表明，我们的增强调整方法相对于以前的最新结果可以获得一致的性能提升，并且相对于最近的有监督方法具有竞争性的结果。

    Self-supervised learning (SSL) has emerged as a promising paradigm that presents self-generated supervisory signals to real-world problems, bypassing the extensive manual labeling burden. SSL is especially attractive for unsupervised tasks such as anomaly detection, where labeled anomalies are often nonexistent and costly to obtain. While self-supervised anomaly detection (SSAD) has seen a recent surge of interest, the literature has failed to treat data augmentation as a hyperparameter. Meanwhile, recent works have reported that the choice of augmentation has significant impact on detection performance. In this paper, we introduce ST-SSAD (Self-Tuning Self-Supervised Anomaly Detection), the first systematic approach to SSAD in regards to rigorously tuning augmentation. To this end, our work presents two key contributions. The first is a new unsupervised validation loss that quantifies the alignment between the augmented training data and the (unlabeled) test data. In principle we adop
    
[^5]: CBAGAN-RRT: 卷积块注意力生成对抗网络用于基于采样的路径规划

    CBAGAN-RRT: Convolutional Block Attention Generative Adversarial Network for Sampling-Based Path Planning. (arXiv:2305.10442v1 [cs.RO])

    [http://arxiv.org/abs/2305.10442](http://arxiv.org/abs/2305.10442)

    本文介绍了一种基于图像处理学习算法（CBAGAN-RRT）的路径规划方法，使用卷积块注意力生成对抗网络和一种新的损失函数，找到更优的最佳路径并提高算法的收敛速度，与先前的最先进算法相比，在图像质量生成指标和路径规划指标方面都表现更优。

    

    基于采样的路径规划算法在自主机器人中发挥着重要作用。但是，基于RRT算法的一个常见问题是生成的初始路径不是最优的，而且收敛速度过慢，无法应用于实际场景。本文提出了一种使用卷积块注意力生成对抗网络和一种新的损失函数的图像处理学习算法（CBAGAN-RRT），以设计启发式算法，找到更优的最佳路径，并提高算法的收敛速度。我们的GAN模型生成的路径概率分布用于引导RRT算法的采样过程。我们在由 \cite {zhang2021generative} 生成的数据集上进行了网络的训练和测试，并证明了我们的算法在图像质量生成指标（如IOU分数，Dice分数）和路径规划指标（如路径长度和成功率）方面均优于先前的最先进算法。

    Sampling-based path planning algorithms play an important role in autonomous robotics. However, a common problem among the RRT-based algorithms is that the initial path generated is not optimal and the convergence is too slow to be used in real-world applications. In this paper, we propose a novel image-based learning algorithm (CBAGAN-RRT) using a Convolutional Block Attention Generative Adversarial Network with a combination of spatial and channel attention and a novel loss function to design the heuristics, find a better optimal path, and improve the convergence of the algorithm both concerning time and speed. The probability distribution of the paths generated from our GAN model is used to guide the sampling process for the RRT algorithm. We train and test our network on the dataset generated by \cite{zhang2021generative} and demonstrate that our algorithm outperforms the previous state-of-the-art algorithms using both the image quality generation metrics like IOU Score, Dice Score
    

