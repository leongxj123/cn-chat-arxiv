# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning](https://arxiv.org/abs/2403.07376) | 本文提出了一种名为NavCoT的新策略，在视觉与语言导航中通过学习解耦推理，实现了自主导航决策，有效减轻了领域差距。 |
| [^2] | [Implicit Image-to-Image Schrodinger Bridge for CT Super-Resolution and Denoising](https://arxiv.org/abs/2403.06069) | I3SB方法通过引入非马尔可夫过程，结合损坏的图像改善纹理恢复，在CT超分辨率和去噪任务中表现优异。 |
| [^3] | [Downstream Task Guided Masking Learning in Masked Autoencoders Using Multi-Level Optimization](https://arxiv.org/abs/2402.18128) | 提出了一种新颖的框架 - 多级优化遮罩自动编码器（MLO-MAE），该框架利用来自下游任务的反馈，在预训练期间学习最佳的遮罩策略，显著提升了视觉表示学习。 |
| [^4] | [Zero shot VLMs for hate meme detection: Are we there yet?](https://arxiv.org/abs/2402.12198) | 本研究探讨了零-shot分类在处理复杂任务如恶意模因检测中的有效性 |
| [^5] | [Promoting Segment Anything Model towards Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2401.00248) | 将段分离任意模型推进至高度准确的二元图像分割，通过提出DIS-SAM框架，成功改进SAM模型在细节方面的表现，实现了显著增强的分割精度。 |
| [^6] | [DataDAM: Efficient Dataset Distillation with Attention Matching.](http://arxiv.org/abs/2310.00093) | 本研究提出了一种基于注意力匹配的高效数据集精炼(DataDAM)方法。通过匹配空间注意力来学习合成图像，从而实现了最新技术水平的性能，同时减少了训练成本。 |
| [^7] | [Mixup Your Own Pairs.](http://arxiv.org/abs/2309.16633) | 本文提出了一种名为SupReMix的方法，通过混合样本，特别是混合负样本和混合正样本，来解决回归问题中表示学习的挑战。这种方法能够提供更好的性能和更准确的回归结果。 |
| [^8] | [Confidence intervals for performance estimates in 3D medical image segmentation.](http://arxiv.org/abs/2307.10926) | 本文研究了医学图像分割中性能估计的置信区间，通过实验发现参数置信区间的宽度与分割问题的特点有关。 |

# 详细

[^1]: NavCoT: 通过学习解耦推理提升基于LLM的视觉与语言导航

    NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning

    [https://arxiv.org/abs/2403.07376](https://arxiv.org/abs/2403.07376)

    本文提出了一种名为NavCoT的新策略，在视觉与语言导航中通过学习解耦推理，实现了自主导航决策，有效减轻了领域差距。

    

    视觉与语言导航(VLN)作为具有重要研究价值的具身人工智能问题，需要一个具身代理根据自然语言指示穿越复杂的3D环境。最近的研究突出了大型语言模型(LLMs)在VLN中提高导航推理准确性和可解释性的潜力。然而，它们主要在离线方式下的使用通常在VLN任务和LLM训练语料库之间遭受显著的领域差距。本文引入了一种名为导航思维链(NavCoT)的新型策略，我们通过完成领域内高效参数训练，实现自主导航决策，有效减轻领域差距的成本。具体地，在每个时间步，LLM被提示通过作为世界模型来预测导航思维链：1)根据

    arXiv:2403.07376v1 Announce Type: cross  Abstract: Vision-and-Language Navigation (VLN), as a crucial research problem of Embodied AI, requires an embodied agent to navigate through complex 3D environments following natural language instructions. Recent research has highlighted the promising capacity of large language models (LLMs) in VLN by improving navigational reasoning accuracy and interpretability. However, their predominant use in an offline manner usually suffers from substantial domain gap between the VLN task and the LLM training corpus. This paper introduces a novel strategy called Navigational Chain-of-Thought (NavCoT), where we fulfill parameter-efficient in-domain training to enable self-guided navigational decision, leading to a significant mitigation of the domain gap in a cost-effective manner. Specifically, at each timestep, the LLM is prompted to forecast the navigational chain-of-thought by: 1) acting as a world model to imagine the next observation according to the
    
[^2]: 隐式图像对图像Schrodinger桥用于CT超分辨率和去噪

    Implicit Image-to-Image Schrodinger Bridge for CT Super-Resolution and Denoising

    [https://arxiv.org/abs/2403.06069](https://arxiv.org/abs/2403.06069)

    I3SB方法通过引入非马尔可夫过程，结合损坏的图像改善纹理恢复，在CT超分辨率和去噪任务中表现优异。

    

    有条件扩散模型因其在图像恢复任务中的有效性而得到认可，然而，其从高斯噪声开始的迭代去噪过程往往导致推断速度慢。作为一种有希望的替代方案，图像对图像Schrödinger桥（I2SB）从损坏的图像开始初始化生成过程，并集成了有条件扩散模型的训练技术。在本研究中，我们通过引入隐式图像对图像Schrödinger桥（I3SB）扩展了I2SB方法，通过在每一生成步骤中纳入损坏的图像，将其生成过程转换为非马尔可夫过程。这种增强使得I3SB能够在少量生成步骤中生成具有更好纹理恢复的图像。所提出的方法在CT超分辨率和去噪任务上得到验证，并超越了包括有条件去噪扩散概率模型在内的现有方法。

    arXiv:2403.06069v1 Announce Type: cross  Abstract: Conditional diffusion models have gained recognition for their effectiveness in image restoration tasks, yet their iterative denoising process, starting from Gaussian noise, often leads to slow inference speeds. As a promising alternative, the Image-to-Image Schr\"odinger Bridge (I2SB) initializes the generative process from corrupted images and integrates training techniques from conditional diffusion models. In this study, we extended the I2SB method by introducing the Implicit Image-to-Image Schrodinger Bridge (I3SB), transitioning its generative process to a non-Markovian process by incorporating corrupted images in each generative step. This enhancement empowers I3SB to generate images with better texture restoration using a small number of generative steps. The proposed method was validated on CT super-resolution and denoising tasks and outperformed existing methods, including the conditional denoising diffusion probabilistic mod
    
[^3]: 在遮罩自动编码器中使用多级优化的下游任务引导学习

    Downstream Task Guided Masking Learning in Masked Autoencoders Using Multi-Level Optimization

    [https://arxiv.org/abs/2402.18128](https://arxiv.org/abs/2402.18128)

    提出了一种新颖的框架 - 多级优化遮罩自动编码器（MLO-MAE），该框架利用来自下游任务的反馈，在预训练期间学习最佳的遮罩策略，显著提升了视觉表示学习。

    

    遮罩自动编码器（MAE）是视觉表示学习中自监督预训练的一个显著方法。它通过随机遮罩图像补丁，并使用未遮罩的补丁重建这些遮罩补丁。 MAE的一个关键局限性在于其忽视不同补丁的信息量不同，因为它会统一选择要遮罩的补丁。为了克服这一问题，一些方法提出基于补丁信息量进行遮罩。然而，这些方法通常不考虑下游任务的特定需求，可能导致这些任务的表示次优。作为响应，我们引入了多级优化遮罩自动编码器（MLO-MAE），这是一个新颖的框架，利用来自下游任务的端到端反馈，在预训练期间学习最佳遮罩策略。我们的实验结果突显了MLO-MAE在视觉表示学习中的显著进展。与现有方法相比，

    arXiv:2402.18128v1 Announce Type: cross  Abstract: Masked Autoencoder (MAE) is a notable method for self-supervised pretraining in visual representation learning. It operates by randomly masking image patches and reconstructing these masked patches using the unmasked ones. A key limitation of MAE lies in its disregard for the varying informativeness of different patches, as it uniformly selects patches to mask. To overcome this, some approaches propose masking based on patch informativeness. However, these methods often do not consider the specific requirements of downstream tasks, potentially leading to suboptimal representations for these tasks. In response, we introduce the Multi-level Optimized Mask Autoencoder (MLO-MAE), a novel framework that leverages end-to-end feedback from downstream tasks to learn an optimal masking strategy during pretraining. Our experimental findings highlight MLO-MAE's significant advancements in visual representation learning. Compared to existing metho
    
[^4]: 零-shot 可见语言模型用于仇恨模因检测：我们已经到达目标了吗？

    Zero shot VLMs for hate meme detection: Are we there yet?

    [https://arxiv.org/abs/2402.12198](https://arxiv.org/abs/2402.12198)

    本研究探讨了零-shot分类在处理复杂任务如恶意模因检测中的有效性

    

    社交媒体上的多媒体内容正在迅速发展，其中模因作为一种独特形式变得日益重要。不幸的是，一些恶意用户利用模因针对个人或易受攻击的社区，因此有必要识别和解决此类恶意模因。已经进行了大量研究来解决这个问题，通过开发仇恨模因检测模型。然而，传统的机器学习/深度学习模型的一个显著局限性是需要带标签的数据集才能进行准确分类。最近，研究界见证了几种可见语言模型的出现，在各种任务中展现出卓越的性能。在这项研究中，我们旨在调查这些可见语言模型在处理诸如仇恨模因检测等复杂任务中的有效性。我们使用各种提示设置来专注于对恶意/有害模因的零-shot 分类。通过我们的分析，我们o

    arXiv:2402.12198v1 Announce Type: new  Abstract: Multimedia content on social media is rapidly evolving, with memes gaining prominence as a distinctive form. Unfortunately, some malicious users exploit memes to target individuals or vulnerable communities, making it imperative to identify and address such instances of hateful memes. Extensive research has been conducted to address this issue by developing hate meme detection models. However, a notable limitation of traditional machine/deep learning models is the requirement for labeled datasets for accurate classification. Recently, the research community has witnessed the emergence of several visual language models that have exhibited outstanding performance across various tasks. In this study, we aim to investigate the efficacy of these visual language models in handling intricate tasks such as hate meme detection. We use various prompt settings to focus on zero-shot classification of hateful/harmful memes. Through our analysis, we o
    
[^5]: 将“段分离任意模型”推进至高度准确的二元图像分割

    Promoting Segment Anything Model towards Highly Accurate Dichotomous Image Segmentation

    [https://arxiv.org/abs/2401.00248](https://arxiv.org/abs/2401.00248)

    将段分离任意模型推进至高度准确的二元图像分割，通过提出DIS-SAM框架，成功改进SAM模型在细节方面的表现，实现了显著增强的分割精度。

    

    Segment Anything Model (SAM)代表了计算机视觉基础模型的重大突破，提供了大规模图像分割模型。然而，尽管SAM的零-shot表现，其分割蒙版缺乏细粒度细节，特别是在准确描绘对象边界方面。我们对SAM是否可以作为基础模型进一步改进以实现高度精确的对象分割（即称为二元图像分割DIS）抱有很高期望。为解决这一问题，我们提出了DIS-SAM，将SAM推进至DIS，具有极高的精确细节。DIS-SAM是一个专门为高度准确分割而设计的框架，保持了SAM的可促进设计。DIS-SAM采用了两阶段方法，将SAM与专门用于DIS的修改后的IS-Net集成在一起。尽管简单，DIS-SAM相比SAM和HQ-SA表现出显着增强的分割精度。

    arXiv:2401.00248v2 Announce Type: replace-cross  Abstract: The Segment Anything Model (SAM) represents a significant breakthrough into foundation models for computer vision, providing a large-scale image segmentation model. However, despite SAM's zero-shot performance, its segmentation masks lack fine-grained details, particularly in accurately delineating object boundaries. We have high expectations regarding whether SAM, as a foundation model, can be improved towards highly accurate object segmentation, which is known as dichotomous image segmentation (DIS). To address this issue, we propose DIS-SAM, which advances SAM towards DIS with extremely accurate details. DIS-SAM is a framework specifically tailored for highly accurate segmentation, maintaining SAM's promptable design. DIS-SAM employs a two-stage approach, integrating SAM with a modified IS-Net dedicated to DIS. Despite its simplicity, DIS-SAM demonstrates significantly enhanced segmentation accuracy compared to SAM and HQ-SA
    
[^6]: DataDAM: 基于注意力匹配的高效数据集精炼

    DataDAM: Efficient Dataset Distillation with Attention Matching. (arXiv:2310.00093v1 [cs.CV])

    [http://arxiv.org/abs/2310.00093](http://arxiv.org/abs/2310.00093)

    本研究提出了一种基于注意力匹配的高效数据集精炼(DataDAM)方法。通过匹配空间注意力来学习合成图像，从而实现了最新技术水平的性能，同时减少了训练成本。

    

    研究人员长期以来一直在尽量减少深度学习的训练成本，同时保持在多样化数据集上的强大泛化能力。最近的数据集精炼研究旨在通过创建一个包含更大真实数据集信息的小型合成数据集来减少训练成本，并最终实现与整个数据集训练的模型相当的测试准确性。然而，之前方法生成的合成数据并不能像原始训练数据那样分布和区分，而且会带来显著的计算成本。尽管取得了令人期待的结果，但精炼合成数据集上训练的模型与整个数据集上训练的模型之间仍然存在明显的性能差距。在本文中，我们通过使用基于注意力匹配的高效数据集精炼(DataDAM)来应对这些挑战，实现了最新技术水平的性能，同时减少了训练成本。具体而言，我们通过匹配空间注意力来学习合成图像。

    Researchers have long tried to minimize training costs in deep learning while maintaining strong generalization across diverse datasets. Emerging research on dataset distillation aims to reduce training costs by creating a small synthetic set that contains the information of a larger real dataset and ultimately achieves test accuracy equivalent to a model trained on the whole dataset. Unfortunately, the synthetic data generated by previous methods are not guaranteed to distribute and discriminate as well as the original training data, and they incur significant computational costs. Despite promising results, there still exists a significant performance gap between models trained on condensed synthetic sets and those trained on the whole dataset. In this paper, we address these challenges using efficient Dataset Distillation with Attention Matching (DataDAM), achieving state-of-the-art performance while reducing training costs. Specifically, we learn synthetic images by matching the spa
    
[^7]: 混合你自己的对比对

    Mixup Your Own Pairs. (arXiv:2309.16633v1 [cs.LG])

    [http://arxiv.org/abs/2309.16633](http://arxiv.org/abs/2309.16633)

    本文提出了一种名为SupReMix的方法，通过混合样本，特别是混合负样本和混合正样本，来解决回归问题中表示学习的挑战。这种方法能够提供更好的性能和更准确的回归结果。

    

    在表示学习中，回归问题传统上比分类问题受到的关注较少。直接应用为分类设计的表示学习技术到回归问题往往会导致潜空间中碎片化的表示，从而产生次优的性能。本文认为，由于忽视了两个关键方面：序序感知和难度，对于回归问题而言，对比学习的潜能被忽视了。为了解决这些挑战，我们提倡“混合自己的对比对进行监督性对比回归”，而不仅仅依靠真实/增强样本。具体来说，我们提出了混合式监督对比回归学习（SupReMix）。它在嵌入级别上以锚点包含的混合（锚点和一个不同的负样本的混合）作为困难负对，以锚点排除的混合（两个不同的负样本的混合）作为困难正对。这一策略形成了困难样本对学习的方式。

    In representation learning, regression has traditionally received less attention than classification. Directly applying representation learning techniques designed for classification to regression often results in fragmented representations in the latent space, yielding sub-optimal performance. In this paper, we argue that the potential of contrastive learning for regression has been overshadowed due to the neglect of two crucial aspects: ordinality-awareness and hardness. To address these challenges, we advocate "mixup your own contrastive pairs for supervised contrastive regression", instead of relying solely on real/augmented samples. Specifically, we propose Supervised Contrastive Learning for Regression with Mixup (SupReMix). It takes anchor-inclusive mixtures (mixup of the anchor and a distinct negative sample) as hard negative pairs and anchor-exclusive mixtures (mixup of two distinct negative samples) as hard positive pairs at the embedding level. This strategy formulates harde
    
[^8]: 对3D医学图像分割性能估计的置信区间研究

    Confidence intervals for performance estimates in 3D medical image segmentation. (arXiv:2307.10926v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2307.10926](http://arxiv.org/abs/2307.10926)

    本文研究了医学图像分割中性能估计的置信区间，通过实验发现参数置信区间的宽度与分割问题的特点有关。

    

    医学分割模型的评估是基于有限的例图像，因此评估结果存在噪声。除了报告平均性能指标外，报告置信区间也是非常重要的。然而，在医学图像分割中，很少有人这样做。置信区间的宽度取决于测试集大小和性能指标的散布程度（即测试集上的标准差）。对于分类问题，需要许多测试图像以避免宽泛的置信区间。然而，对于分割问题，这个情况尚未研究，因为给定的测试图像所提供的信息量不同。本文研究了医学图像分割中典型的置信区间。我们使用标准的nnU-net框架在两个来自Medical Decathlon挑战赛的数据集上进行了3D图像分割的实验，并使用Dice准确度和Hausdorff距离两个性能指标。我们发现参数置信区间的宽度与分割问题的特点有关，需要更多的研究才能得到更准确的结果。

    Medical segmentation models are evaluated empirically. As such an evaluation is based on a limited set of example images, it is unavoidably noisy. Beyond a mean performance measure, reporting confidence intervals is thus crucial. However, this is rarely done in medical image segmentation. The width of the confidence interval depends on the test set size and on the spread of the performance measure (its standard-deviation across of the test set). For classification, many test images are needed to avoid wide confidence intervals. Segmentation, however, has not been studied, and it differs by the amount of information brought by a given test image. In this paper, we study the typical confidence intervals in medical image segmentation. We carry experiments on 3D image segmentation using the standard nnU-net framework, two datasets from the Medical Decathlon challenge and two performance measures: the Dice accuracy and the Hausdorff distance. We show that the parametric confidence intervals
    

