# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiffusionAct: Controllable Diffusion Autoencoder for One-shot Face Reenactment](https://arxiv.org/abs/2403.17217) | DiffusionAct是一种利用扩散模型进行神经人脸再现的新方法，能够编辑输入图像的面部姿势，实现身份和外观的保留，以及目标头部姿势和面部表情的转移。 |
| [^2] | [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey](https://arxiv.org/abs/2402.02242) | 本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。 |
| [^3] | [Promoting Segment Anything Model towards Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2401.00248) | 将段分离任意模型推进至高度准确的二元图像分割，通过提出DIS-SAM框架，成功改进SAM模型在细节方面的表现，实现了显著增强的分割精度。 |
| [^4] | [Masking Augmentation for Supervised Learning](https://arxiv.org/abs/2306.11339) | 提出了一种名为MaskSub的新方法，通过使用遮罩子模型和放松的损失函数来强化监督学习中的遮罩增强，提高了性能并加速训练过程。 |

# 详细

[^1]: DiffusionAct：可控扩散自编码器用于一次性人脸再现

    DiffusionAct: Controllable Diffusion Autoencoder for One-shot Face Reenactment

    [https://arxiv.org/abs/2403.17217](https://arxiv.org/abs/2403.17217)

    DiffusionAct是一种利用扩散模型进行神经人脸再现的新方法，能够编辑输入图像的面部姿势，实现身份和外观的保留，以及目标头部姿势和面部表情的转移。

    

    视频驱动的神经人脸再现旨在合成能成功保留源脸的身份和外观，同时转移目标头部姿势和面部表情的逼真面部图像。现有基于GAN的方法要么存在失真和视觉伪影，要么重构质量较差，即背景和一些重要的外观细节（如发型/颜色、眼镜和配饰）未被忠实重建。最近在扩散概率模型（DPMs）领域的进展使得生成高质量逼真图像成为可能。为此，本文提出了DiffusionAct，这是一种利用扩散模型的照片逼真图像生成来进行神经人脸再现的新方法。具体来说，我们提出控制Diffusion自编码器（DiffAE）的语义空间，以便编辑输入图像的面部姿势，定义为头部姿势方向。

    arXiv:2403.17217v1 Announce Type: cross  Abstract: Video-driven neural face reenactment aims to synthesize realistic facial images that successfully preserve the identity and appearance of a source face, while transferring the target head pose and facial expressions. Existing GAN-based methods suffer from either distortions and visual artifacts or poor reconstruction quality, i.e., the background and several important appearance details, such as hair style/color, glasses and accessories, are not faithfully reconstructed. Recent advances in Diffusion Probabilistic Models (DPMs) enable the generation of high-quality realistic images. To this end, in this paper we present DiffusionAct, a novel method that leverages the photo-realistic image generation of diffusion models to perform neural face reenactment. Specifically, we propose to control the semantic space of a Diffusion Autoencoder (DiffAE), in order to edit the facial pose of the input images, defined as the head pose orientation an
    
[^2]: 面向预训练视觉模型的参数高效微调：一项综述

    Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey

    [https://arxiv.org/abs/2402.02242](https://arxiv.org/abs/2402.02242)

    本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。

    

    大规模预训练的视觉模型（PVMs）展示了在各种下游视觉任务中的适应能力潜力。然而，随着最先进的PVMs达到数十亿甚至数万亿个参数，标准的全面微调范式由于高计算和存储需求变得不可持续。作为响应，研究人员正在探索参数高效微调（PEFT），旨在以最小参数修改超越全面微调的性能。本综述提供了视觉PEFT的全面概述和未来方向，对最新进展进行了系统审查。首先，我们提供了PEFT的正式定义，并讨论了模型预训练方法。然后，我们将现有方法分为三类：基于添加的、基于部分的和基于统一的。最后，我们介绍了常用的数据集和应用，并提出了潜在的未来研究挑战。该综述还提供了丰富的资源收藏。

    Large-scale pre-trained vision models (PVMs) have shown great potential for adaptability across various downstream vision tasks. However, with state-of-the-art PVMs growing to billions or even trillions of parameters, the standard full fine-tuning paradigm is becoming unsustainable due to high computational and storage demands. In response, researchers are exploring parameter-efficient fine-tuning (PEFT), which seeks to exceed the performance of full fine-tuning with minimal parameter modifications. This survey provides a comprehensive overview and future directions for visual PEFT, offering a systematic review of the latest advancements. First, we provide a formal definition of PEFT and discuss model pre-training methods. We then categorize existing methods into three categories: addition-based, partial-based, and unified-based. Finally, we introduce the commonly used datasets and applications and suggest potential future research challenges. A comprehensive collection of resources is
    
[^3]: 将“段分离任意模型”推进至高度准确的二元图像分割

    Promoting Segment Anything Model towards Highly Accurate Dichotomous Image Segmentation

    [https://arxiv.org/abs/2401.00248](https://arxiv.org/abs/2401.00248)

    将段分离任意模型推进至高度准确的二元图像分割，通过提出DIS-SAM框架，成功改进SAM模型在细节方面的表现，实现了显著增强的分割精度。

    

    Segment Anything Model (SAM)代表了计算机视觉基础模型的重大突破，提供了大规模图像分割模型。然而，尽管SAM的零-shot表现，其分割蒙版缺乏细粒度细节，特别是在准确描绘对象边界方面。我们对SAM是否可以作为基础模型进一步改进以实现高度精确的对象分割（即称为二元图像分割DIS）抱有很高期望。为解决这一问题，我们提出了DIS-SAM，将SAM推进至DIS，具有极高的精确细节。DIS-SAM是一个专门为高度准确分割而设计的框架，保持了SAM的可促进设计。DIS-SAM采用了两阶段方法，将SAM与专门用于DIS的修改后的IS-Net集成在一起。尽管简单，DIS-SAM相比SAM和HQ-SA表现出显着增强的分割精度。

    arXiv:2401.00248v2 Announce Type: replace-cross  Abstract: The Segment Anything Model (SAM) represents a significant breakthrough into foundation models for computer vision, providing a large-scale image segmentation model. However, despite SAM's zero-shot performance, its segmentation masks lack fine-grained details, particularly in accurately delineating object boundaries. We have high expectations regarding whether SAM, as a foundation model, can be improved towards highly accurate object segmentation, which is known as dichotomous image segmentation (DIS). To address this issue, we propose DIS-SAM, which advances SAM towards DIS with extremely accurate details. DIS-SAM is a framework specifically tailored for highly accurate segmentation, maintaining SAM's promptable design. DIS-SAM employs a two-stage approach, integrating SAM with a modified IS-Net dedicated to DIS. Despite its simplicity, DIS-SAM demonstrates significantly enhanced segmentation accuracy compared to SAM and HQ-SA
    
[^4]: 遮罩数据增强用于监督学习

    Masking Augmentation for Supervised Learning

    [https://arxiv.org/abs/2306.11339](https://arxiv.org/abs/2306.11339)

    提出了一种名为MaskSub的新方法，通过使用遮罩子模型和放松的损失函数来强化监督学习中的遮罩增强，提高了性能并加速训练过程。

    

    使用随机遮罩进行预训练已经成为训练技术中的新趋势。然而，监督学习在采用遮罩增强方面面临挑战，主要是由于不稳定的训练。本文提出了一种涉及遮罩增强的新方法，称为Masked Sub-model (MaskSub)。MaskSub由主模型和子模型组成；前者享受传统训练方法，而后者利用强大的遮罩增强来训练。MaskSub通过缓解类似于自蒸馏损失的放松损失函数来解决挑战。我们的分析表明，MaskSub提高了性能，训练损失的收敛速度甚至比常规训练更快，这表明我们的方法有助于训练。我们进一步验证了MaskSub在各种训练方法和模型上的有效性，包括DeiT-III，MAE微调，CLIP微调，ResNet和Swin T。

    arXiv:2306.11339v2 Announce Type: replace-cross  Abstract: Pre-training using random masking has emerged as a novel trend in training techniques. However, supervised learning faces a challenge in adopting masking augmentations, primarily due to unstable training. In this paper, we propose a novel way to involve masking augmentations dubbed Masked Sub-model (MaskSub). MaskSub consists of the main-model and sub-model; while the former enjoys conventional training recipes, the latter leverages the benefit of strong masking augmentations in training. MaskSub addresses the challenge by mitigating adverse effects through a relaxed loss function similar to a self-distillation loss. Our analysis shows that MaskSub improves performance, with the training loss converging even faster than regular training, which suggests our method facilitates training. We further validate MaskSub across diverse training recipes and models, including DeiT-III, MAE fine-tuning, CLIP fine-tuning, ResNet, and Swin T
    

