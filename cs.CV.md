# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nellie: Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy](https://arxiv.org/abs/2403.13214) | Nellie是一个自动化、无偏见的管道，用于在2D/3D活细胞显微镜下分割、跟踪和提取多样细胞内结构特征，具有强大的分层分割和高度可定制的分析能力。 |
| [^2] | [AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning](https://arxiv.org/abs/2402.00769) | AnimateLCM提出了一种分离的一致性学习策略，通过将图像生成优先级和动作生成优先级的蒸馏分离开来，提高了训练效率并增强了生成的视觉质量。 |
| [^3] | [Augmentation-aware Self-supervised Learning with Guided Projector.](http://arxiv.org/abs/2306.06082) | 本文提出了一种名为CASSLE的方法，它通过修改自监督学习中的有向投影网络，利用增强信息来提高模型处理图像特征的鲁棒性。 |

# 详细

[^1]: Nellie：自动的2D/3D活细胞显微镜下器官分割、跟踪和分层特征提取

    Nellie: Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy

    [https://arxiv.org/abs/2403.13214](https://arxiv.org/abs/2403.13214)

    Nellie是一个自动化、无偏见的管道，用于在2D/3D活细胞显微镜下分割、跟踪和提取多样细胞内结构特征，具有强大的分层分割和高度可定制的分析能力。

    

    动态细胞器的分析仍然是一个严峻的挑战，但对于理解生物学过程至关重要。我们介绍了Nellie，这是一个自动且无偏见的管道，用于分割、跟踪和提取多样的细胞内结构特征。Nellie能够适应图像的元数据，消除了用户的输入。Nellie的预处理管道在多个细胞内尺度上增强了结构对比度，从而实现对亚器官区域的强大分层分割。通过半径自适应的模式匹配方案生成和跟踪内部运动捕捉标记，并用作亚体积流插值的指南。Nellie在多个分层水平提取大量特征，用于深度和可定制的分析。Nellie具有基于Napari的GUI，实现无代码操作和可视化，同时其模块化的开源代码库提供了经验丰富用户的自定义能力。

    arXiv:2403.13214v1 Announce Type: cross  Abstract: The analysis of dynamic organelles remains a formidable challenge, though key to understanding biological processes. We introduce Nellie, an automated and unbiased pipeline for segmentation, tracking, and feature extraction of diverse intracellular structures. Nellie adapts to image metadata, eliminating user input. Nellie's preprocessing pipeline enhances structural contrast on multiple intracellular scales allowing for robust hierarchical segmentation of sub-organellar regions. Internal motion capture markers are generated and tracked via a radius-adaptive pattern matching scheme, and used as guides for sub-voxel flow interpolation. Nellie extracts a plethora of features at multiple hierarchical levels for deep and customizable analysis. Nellie features a Napari-based GUI that allows for code-free operation and visualization, while its modular open-source codebase invites customization by experienced users. We demonstrate Nellie's wi
    
[^2]: AnimateLCM: 使用分离的一致性学习加速个性化的扩散模型和适配器的动画生成

    AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning

    [https://arxiv.org/abs/2402.00769](https://arxiv.org/abs/2402.00769)

    AnimateLCM提出了一种分离的一致性学习策略，通过将图像生成优先级和动作生成优先级的蒸馏分离开来，提高了训练效率并增强了生成的视觉质量。

    

    视频扩散模型因其能够产生连贯且高保真度的视频而受到越来越多的关注。然而，迭代的去噪过程使其计算密集且耗时，从而限制了其应用。受一致性模型（CM）的启发，该模型通过最小的步骤蒸馏预训练的图像扩散模型以加速采样，以及其在条件图像生成上的成功扩展——潜在一致性模型（LCM），我们提出了AnimateLCM，允许在最小的步骤内生成高保真度的视频。我们提出了一种分离的一致性学习策略，将图像生成优先级和动作生成优先级的蒸馏分离开来，这提高了训练效率并增强了生成的视觉质量。此外，为了实现稳定的扩散社区中的即插即用适配器的组合以实现各种修改，我们还引入了适配器的概念。

    Video diffusion models has been gaining increasing attention for its ability to produce videos that are both coherent and of high fidelity. However, the iterative denoising process makes it computationally intensive and time-consuming, thus limiting its applications. Inspired by the Consistency Model (CM) that distills pretrained image diffusion models to accelerate the sampling with minimal steps and its successful extension Latent Consistency Model (LCM) on conditional image generation, we propose AnimateLCM, allowing for high-fidelity video generation within minimal steps. Instead of directly conducting consistency learning on the raw video dataset, we propose a decoupled consistency learning strategy that decouples the distillation of image generation priors and motion generation priors, which improves the training efficiency and enhance the generation visual quality. Additionally, to enable the combination of plug-and-play adapters in stable diffusion community to achieve various 
    
[^3]: 增强感知的有向投影自监督学习

    Augmentation-aware Self-supervised Learning with Guided Projector. (arXiv:2306.06082v1 [cs.CV])

    [http://arxiv.org/abs/2306.06082](http://arxiv.org/abs/2306.06082)

    本文提出了一种名为CASSLE的方法，它通过修改自监督学习中的有向投影网络，利用增强信息来提高模型处理图像特征的鲁棒性。

    

    自监督学习是从无标签数据中学习健壮表示的强大技术。SimCLR和MoCo等方法通过学习对应用的数据增强保持不变，能够达到与监督方法相当的质量。然而，这种不变性可能对解决某些下游任务有害，这些任务依赖于受到预训练期间使用的增强影响的特征，例如颜色。在本文中，我们提出通过修改自监督架构的常见组件之一的有向投影网络，来促进表示空间对这些特征的敏感性。具体而言，我们为投影器补充有关应用于图像的增强的信息。为了让投影器在解决自监督学习任务时利用这种辅助指导，特征提取器学习在其表示中保留增强信息。我们的方法被称为有向投影自监督学习（CASSLE），通过这种方法提高了模型处理图像特征的鲁棒性。

    Self-supervised learning (SSL) is a powerful technique for learning robust representations from unlabeled data. By learning to remain invariant to applied data augmentations, methods such as SimCLR and MoCo are able to reach quality on par with supervised approaches. However, this invariance may be harmful to solving some downstream tasks which depend on traits affected by augmentations used during pretraining, such as color. In this paper, we propose to foster sensitivity to such characteristics in the representation space by modifying the projector network, a common component of self-supervised architectures. Specifically, we supplement the projector with information about augmentations applied to images. In order for the projector to take advantage of this auxiliary guidance when solving the SSL task, the feature extractor learns to preserve the augmentation information in its representations. Our approach, coined Conditional Augmentation-aware Selfsupervised Learning (CASSLE), is d
    

