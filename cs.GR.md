# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Enhanced Controllability of Diffusion Models.](http://arxiv.org/abs/2302.14368) | 本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。 |

# 详细

[^1]: 实现扩展扩展扩散模型的可控性

    Towards Enhanced Controllability of Diffusion Models. (arXiv:2302.14368v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.14368](http://arxiv.org/abs/2302.14368)

    本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。

    

    去噪扩散模型在生成逼真、高质量和多样化图像方面表现出卓越能力。然而，在生成过程中的可控程度尚未得到充分探讨。受基于GAN潜在空间的图像操纵技术启发，我们训练了一个条件于两个潜在编码、一个空间内容掩码和一个扁平的样式嵌入的扩散模型。我们依赖于扩散模型渐进去噪过程的感性偏置，在空间结构掩码中编码姿势/布局信息，在样式代码中编码语义/样式信息。我们提出了两种通用的采样技术来改善可控性。我们扩展了可组合的扩散模型，允许部分依赖于条件输入，以提高生成质量，同时还提供对每个潜在代码和它们的联合分布量的控制。我们还提出了时间步相关的内容和样式潜在权重调度，进一步提高了控制性。

    Denoising Diffusion models have shown remarkable capabilities in generating realistic, high-quality and diverse images. However, the extent of controllability during generation is underexplored. Inspired by techniques based on GAN latent space for image manipulation, we train a diffusion model conditioned on two latent codes, a spatial content mask and a flattened style embedding. We rely on the inductive bias of the progressive denoising process of diffusion models to encode pose/layout information in the spatial structure mask and semantic/style information in the style code. We propose two generic sampling techniques for improving controllability. We extend composable diffusion models to allow for some dependence between conditional inputs, to improve the quality of generations while also providing control over the amount of guidance from each latent code and their joint distribution. We also propose timestep dependent weight scheduling for content and style latents to further impro
    

