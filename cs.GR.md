# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics](https://arxiv.org/abs/2403.11821) | 评估文本到图像合成中，提出了针对图像质量的新评估指标，以确保文本和图像内容的对齐，并提出了新的分类法来归纳这些指标 |
| [^2] | [Gaussian Splashing: Dynamic Fluid Synthesis with Gaussian Splatting.](http://arxiv.org/abs/2401.15318) | 高斯喷溅技术相结合物理基础动画和3D高斯喷溅，可以在虚拟场景中创造出无可比拟的效果，同时实现渲染、视图合成以及固体和流体的动态管理和交互。 |
| [^3] | [Towards Enhanced Controllability of Diffusion Models.](http://arxiv.org/abs/2302.14368) | 本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。 |

# 详细

[^1]: 评估文本到图像合成：图像质量度量的调查与分类

    Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics

    [https://arxiv.org/abs/2403.11821](https://arxiv.org/abs/2403.11821)

    评估文本到图像合成中，提出了针对图像质量的新评估指标，以确保文本和图像内容的对齐，并提出了新的分类法来归纳这些指标

    

    最近，通过利用语言和视觉结合的基础模型，推动了文本到图像合成方面的进展。这些模型在互联网或其他大规模数据库中的海量文本-图像对上进行了预训练。随着对高质量图像生成的需求转向确保文本与图像之间的内容对齐，已开发了新颖的评估度量标准，旨在模拟人类判断。因此，研究人员开始收集具有越来越复杂注释的数据集，以研究视觉语言模型的组成性及其作为文本与图像内容组成对齐质量度量的其纳入。在这项工作中，我们全面介绍了现有的文本到图像评估指标，并提出了一个新的分类法来对这些指标进行分类。我们还审查了经常采用的文本-图像基准数据集

    arXiv:2403.11821v1 Announce Type: cross  Abstract: Recent advances in text-to-image synthesis have been enabled by exploiting a combination of language and vision through foundation models. These models are pre-trained on tremendous amounts of text-image pairs sourced from the World Wide Web or other large-scale databases. As the demand for high-quality image generation shifts towards ensuring content alignment between text and image, novel evaluation metrics have been developed with the aim of mimicking human judgments. Thus, researchers have started to collect datasets with increasingly complex annotations to study the compositionality of vision-language models and their incorporation as a quality measure of compositional alignment between text and image contents. In this work, we provide a comprehensive overview of existing text-to-image evaluation metrics and propose a new taxonomy for categorizing these metrics. We also review frequently adopted text-image benchmark datasets befor
    
[^2]: 高斯喷溅：利用高斯飘落动态合成流体

    Gaussian Splashing: Dynamic Fluid Synthesis with Gaussian Splatting. (arXiv:2401.15318v1 [cs.GR])

    [http://arxiv.org/abs/2401.15318](http://arxiv.org/abs/2401.15318)

    高斯喷溅技术相结合物理基础动画和3D高斯喷溅，可以在虚拟场景中创造出无可比拟的效果，同时实现渲染、视图合成以及固体和流体的动态管理和交互。

    

    我们展示了将物理基础动画与3D高斯喷溅（3DGS）相结合的可行性，以在使用3DGS重建的虚拟场景中创建新效果。利用高斯喷溅和基于位置的动力学（PBD）在底层表示中的一致性，我们以连贯的方式管理渲染、视图合成以及固体和流体的动态。类似于高斯着色器，我们通过添加法线增强每个高斯核，将核的方向与表面法线对齐，以改进PBD模拟。这种方法有效消除了固体旋转变形产生的尖峰噪声。它还使我们能够将基于物理的渲染集成到流体的动态表面反射中。因此，我们的框架能够真实地复现动态流体上的表面亮点，并促进场景对象与流体之间的交互。

    We demonstrate the feasibility of integrating physics-based animations of solids and fluids with 3D Gaussian Splatting (3DGS) to create novel effects in virtual scenes reconstructed using 3DGS. Leveraging the coherence of the Gaussian splatting and position-based dynamics (PBD) in the underlying representation, we manage rendering, view synthesis, and the dynamics of solids and fluids in a cohesive manner. Similar to Gaussian shader, we enhance each Gaussian kernel with an added normal, aligning the kernel's orientation with the surface normal to refine the PBD simulation. This approach effectively eliminates spiky noises that arise from rotational deformation in solids. It also allows us to integrate physically based rendering to augment the dynamic surface reflections on fluids. Consequently, our framework is capable of realistically reproducing surface highlights on dynamic fluids and facilitating interactions between scene objects and fluids from new views. For more information, pl
    
[^3]: 实现扩展扩展扩散模型的可控性

    Towards Enhanced Controllability of Diffusion Models. (arXiv:2302.14368v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.14368](http://arxiv.org/abs/2302.14368)

    本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。

    

    去噪扩散模型在生成逼真、高质量和多样化图像方面表现出卓越能力。然而，在生成过程中的可控程度尚未得到充分探讨。受基于GAN潜在空间的图像操纵技术启发，我们训练了一个条件于两个潜在编码、一个空间内容掩码和一个扁平的样式嵌入的扩散模型。我们依赖于扩散模型渐进去噪过程的感性偏置，在空间结构掩码中编码姿势/布局信息，在样式代码中编码语义/样式信息。我们提出了两种通用的采样技术来改善可控性。我们扩展了可组合的扩散模型，允许部分依赖于条件输入，以提高生成质量，同时还提供对每个潜在代码和它们的联合分布量的控制。我们还提出了时间步相关的内容和样式潜在权重调度，进一步提高了控制性。

    Denoising Diffusion models have shown remarkable capabilities in generating realistic, high-quality and diverse images. However, the extent of controllability during generation is underexplored. Inspired by techniques based on GAN latent space for image manipulation, we train a diffusion model conditioned on two latent codes, a spatial content mask and a flattened style embedding. We rely on the inductive bias of the progressive denoising process of diffusion models to encode pose/layout information in the spatial structure mask and semantic/style information in the style code. We propose two generic sampling techniques for improving controllability. We extend composable diffusion models to allow for some dependence between conditional inputs, to improve the quality of generations while also providing control over the amount of guidance from each latent code and their joint distribution. We also propose timestep dependent weight scheduling for content and style latents to further impro
    

