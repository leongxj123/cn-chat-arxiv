# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Adversarially Robust Dataset Distillation by Curvature Regularization](https://arxiv.org/abs/2403.10045) | 本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。 |
| [^2] | [Towards Enhanced Controllability of Diffusion Models.](http://arxiv.org/abs/2302.14368) | 本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。 |

# 详细

[^1]: 通过曲率正则化实现对抗鲁棒性数据集精炼

    Towards Adversarially Robust Dataset Distillation by Curvature Regularization

    [https://arxiv.org/abs/2403.10045](https://arxiv.org/abs/2403.10045)

    本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。

    

    数据集精炼（DD）允许将数据集精炼为原始大小的分数，同时保留丰富的分布信息，使得在精炼数据集上训练的模型可以在节省显著计算负载的同时达到可比的准确性。最近在这一领域的研究集中在提高在精炼数据集上训练的模型的准确性。在本文中，我们旨在探索DD的一种新视角。我们研究如何在精炼数据集中嵌入对抗鲁棒性，以使在这些数据集上训练的模型保持高精度的同时获得更好的对抗鲁棒性。我们提出了一种通过将曲率正则化纳入到精炼过程中来实现这一目标的新方法，而这种方法的计算开销比标准的对抗训练要少得多。大量的实证实验表明，我们的方法不仅在准确性上优于标准对抗训练，同时在对抗性能方面也取得了显著改进。

    arXiv:2403.10045v1 Announce Type: new  Abstract: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accur
    
[^2]: 实现扩展扩展扩散模型的可控性

    Towards Enhanced Controllability of Diffusion Models. (arXiv:2302.14368v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.14368](http://arxiv.org/abs/2302.14368)

    本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。

    

    去噪扩散模型在生成逼真、高质量和多样化图像方面表现出卓越能力。然而，在生成过程中的可控程度尚未得到充分探讨。受基于GAN潜在空间的图像操纵技术启发，我们训练了一个条件于两个潜在编码、一个空间内容掩码和一个扁平的样式嵌入的扩散模型。我们依赖于扩散模型渐进去噪过程的感性偏置，在空间结构掩码中编码姿势/布局信息，在样式代码中编码语义/样式信息。我们提出了两种通用的采样技术来改善可控性。我们扩展了可组合的扩散模型，允许部分依赖于条件输入，以提高生成质量，同时还提供对每个潜在代码和它们的联合分布量的控制。我们还提出了时间步相关的内容和样式潜在权重调度，进一步提高了控制性。

    Denoising Diffusion models have shown remarkable capabilities in generating realistic, high-quality and diverse images. However, the extent of controllability during generation is underexplored. Inspired by techniques based on GAN latent space for image manipulation, we train a diffusion model conditioned on two latent codes, a spatial content mask and a flattened style embedding. We rely on the inductive bias of the progressive denoising process of diffusion models to encode pose/layout information in the spatial structure mask and semantic/style information in the style code. We propose two generic sampling techniques for improving controllability. We extend composable diffusion models to allow for some dependence between conditional inputs, to improve the quality of generations while also providing control over the amount of guidance from each latent code and their joint distribution. We also propose timestep dependent weight scheduling for content and style latents to further impro
    

