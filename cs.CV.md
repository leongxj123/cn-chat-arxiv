# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spatial-and-Frequency-aware Restoration method for Images based on Diffusion Models](https://arxiv.org/abs/2401.17629) | 本文提出了一种名为SaFaRI的基于空间和频率感知的扩散模型，用于图像恢复。在各种噪声逆问题上，SaFaRI在ImageNet数据集和FFHQ数据集上实现了最先进的性能。 |
| [^2] | [LSAP: Rethinking Inversion Fidelity, Perception and Editability in GAN Latent Space.](http://arxiv.org/abs/2209.12746) | LSAP通过对潜空间实现对齐解决了反演和编辑结果中保真度、感知和可编辑性的问题，使得在保留重建保真度的前提下具有更好的感知和可编辑性。 |

# 详细

[^1]: 基于扩散模型的空间和频率感知图像恢复方法

    Spatial-and-Frequency-aware Restoration method for Images based on Diffusion Models

    [https://arxiv.org/abs/2401.17629](https://arxiv.org/abs/2401.17629)

    本文提出了一种名为SaFaRI的基于空间和频率感知的扩散模型，用于图像恢复。在各种噪声逆问题上，SaFaRI在ImageNet数据集和FFHQ数据集上实现了最先进的性能。

    

    扩散模型最近成为一种有前途的图像恢复（IR）框架，因为它们能够产生高质量的重建结果并且与现有方法兼容。现有的解决IR中噪声逆问题的方法通常仅考虑像素级的数据保真度。在本文中，我们提出了一种名为SaFaRI的基于空间和频率感知的扩散模型，用于处理带有高斯噪声的IR问题。我们的模型鼓励图像在空间和频率域中保持数据保真度，从而提高重建质量。我们全面评估了我们的模型在各种噪声逆问题上的性能，包括修复、降噪和超分辨率。我们的细致评估表明，SaFaRI在ImageNet数据集和FFHQ数据集上实现了最先进的性能，以LPIPS和FID指标超过了现有的零样本IR方法。

    Diffusion models have recently emerged as a promising framework for Image Restoration (IR), owing to their ability to produce high-quality reconstructions and their compatibility with established methods. Existing methods for solving noisy inverse problems in IR, considers the pixel-wise data-fidelity. In this paper, we propose SaFaRI, a spatial-and-frequency-aware diffusion model for IR with Gaussian noise. Our model encourages images to preserve data-fidelity in both the spatial and frequency domains, resulting in enhanced reconstruction quality. We comprehensively evaluate the performance of our model on a variety of noisy inverse problems, including inpainting, denoising, and super-resolution. Our thorough evaluation demonstrates that SaFaRI achieves state-of-the-art performance on both the ImageNet datasets and FFHQ datasets, outperforming existing zero-shot IR methods in terms of LPIPS and FID metrics.
    
[^2]: LSAP: 重新思考GAN潜空间中反演的保真度、感知和可编辑性

    LSAP: Rethinking Inversion Fidelity, Perception and Editability in GAN Latent Space. (arXiv:2209.12746v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.12746](http://arxiv.org/abs/2209.12746)

    LSAP通过对潜空间实现对齐解决了反演和编辑结果中保真度、感知和可编辑性的问题，使得在保留重建保真度的前提下具有更好的感知和可编辑性。

    

    随着方法的发展，反演主要分为两个步骤。第一步是图像嵌入，在这个步骤中，编码器或者优化过程嵌入图像以获取相应的潜在码。之后，第二步旨在改善反演和编辑结果，我们称之为结果细化。尽管第二步显著提高了保真度，但感知和可编辑性几乎没有改变，深度依赖于在第一步中获得的反向潜在码。因此，重要的问题是在保留重建保真度的同时获得具有更好感知和可编辑性的潜在码。在这项工作中，我们首先指出这两个特征与反向码与合成分布的对齐（或不对齐）程度有关。然后，我们提出了潜空间对齐反演范例（LSAP），其中包括评估指标和解决此问题的解决方法。具体而言，我们引入了标准化风格空间（$\mathcal{S^N}$）和标准化内容空间（$\mathcal{C^N}$），分别在风格和内容上对齐正向和负向潜在码和合成分布。 LSAP在各种任务中都取得了最先进的结果，例如图像编辑、图像转换和图像合成。此外，我们证明了LSAP具有比以前方法更好的特性，如改进的可编辑性、视觉质量和更少的模式崩塌。

    As the methods evolve, inversion is mainly divided into two steps. The first step is Image Embedding, in which an encoder or optimization process embeds images to get the corresponding latent codes. Afterward, the second step aims to refine the inversion and editing results, which we named Result Refinement. Although the second step significantly improves fidelity, perception and editability are almost unchanged, deeply dependent on inverse latent codes attained in the first step. Therefore, a crucial problem is gaining the latent codes with better perception and editability while retaining the reconstruction fidelity. In this work, we first point out that these two characteristics are related to the degree of alignment (or disalignment) of the inverse codes with the synthetic distribution. Then, we propose Latent Space Alignment Inversion Paradigm (LSAP), which consists of evaluation metric and solution for this problem. Specifically, we introduce Normalized Style Space ($\mathcal{S^N
    

