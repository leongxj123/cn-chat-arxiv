# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](https://arxiv.org/abs/2403.12034) | 利用预训练的视频扩散模型，本文提出了一个可生成大规模3D数据集的VFusion3D模型。 |

# 详细

[^1]: VFusion3D: 从视频扩散模型中学习可扩展的3D生成模型

    VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models

    [https://arxiv.org/abs/2403.12034](https://arxiv.org/abs/2403.12034)

    利用预训练的视频扩散模型，本文提出了一个可生成大规模3D数据集的VFusion3D模型。

    

    本文提出了一种新颖的范式，利用预训练的视频扩散模型构建可扩展的3D生成模型。构建基础3D生成模型的主要障碍是3D数据的有限可用性。与图像、文本或视频不同，3D数据不容易获取且难以获得，这导致与其他类型数据的数量相比存在显着的规模差异。为了解决这个问题，我们提出使用一个通过大量文本、图像和视频训练的视频扩散模型作为3D数据的知识源。通过微调解锁其多视角生成能力，我们生成一个大规模的合成多视角数据集来训练前馈3D生成模型。

    arXiv:2403.12034v1 Announce Type: cross  Abstract: This paper presents a novel paradigm for building scalable 3D generative models utilizing pre-trained video diffusion models. The primary obstacle in developing foundation 3D generative models is the limited availability of 3D data. Unlike images, texts, or videos, 3D data are not readily accessible and are difficult to acquire. This results in a significant disparity in scale compared to the vast quantities of other types of data. To address this issue, we propose using a video diffusion model, trained with extensive volumes of text, images, and videos, as a knowledge source for 3D data. By unlocking its multi-view generative capabilities through fine-tuning, we generate a large-scale synthetic multi-view dataset to train a feed-forward 3D generative model. The proposed model, VFusion3D, trained on nearly 3M synthetic multi-view data, can generate a 3D asset from a single image in seconds and achieves superior performance when compare
    

