# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models.](http://arxiv.org/abs/2305.13840) | 这篇论文提出了一种基于控制信号的可控文本生成视频的模型，通过空间-时间自注意机制和残差噪声初始化策略，可以生成更连贯的超高质量视频，成功实现了资源高效的收敛。 |

# 详细

[^1]: Control-A-Video: 控制性文本生成视频的扩散模型

    Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models. (arXiv:2305.13840v1 [cs.CV])

    [http://arxiv.org/abs/2305.13840](http://arxiv.org/abs/2305.13840)

    这篇论文提出了一种基于控制信号的可控文本生成视频的模型，通过空间-时间自注意机制和残差噪声初始化策略，可以生成更连贯的超高质量视频，成功实现了资源高效的收敛。

    

    本文提出了一种基于控制信号的可控文本生成视频（T2V）扩散模型，称为Video-ControlNet。该模型是在预训练的有条件文本生成图像（T2I）扩散模型基础上构建的，其中包括一种空间-时间自注意机制和可训练的时间层，用于有效的跨帧建模。提出了一种第一帧条件策略，以促进模型在自回归方式下生成转换自图像领域以及任意长度视频。此外，Video-ControlNet采用一种基于残差的噪声初始化策略，从输入视频中引入运动先验，从而产生更连贯的视频。通过提出的架构和策略，Video-ControlNet可以实现资源高效的收敛，生成具有细粒度控制的优质一致视频。广泛的实验证明了它的成功。

    This paper presents a controllable text-to-video (T2V) diffusion model, named Video-ControlNet, that generates videos conditioned on a sequence of control signals, such as edge or depth maps. Video-ControlNet is built on a pre-trained conditional text-to-image (T2I) diffusion model by incorporating a spatial-temporal self-attention mechanism and trainable temporal layers for efficient cross-frame modeling. A first-frame conditioning strategy is proposed to facilitate the model to generate videos transferred from the image domain as well as arbitrary-length videos in an auto-regressive manner. Moreover, Video-ControlNet employs a novel residual-based noise initialization strategy to introduce motion prior from an input video, producing more coherent videos. With the proposed architecture and strategies, Video-ControlNet can achieve resource-efficient convergence and generate superior quality and consistent videos with fine-grained control. Extensive experiments demonstrate its success i
    

