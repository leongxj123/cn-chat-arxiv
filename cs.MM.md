# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Generate Conditional Tri-plane for 3D-aware Expression Controllable Portrait Animation](https://arxiv.org/abs/2404.00636) | 本文提出了一种一次性的3D感知肖像动画方法Export3D，通过引入三平面生成器和对比预训练框架，实现了控制给定肖像图像的面部表情和摄像机视角，提供了一种新的表达方式。 |
| [^2] | [DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video.](http://arxiv.org/abs/2303.13397) | 提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。 |

# 详细

[^1]: 学习生成条件化三平面用于3D感知表情可控肖像动画

    Learning to Generate Conditional Tri-plane for 3D-aware Expression Controllable Portrait Animation

    [https://arxiv.org/abs/2404.00636](https://arxiv.org/abs/2404.00636)

    本文提出了一种一次性的3D感知肖像动画方法Export3D，通过引入三平面生成器和对比预训练框架，实现了控制给定肖像图像的面部表情和摄像机视角，提供了一种新的表达方式。

    

    在本文中，我们提出了一种一次性的3D感知肖像动画方法Export3D，能够控制给定肖像图像的面部表情和摄像机视角。为了实现这一目标，我们引入了一个三平面生成器，通过将3DMM的表情参数转移到源图像中直接生成3D先验的三平面。然后，通过可微分体积渲染将三平面解码为不同视角的图像。现有的肖像动画方法严重依赖于图像变形来在运动空间中传输表情，挑战在外观和表情的分离上。相比之下，我们提出了一个用于无外观表情参数的对比预训练框架，消除了在传输跨身份表达时不良外观交换。大量实验证明，我们的预训练框架能够学习隐藏在3DMM中的无外观表达表示。

    arXiv:2404.00636v1 Announce Type: cross  Abstract: In this paper, we present Export3D, a one-shot 3D-aware portrait animation method that is able to control the facial expression and camera view of a given portrait image. To achieve this, we introduce a tri-plane generator that directly generates a tri-plane of 3D prior by transferring the expression parameter of 3DMM into the source image. The tri-plane is then decoded into the image of different view through a differentiable volume rendering. Existing portrait animation methods heavily rely on image warping to transfer the expression in the motion space, challenging on disentanglement of appearance and expression. In contrast, we propose a contrastive pre-training framework for appearance-free expression parameter, eliminating undesirable appearance swap when transferring a cross-identity expression. Extensive experiments show that our pre-training framework can learn the appearance-free expression representation hidden in 3DMM, and 
    
[^2]: DDT：一种基于扩散驱动变压器的从视频中恢复人体网格的框架

    DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video. (arXiv:2303.13397v1 [cs.CV])

    [http://arxiv.org/abs/2303.13397](http://arxiv.org/abs/2303.13397)

    提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。

    

    人体网格恢复（HMR）为各种实际应用提供了丰富的人体信息，例如游戏、人机交互和虚拟现实。与单一图像方法相比，基于视频的方法可以利用时间信息通过融合人体运动先验进一步提高性能。然而，像 VIBE 这样的多对多方法存在运动平滑性和时间一致性的挑战。而像 TCMR 和 MPS-Net 这样的多对一方法则依赖于未来帧，在推理过程中是非因果和时间效率低下的。为了解决这些挑战，提出了一种新的基于扩散驱动变压器的视频 HMR 框架（DDT）。DDT 旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性。作为一种多对多方法，DDT 的解码器输出所有帧的人体网格，使 DDT 更适用于时间效率至关重要的实际应用。

    Human mesh recovery (HMR) provides rich human body information for various real-world applications such as gaming, human-computer interaction, and virtual reality. Compared to single image-based methods, video-based methods can utilize temporal information to further improve performance by incorporating human body motion priors. However, many-to-many approaches such as VIBE suffer from motion smoothness and temporal inconsistency. While many-to-one approaches such as TCMR and MPS-Net rely on the future frames, which is non-causal and time inefficient during inference. To address these challenges, a novel Diffusion-Driven Transformer-based framework (DDT) for video-based HMR is presented. DDT is designed to decode specific motion patterns from the input sequence, enhancing motion smoothness and temporal consistency. As a many-to-many approach, the decoder of our DDT outputs the human mesh of all the frames, making DDT more viable for real-world applications where time efficiency is cruc
    

