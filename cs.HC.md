# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video.](http://arxiv.org/abs/2303.13397) | 提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。 |

# 详细

[^1]: DDT：一种基于扩散驱动变压器的从视频中恢复人体网格的框架

    DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video. (arXiv:2303.13397v1 [cs.CV])

    [http://arxiv.org/abs/2303.13397](http://arxiv.org/abs/2303.13397)

    提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。

    

    人体网格恢复（HMR）为各种实际应用提供了丰富的人体信息，例如游戏、人机交互和虚拟现实。与单一图像方法相比，基于视频的方法可以利用时间信息通过融合人体运动先验进一步提高性能。然而，像 VIBE 这样的多对多方法存在运动平滑性和时间一致性的挑战。而像 TCMR 和 MPS-Net 这样的多对一方法则依赖于未来帧，在推理过程中是非因果和时间效率低下的。为了解决这些挑战，提出了一种新的基于扩散驱动变压器的视频 HMR 框架（DDT）。DDT 旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性。作为一种多对多方法，DDT 的解码器输出所有帧的人体网格，使 DDT 更适用于时间效率至关重要的实际应用。

    Human mesh recovery (HMR) provides rich human body information for various real-world applications such as gaming, human-computer interaction, and virtual reality. Compared to single image-based methods, video-based methods can utilize temporal information to further improve performance by incorporating human body motion priors. However, many-to-many approaches such as VIBE suffer from motion smoothness and temporal inconsistency. While many-to-one approaches such as TCMR and MPS-Net rely on the future frames, which is non-causal and time inefficient during inference. To address these challenges, a novel Diffusion-Driven Transformer-based framework (DDT) for video-based HMR is presented. DDT is designed to decode specific motion patterns from the input sequence, enhancing motion smoothness and temporal consistency. As a many-to-many approach, the decoder of our DDT outputs the human mesh of all the frames, making DDT more viable for real-world applications where time efficiency is cruc
    

