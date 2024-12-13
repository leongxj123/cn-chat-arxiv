# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Comprehensive Multi-scale Approach for Speech and Dynamics Synchrony in Talking Head Generation.](http://arxiv.org/abs/2307.03270) | 该论文提出了一种多尺度方法，通过使用多尺度音视同步损失和多尺度自回归生成对抗网络，实现了语音和头部动力学的同步生成。实验证明，在当前状态下取得了显著的改进。 |

# 详细

[^1]: 语音和动力学同步的全面多尺度方法在虚拟说话头生成中的应用

    A Comprehensive Multi-scale Approach for Speech and Dynamics Synchrony in Talking Head Generation. (arXiv:2307.03270v1 [cs.GR])

    [http://arxiv.org/abs/2307.03270](http://arxiv.org/abs/2307.03270)

    该论文提出了一种多尺度方法，通过使用多尺度音视同步损失和多尺度自回归生成对抗网络，实现了语音和头部动力学的同步生成。实验证明，在当前状态下取得了显著的改进。

    

    使用深度生成模型使用语音输入信号对静态面部图像进行动画化是一个活跃的研究课题，并且近期取得了重要的进展。然而，目前很大一部分工作都集中在嘴唇同步和渲染质量上，很少关注自然头部运动的生成，更不用说头部运动与语音的视听相关性了。本文提出了一种多尺度音视同步损失和多尺度自回归生成对抗网络，以更好地处理语音与头部和嘴唇动力学之间的短期和长期相关性。特别地，我们在多模态输入金字塔上训练了一堆同步模型，并将这些模型用作多尺度生成网络中的指导，以产生不同时间尺度上的音频对齐运动展开。我们的生成器在面部标志域中操作，这是一种标准的低维头部表示方法。实验证明，在头部运动动力学方面取得了显著的改进。

    Animating still face images with deep generative models using a speech input signal is an active research topic and has seen important recent progress. However, much of the effort has been put into lip syncing and rendering quality while the generation of natural head motion, let alone the audio-visual correlation between head motion and speech, has often been neglected. In this work, we propose a multi-scale audio-visual synchrony loss and a multi-scale autoregressive GAN to better handle short and long-term correlation between speech and the dynamics of the head and lips. In particular, we train a stack of syncer models on multimodal input pyramids and use these models as guidance in a multi-scale generator network to produce audio-aligned motion unfolding over diverse time scales. Our generator operates in the facial landmark domain, which is a standard low-dimensional head representation. The experiments show significant improvements over the state of the art in head motion dynamic
    

