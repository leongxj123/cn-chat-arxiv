# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RGI-Net: 3D Room Geometry Inference from Room Impulse Responses in the Absence of First-order Echoes](https://arxiv.org/abs/2309.01513) | RGI-Net通过深度神经网络学习和利用房间脉冲响应中高阶反射之间的关系，实现在没有传统假设的情况下推断房间几何信息。 |
| [^2] | [SongDriver2: Real-time Emotion-based Music Arrangement with Soft Transition.](http://arxiv.org/abs/2305.08029) | SongDriver2实现了基于情绪的实时音乐编排，并提出了柔和过渡机制，使音乐具有高度真实性和平滑过渡。 |
| [^3] | [Leveraging Pre-trained AudioLDM for Text to Sound Generation: A Benchmark Study.](http://arxiv.org/abs/2303.03857) | 本文研究了使用预训练的AudioLDM作为声音生成的骨干的优势，证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势，并在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，为未来的研究提供了基础。 |

# 详细

[^1]: RGI-Net：在没有一阶回声的情况下从房间脉冲响应中推断3D房间几何

    RGI-Net: 3D Room Geometry Inference from Room Impulse Responses in the Absence of First-order Echoes

    [https://arxiv.org/abs/2309.01513](https://arxiv.org/abs/2309.01513)

    RGI-Net通过深度神经网络学习和利用房间脉冲响应中高阶反射之间的关系，实现在没有传统假设的情况下推断房间几何信息。

    

    房间几何是实现逼真的3D音频渲染的重要先验信息。为此，利用房间脉冲响应中到达时间（TOA）或到达时间差（TDOA）信息发展了各种房间几何推断（RGI）方法。然而，传统的RGI技术提出了一些假设，如凸房间形状、已知墙壁数量和一阶反射的可见性。在这项工作中，我们引入了深度神经网络（DNN）RGI-Net，它可以在没有上述假设的情况下估计房间几何。RGI-Net学习并利用房间脉冲响应（RIRs）中的高阶反射之间的复杂关系，因此可以在形状为非凸形或RIRs中缺少一阶反射的情况下估计房间形状。该网络采用从装有圆形麦克风的紧凑音频设备测量的RIRs。

    arXiv:2309.01513v2 Announce Type: replace-cross  Abstract: Room geometry is important prior information for implementing realistic 3D audio rendering. For this reason, various room geometry inference (RGI) methods have been developed by utilizing the time of arrival (TOA) or time difference of arrival (TDOA) information in room impulse responses. However, the conventional RGI technique poses several assumptions, such as convex room shapes, the number of walls known in priori, and the visibility of first-order reflections. In this work, we introduce the deep neural network (DNN), RGI-Net, which can estimate room geometries without the aforementioned assumptions. RGI-Net learns and exploits complex relationships between high-order reflections in room impulse responses (RIRs) and, thus, can estimate room shapes even when the shape is non-convex or first-order reflections are missing in the RIRs. The network takes RIRs measured from a compact audio device equipped with a circular microphon
    
[^2]: SongDriver2：基于情绪的实时音乐编排与柔和过渡

    SongDriver2: Real-time Emotion-based Music Arrangement with Soft Transition. (arXiv:2305.08029v1 [cs.SD])

    [http://arxiv.org/abs/2305.08029](http://arxiv.org/abs/2305.08029)

    SongDriver2实现了基于情绪的实时音乐编排，并提出了柔和过渡机制，使音乐具有高度真实性和平滑过渡。

    

    基于情绪的实时音乐编排旨在将给定的音乐转化为另一个能够实时引起用户特定情感共鸣的音乐，在音乐疗法、游戏配乐和电影配乐等各种场景中具有重要应用价值。然而，由于目标情感的细粒度和可变性，平衡情感实时匹配和柔和情感转换是一项挑战。现有的研究主要集中在实现情感实时匹配，而柔和过渡的问题仍未得到充分研究，影响了音乐的整体情感一致性。本文提出了SongDriver2来解决这个问题。具体地，我们首先识别最后一个时间步的音乐情绪，然后将其与当前时间步的目标输入情绪融合。融合的情感随后作为SongDriver2根据输入旋律数据生成即将到来的音乐的指导。为了调整音乐相似性和情感实时匹配，以实现两种不同情感之间的过渡，我们设计了一种软过渡机制，将插值和平滑滤波器相结合。我们证明，所提出的SongDriver2可以生成具有高度真实性和平滑过渡的情感音乐，这表明其在基于情绪的实时音乐编排应用中具有潜在价值。

    Real-time emotion-based music arrangement, which aims to transform a given music piece into another one that evokes specific emotional resonance with the user in real-time, holds significant application value in various scenarios, e.g., music therapy, video game soundtracks, and movie scores. However, balancing emotion real-time fit with soft emotion transition is a challenge due to the fine-grained and mutable nature of the target emotion. Existing studies mainly focus on achieving emotion real-time fit, while the issue of soft transition remains understudied, affecting the overall emotional coherence of the music. In this paper, we propose SongDriver2 to address this balance. Specifically, we first recognize the last timestep's music emotion and then fuse it with the current timestep's target input emotion. The fused emotion then serves as the guidance for SongDriver2 to generate the upcoming music based on the input melody data. To adjust music similarity and emotion real-time fit f
    
[^3]: 利用预训练的AudioLDM进行文本到声音生成：基准研究

    Leveraging Pre-trained AudioLDM for Text to Sound Generation: A Benchmark Study. (arXiv:2303.03857v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2303.03857](http://arxiv.org/abs/2303.03857)

    本文研究了使用预训练的AudioLDM作为声音生成的骨干的优势，证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势，并在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，为未来的研究提供了基础。

    This paper investigates the advantages of using pre-trained AudioLDM as the backbone for sound generation, demonstrates the benefits of using pre-trained models for text-to-sound generation in data-scarcity scenarios, and evaluates various text-to-sound generation systems on several frequently used datasets under the same evaluation protocols to provide a basis for future research.

    深度神经网络最近在文本提示下实现了声音生成的突破。尽管它们的表现很有前途，但当前的文本到声音生成模型在小规模数据集（例如过度拟合）上面临问题，从而显著限制了它们的性能。在本文中，我们研究了使用预训练的AudioLDM作为声音生成的骨干的优势。我们的研究证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势。此外，实验表明，不同的训练策略（例如训练条件）可能会影响AudioLDM在不同规模的数据集上的性能。为了促进未来的研究，我们还在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，这些协议允许在共同基础上公平比较和基准测试这些方法。

    Deep neural networks have recently achieved breakthroughs in sound generation with text prompts. Despite their promising performance, current text-to-sound generation models face issues on small-scale datasets (e.g., overfitting), significantly limiting their performance. In this paper, we investigate the use of pre-trained AudioLDM, the state-of-the-art model for text-to-audio generation, as the backbone for sound generation. Our study demonstrates the advantages of using pre-trained models for text-to-sound generation, especially in data-scarcity scenarios. In addition, experiments show that different training strategies (e.g., training conditions) may affect the performance of AudioLDM on datasets of different scales. To facilitate future studies, we also evaluate various text-to-sound generation systems on several frequently used datasets under the same evaluation protocols, which allow fair comparisons and benchmarking of these methods on the common ground.
    

