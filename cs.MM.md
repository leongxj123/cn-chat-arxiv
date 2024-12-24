# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Content Adaptive Learnable Time-Frequency Representation For Audio Signal Processing.](http://arxiv.org/abs/2303.10446) | 该论文提出了一种用于音频信号处理的内容自适应可学习时频表示法，通过学习卷积滤波器与变换器架构来将小的波形块投影到小的潜在维度上。 |

# 详细

[^1]: 一种用于音频信号处理的内容自适应可学习时频表示法

    A Content Adaptive Learnable Time-Frequency Representation For Audio Signal Processing. (arXiv:2303.10446v1 [cs.SD])

    [http://arxiv.org/abs/2303.10446](http://arxiv.org/abs/2303.10446)

    该论文提出了一种用于音频信号处理的内容自适应可学习时频表示法，通过学习卷积滤波器与变换器架构来将小的波形块投影到小的潜在维度上。

    

    我们提出了一个可学习的内容自适应前端，用于音频信号处理。在深度学习的现代出现之前，我们使用固定表示的、不可学习的前端，如谱图或梅尔谱图，带/不带神经结构。随着卷积架构支持ASR和声学场景理解等各种应用，转向可学习前端，即从头开始学习和优化特定任务所需的基础函数和权重。在没有卷积块的变形器架构中，线性层将小的波形块投影到小的潜在维度上，然后将它们馈送到变形器架构中。在这项工作中，我们提出了一种计算内容自适应学习时频表示的方法。

    We propose a learnable content adaptive front end for audio signal processing. Before the modern advent of deep learning, we used fixed representation non-learnable front-ends like spectrogram or mel-spectrogram with/without neural architectures. With convolutional architectures supporting various applications such as ASR and acoustic scene understanding, a shift to a learnable front ends occurred in which both the type of basis functions and the weight were learned from scratch and optimized for the particular task of interest. With the shift to transformer-based architectures with no convolutional blocks present, a linear layer projects small waveform patches onto a small latent dimension before feeding them to a transformer architecture. In this work, we propose a way of computing a content-adaptive learnable time-frequency representation. We pass each audio signal through a bank of convolutional filters, each giving a fixed-dimensional vector. It is akin to learning a bank of finit
    

