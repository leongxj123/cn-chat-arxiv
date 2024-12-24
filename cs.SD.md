# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Content Adaptive Learnable Time-Frequency Representation For Audio Signal Processing.](http://arxiv.org/abs/2303.10446) | 该论文提出了一种用于音频信号处理的内容自适应可学习时频表示法，通过学习卷积滤波器与变换器架构来将小的波形块投影到小的潜在维度上。 |
| [^2] | [A Language Model With Million Sample Context For Raw Audio Using Transformer Architectures.](http://arxiv.org/abs/2206.08297) | 本文提出了一种采用Transformer结构和百万级样本上下文进行原始音频语言模型的自回归生成架构，能够高效地建模音频信号的长期依赖性，并取得了最先进的性能表现。 |

# 详细

[^1]: 一种用于音频信号处理的内容自适应可学习时频表示法

    A Content Adaptive Learnable Time-Frequency Representation For Audio Signal Processing. (arXiv:2303.10446v1 [cs.SD])

    [http://arxiv.org/abs/2303.10446](http://arxiv.org/abs/2303.10446)

    该论文提出了一种用于音频信号处理的内容自适应可学习时频表示法，通过学习卷积滤波器与变换器架构来将小的波形块投影到小的潜在维度上。

    

    我们提出了一个可学习的内容自适应前端，用于音频信号处理。在深度学习的现代出现之前，我们使用固定表示的、不可学习的前端，如谱图或梅尔谱图，带/不带神经结构。随着卷积架构支持ASR和声学场景理解等各种应用，转向可学习前端，即从头开始学习和优化特定任务所需的基础函数和权重。在没有卷积块的变形器架构中，线性层将小的波形块投影到小的潜在维度上，然后将它们馈送到变形器架构中。在这项工作中，我们提出了一种计算内容自适应学习时频表示的方法。

    We propose a learnable content adaptive front end for audio signal processing. Before the modern advent of deep learning, we used fixed representation non-learnable front-ends like spectrogram or mel-spectrogram with/without neural architectures. With convolutional architectures supporting various applications such as ASR and acoustic scene understanding, a shift to a learnable front ends occurred in which both the type of basis functions and the weight were learned from scratch and optimized for the particular task of interest. With the shift to transformer-based architectures with no convolutional blocks present, a linear layer projects small waveform patches onto a small latent dimension before feeding them to a transformer architecture. In this work, we propose a way of computing a content-adaptive learnable time-frequency representation. We pass each audio signal through a bank of convolutional filters, each giving a fixed-dimensional vector. It is akin to learning a bank of finit
    
[^2]: 一种使用Transformer结构并利用百万级样本上下文进行原始音频的语言模型

    A Language Model With Million Sample Context For Raw Audio Using Transformer Architectures. (arXiv:2206.08297v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2206.08297](http://arxiv.org/abs/2206.08297)

    本文提出了一种采用Transformer结构和百万级样本上下文进行原始音频语言模型的自回归生成架构，能够高效地建模音频信号的长期依赖性，并取得了最先进的性能表现。

    

    对于音频信号进行长期依赖性建模是一个特别具有挑战性的问题，因为即使在小的时间尺度上，也会产生数十万个样本。最近，随着Transformer的出现，神经结构变得擅长于对长期依赖性建模，但它们受到二次约束的影响。我们提出了一种生成自回归架构，可以模拟相当大的上下文超过500,000个样本的音频波形。我们的工作通过使用CNN前端来学习潜在表示，然后使用Transformer编码器在这些表示之上学习依赖项，完全端对端地进行了训练：从而允许它根据下一个样本自行学习表示。与以前用不同的时间尺度进行比较以展示改进的作品不同，我们使用标准数据集，并使用相同数目的参数/上下文显示了改进。我们实现了最先进的性能。

    Modeling long-term dependencies for audio signals is a particularly challenging problem, as even small-time scales yield on the order of a hundred thousand samples. With the recent advent of Transformers, neural architectures became good at modeling dependencies over longer time scales, but they suffered from quadratic constraints to scale them. We propose a generative auto-regressive architecture that can model audio waveforms over quite a large context, greater than 500,000 samples. Our work is adapted to learn time dependencies by learning a latent representation by a CNN front-end, and then learning dependencies over these representations using Transformer encoders, fully trained end-to-end: thereby allowing to learn representations as it deems fit for the next sample. Unlike previous works that compared different time scales to show improvement, we use a standard dataset, with the same number of parameters/context to show improvements. We achieve a state-of-the-art performance as 
    

