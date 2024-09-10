# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multimodal Speech Enhancement Using Burst Propagation](https://arxiv.org/abs/2209.03275) | 本论文提出了一种采用爆发传播的多模态语音增强解决方案，通过学习噪声信号和视觉刺激之间的相关性，放大相关信息并抑制噪声，从而赋予语音含义。 |

# 详细

[^1]: 采用爆发传播的多模态语音增强

    Multimodal Speech Enhancement Using Burst Propagation

    [https://arxiv.org/abs/2209.03275](https://arxiv.org/abs/2209.03275)

    本论文提出了一种采用爆发传播的多模态语音增强解决方案，通过学习噪声信号和视觉刺激之间的相关性，放大相关信息并抑制噪声，从而赋予语音含义。

    

    本论文提出了一种名为MBURST的新颖的多模态解决方案，用于音频-视觉语音增强，并考虑了有关前额叶皮层和其他脑区金字塔细胞的最新神经学发现。所谓的爆发传播通过反馈方式实现了几个准则，以更符合生物学的方式解决信任分配问题：通过反馈控制塑性的符号和幅度，通过不同的权重连接在各层之间多路复用反馈和前馈信息，近似反馈和前馈连接，并线性化反馈信号。MBURST利用这些功能学习噪声信号和视觉刺激之间的相关性，从而通过放大相关信息和抑制噪声赋予语音以含义。在Grid Corpus和基于CHiME3的数据集上进行的实验表明，MBURST能够复现类似的掩模重建，与多模态反向传播基准方法相比。

    This paper proposes the MBURST, a novel multimodal solution for audio-visual speech enhancements that consider the most recent neurological discoveries regarding pyramidal cells of the prefrontal cortex and other brain regions. The so-called burst propagation implements several criteria to address the credit assignment problem in a more biologically plausible manner: steering the sign and magnitude of plasticity through feedback, multiplexing the feedback and feedforward information across layers through different weight connections, approximating feedback and feedforward connections, and linearizing the feedback signals. MBURST benefits from such capabilities to learn correlations between the noisy signal and the visual stimuli, thus attributing meaning to the speech by amplifying relevant information and suppressing noise. Experiments conducted over a Grid Corpus and CHiME3-based dataset show that MBURST can reproduce similar mask reconstructions to the multimodal backpropagation-bas
    

