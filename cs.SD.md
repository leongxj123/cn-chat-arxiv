# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuroVoz: a Castillian Spanish corpus of parkinsonian speech](https://arxiv.org/abs/2403.02371) | 这一研究提出了一个包含108位母语为卡斯蒂利亚语说话者的帕金森病患者语音语料库，涵盖了多种语音任务，通过手动和自动转录确保了数据的准确性和可靠性。 |
| [^2] | [Not All Weights Are Created Equal: Enhancing Energy Efficiency in On-Device Streaming Speech Recognition](https://arxiv.org/abs/2402.13076) | 权重参数在设备上的流式语音识别模型中的功耗影响有所不同，作者提出了基于权重参数敏感性的有针对性压缩方法，将能源使用减少高达47%而维持模型准确性 |
| [^3] | [Mixture Encoder Supporting Continuous Speech Separation for Meeting Recognition.](http://arxiv.org/abs/2309.08454) | 本研究将混合编码器方法从两个说话人情况扩展到了更自然的会议环境，包括任意数量的说话人和动态重叠。实验证明，该方法在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。 |

# 详细

[^1]: NeuroVoz：帕金森病患者语音的卡斯蒂利亚语语料库

    NeuroVoz: a Castillian Spanish corpus of parkinsonian speech

    [https://arxiv.org/abs/2403.02371](https://arxiv.org/abs/2403.02371)

    这一研究提出了一个包含108位母语为卡斯蒂利亚语说话者的帕金森病患者语音语料库，涵盖了多种语音任务，通过手动和自动转录确保了数据的准确性和可靠性。

    

    通过语音分析进行帕金森病（PD）诊断的进展受到公开可用、多样化的语言数据集的显著缺乏的阻碍，限制了现有研究结果的可再现性和进一步探索。为了弥补这一空白，我们引入了一个全面的语料库，包括来自108位母语为卡斯蒂利亚语的说话者，包括55名健康对照组和53名被诊断患有PD的个体，所有这些个体都在药物治疗下，并且在药物优化状态下进行记录。 这一独特数据集涵盖了广泛的语音任务，包括持续发音五个西班牙元音、发音测试、16个听后重复的话语以及自由独白。该数据集通过专家手动转录听后重复任务强调准确性和可靠性，并利用Whisper进行自动独白转录，使其成为帕金森病患者语音的最完整的公开语料库。

    arXiv:2403.02371v1 Announce Type: cross  Abstract: The advancement of Parkinson's Disease (PD) diagnosis through speech analysis is hindered by a notable lack of publicly available, diverse language datasets, limiting the reproducibility and further exploration of existing research.   In response to this gap, we introduce a comprehensive corpus from 108 native Castilian Spanish speakers, comprising 55 healthy controls and 53 individuals diagnosed with PD, all of whom were under pharmacological treatment and recorded in their medication-optimized state. This unique dataset features a wide array of speech tasks, including sustained phonation of the five Spanish vowels, diadochokinetic tests, 16 listen-and-repeat utterances, and free monologues. The dataset emphasizes accuracy and reliability through specialist manual transcriptions of the listen-and-repeat tasks and utilizes Whisper for automated monologue transcriptions, making it the most complete public corpus of Parkinsonian speech, 
    
[^2]: 不是所有的权重都是平等的: 在设备上增强能效的流式语音识别

    Not All Weights Are Created Equal: Enhancing Energy Efficiency in On-Device Streaming Speech Recognition

    [https://arxiv.org/abs/2402.13076](https://arxiv.org/abs/2402.13076)

    权重参数在设备上的流式语音识别模型中的功耗影响有所不同，作者提出了基于权重参数敏感性的有针对性压缩方法，将能源使用减少高达47%而维持模型准确性

    

    电力消耗在设备上的流式语音识别中起着重要作用，因为它直接影响用户体验。本研究深入探讨了语音识别模型中的权重参数如何影响这些模型的总体功耗。我们发现权重参数对功耗的影响因多种因素而异，受到调用频率及其在内存中的位置等因素的影响。凭借这一洞察力，我们制定了旨在优化设备上语音识别模型的设计指南。这些指南侧重于在尽量不显著影响准确性的情况下最小化功耗。我们的方法，基于权重参数变化敏感性的有针对性压缩，表现出优越性能，相比最先进的压缩方法，可以实现高达47%的能源使用减少，同时保持类似的模型准确性，并改善实时流

    arXiv:2402.13076v1 Announce Type: cross  Abstract: Power consumption plays an important role in on-device streaming speech recognition, as it has a direct impact on the user experience. This study delves into how weight parameters in speech recognition models influence the overall power consumption of these models. We discovered that the impact of weight parameters on power consumption varies, influenced by factors including how often they are invoked and their placement in memory. Armed with this insight, we developed design guidelines aimed at optimizing on-device speech recognition models. These guidelines focus on minimizing power use without substantially affecting accuracy. Our method, which employs targeted compression based on the varying sensitivities of weight parameters, demonstrates superior performance compared to state-of-the-art compression methods. It achieves a reduction in energy usage of up to 47% while maintaining similar model accuracy and improving the real-time f
    
[^3]: 混合编码器支持连续语音分离用于会议识别

    Mixture Encoder Supporting Continuous Speech Separation for Meeting Recognition. (arXiv:2309.08454v1 [eess.AS])

    [http://arxiv.org/abs/2309.08454](http://arxiv.org/abs/2309.08454)

    本研究将混合编码器方法从两个说话人情况扩展到了更自然的会议环境，包括任意数量的说话人和动态重叠。实验证明，该方法在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。

    

    自动语音识别（ASR）的许多实际应用需要处理重叠的语音。一种常见的方法是首先将语音分离成无重叠的流，然后对生成的信号进行ASR。最近，提出了在ASR模型中包含混合编码器的方法。该混合编码器利用原始重叠的语音来减轻语音分离引入的伪影效果。然而，先前的方法仅针对两个说话人的情况。在这项工作中，我们将这种方法扩展到更自然的会议环境，包括任意数量的说话人和动态重叠。我们使用不同的语音分离器（包括强大的TF-GridNet模型）评估性能。实验证明，在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。此外，实验还展示了TF-GridNet的强大分离能力，大大缩小了先前方法的差距。

    Many real-life applications of automatic speech recognition (ASR) require processing of overlapped speech. A commonmethod involves first separating the speech into overlap-free streams and then performing ASR on the resulting signals. Recently, the inclusion of a mixture encoder in the ASR model has been proposed. This mixture encoder leverages the original overlapped speech to mitigate the effect of artifacts introduced by the speech separation. Previously, however, the method only addressed two-speaker scenarios. In this work, we extend this approach to more natural meeting contexts featuring an arbitrary number of speakers and dynamic overlaps. We evaluate the performance using different speech separators, including the powerful TF-GridNet model. Our experiments show state-of-the-art performance on the LibriCSS dataset and highlight the advantages of the mixture encoder. Furthermore, they demonstrate the strong separation of TF-GridNet which largely closes the gap between previous m
    

