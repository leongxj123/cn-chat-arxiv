# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prompt-Singer: Controllable Singing-Voice-Synthesis with Natural Language Prompt](https://arxiv.org/abs/2403.11780) | 提出了Prompt-Singer，这是第一个能够用自然语言控制歌手性别、音域和音量的唱歌声音合成方法，采用了基于解码器的变压器模型架构和范围旋律解耦的音高表示方法。 |
| [^2] | [Human Brain Exhibits Distinct Patterns When Listening to Fake Versus Real Audio: Preliminary Evidence](https://arxiv.org/abs/2402.14982) | 人类大脑对真实和虚假音频有不同的反应模式，与深度伪造音频检测算法不同，这为深度伪造音频检测等领域的未来研究方向提供了重要的初步证据。 |
| [^3] | [A comparative study of Grid and Natural sentences effects on Normal-to-Lombard conversion.](http://arxiv.org/abs/2309.10485) | 本文通过比较Grid句子和自然句子在Lombard效应和Normal-to-Lombard转换方面的表现，发现随着噪声水平的增加，Grid句子的alpha比例增加更大。在实验中，基于EMALG训练的StarGAN模型在主观可懂度评估中一致表现优于其他模型。 |

# 详细

[^1]: Prompt-Singer: 带自然语言提示的可控唱歌声音合成

    Prompt-Singer: Controllable Singing-Voice-Synthesis with Natural Language Prompt

    [https://arxiv.org/abs/2403.11780](https://arxiv.org/abs/2403.11780)

    提出了Prompt-Singer，这是第一个能够用自然语言控制歌手性别、音域和音量的唱歌声音合成方法，采用了基于解码器的变压器模型架构和范围旋律解耦的音高表示方法。

    

    近期的唱歌声音合成(SVS)方法取得了显著的音频质量和自然度，然而它们缺乏显式控制合成唱歌风格属性的能力。我们提出Prompt-Singer，这是第一个能够用自然语言控制歌手性别、音域和音量的SVS方法。我们采用基于仅解码器的变压器模型架构，具有多尺度层次结构，并设计了一个分离音高表示的范围旋律解耦的方法，从而实现了基于文本的音域控制同时保持了旋律准确性。此外，我们探索了各种实验设置，包括不同类型的文本表示，文本编码器微调，以及引入语音数据以减轻数据稀缺性，旨在促进进一步研究。实验证明，我们的模型具有良好的控制能力和音频质量。音频示例可访问 http://prompt-singer.

    arXiv:2403.11780v1 Announce Type: cross  Abstract: Recent singing-voice-synthesis (SVS) methods have achieved remarkable audio quality and naturalness, yet they lack the capability to control the style attributes of the synthesized singing explicitly. We propose Prompt-Singer, the first SVS method that enables attribute controlling on singer gender, vocal range and volume with natural language. We adopt a model architecture based on a decoder-only transformer with a multi-scale hierarchy, and design a range-melody decoupled pitch representation that enables text-conditioned vocal range control while keeping melodic accuracy. Furthermore, we explore various experiment settings, including different types of text representations, text encoder fine-tuning, and introducing speech data to alleviate data scarcity, aiming to facilitate further research. Experiments show that our model achieves favorable controlling ability and audio quality. Audio samples are available at http://prompt-singer.
    
[^2]: 人类大脑在听取真实和虚假音频时展现出不同模式：初步证据

    Human Brain Exhibits Distinct Patterns When Listening to Fake Versus Real Audio: Preliminary Evidence

    [https://arxiv.org/abs/2402.14982](https://arxiv.org/abs/2402.14982)

    人类大脑对真实和虚假音频有不同的反应模式，与深度伪造音频检测算法不同，这为深度伪造音频检测等领域的未来研究方向提供了重要的初步证据。

    

    本文研究了人类听取真实和虚假音频时大脑活动的变化。我们的初步结果表明，一种最先进的深度伪造音频检测算法所学习的表示，并没有显示出真实和虚假音频之间的清晰不同模式。相反，人类大脑活动，通过 EEG 测量，在个体接触虚假与真实音频时显示出不同的模式。这些初步证据为未来在深度伪造音频检测等领域提供了研究方向。

    arXiv:2402.14982v1 Announce Type: cross  Abstract: In this paper we study the variations in human brain activity when listening to real and fake audio. Our preliminary results suggest that the representations learned by a state-of-the-art deepfake audio detection algorithm, do not exhibit clear distinct patterns between real and fake audio. In contrast, human brain activity, as measured by EEG, displays distinct patterns when individuals are exposed to fake versus real audio. This preliminary evidence enables future research directions in areas such as deepfake audio detection.
    
[^3]: Grid和自然语句对Normal-to-Lombard转换的比较研究

    A comparative study of Grid and Natural sentences effects on Normal-to-Lombard conversion. (arXiv:2309.10485v1 [cs.SD])

    [http://arxiv.org/abs/2309.10485](http://arxiv.org/abs/2309.10485)

    本文通过比较Grid句子和自然句子在Lombard效应和Normal-to-Lombard转换方面的表现，发现随着噪声水平的增加，Grid句子的alpha比例增加更大。在实验中，基于EMALG训练的StarGAN模型在主观可懂度评估中一致表现优于其他模型。

    

    Grid句子常用于研究Lombard效应和Normal-to-Lombard转换。然而，目前尚不清楚在真实应用中，基于Grid句子训练的Normal-to-Lombard模型是否足以提高自然语音可懂度。本文介绍了一个平行的Lombard语料库（称为Lombard Chinese TIMIT，LCT），并从中提取了中文TIMIT的自然句子。然后，我们使用LCT和Enhanced Mandarin Lombard Grid语料库（EMALG）比较了自然句子和Grid句子在Lombard效应和Normal-to-Lombard转换方面。通过对Lombard效应的参数分析，我们发现随着噪声水平的增加，自然句子和Grid句子的参数变化相似，但在alpha比例增加方面，Grid句子的增加更大。在跨性别和信噪比的主观可懂度评估中，基于EMALG训练的StarGAN模型始终表现优于其他模型。

    Grid sentence is commonly used for studying the Lombard effect and Normal-to-Lombard conversion. However, it's unclear if Normal-to-Lombard models trained on grid sentences are sufficient for improving natural speech intelligibility in real-world applications. This paper presents the recording of a parallel Lombard corpus (called Lombard Chinese TIMIT, LCT) extracting natural sentences from Chinese TIMIT. Then We compare natural and grid sentences in terms of Lombard effect and Normal-to-Lombard conversion using LCT and Enhanced MAndarin Lombard Grid corpus (EMALG). Through a parametric analysis of the Lombard effect, We find that as the noise level increases, both natural sentences and grid sentences exhibit similar changes in parameters, but in terms of the increase of the alpha ratio, grid sentences show a greater increase. Following a subjective intelligibility assessment across genders and Signal-to-Noise Ratios, the StarGAN model trained on EMALG consistently outperforms the mode
    

