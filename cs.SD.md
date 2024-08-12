# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DurFlex-EVC: Duration-Flexible Emotional Voice Conversion with Parallel Generation](https://arxiv.org/abs/2401.08095) | DurFlex-EVC通过引入风格自编码器和交叉注意力，解决了传统情绪语音转换模型中对语言和语音信息的同步问题。 |

# 详细

[^1]: DurFlex-EVC: 具有并行生成的持续灵活情绪语音转换

    DurFlex-EVC: Duration-Flexible Emotional Voice Conversion with Parallel Generation

    [https://arxiv.org/abs/2401.08095](https://arxiv.org/abs/2401.08095)

    DurFlex-EVC通过引入风格自编码器和交叉注意力，解决了传统情绪语音转换模型中对语言和语音信息的同步问题。

    

    情绪语音转换（EVC）旨在修改说话者声音的情绪色彩，同时保留原始的语言内容和说话者独特的声音特征。最近EVC的进展涉及同时建模音高和持续时间，利用序列到序列（seq2seq）模型的潜力。为了增强转换的可靠性和效率，本研究将重点转向并行语音生成。我们介绍了DurFlex-EVC，它集成了风格自编码器和单元对齐器。传统模型虽然融入了包含语言和语音信息的自监督学习（SSL）表示，但却忽视了这种双重性质，导致了可控性的降低。为了解决这个问题，我们实现了交叉注意力以将这些表示与不同情绪进行同步。此外，我们还开发了一个风格自编码器。

    arXiv:2401.08095v2 Announce Type: replace-cross  Abstract: Emotional voice conversion (EVC) seeks to modify the emotional tone of a speaker's voice while preserving the original linguistic content and the speaker's unique vocal characteristics. Recent advancements in EVC have involved the simultaneous modeling of pitch and duration, utilizing the potential of sequence-to-sequence (seq2seq) models. To enhance reliability and efficiency in conversion, this study shifts focus towards parallel speech generation. We introduce Duration-Flexible EVC (DurFlex-EVC), which integrates a style autoencoder and unit aligner. Traditional models, while incorporating self-supervised learning (SSL) representations that contain both linguistic and paralinguistic information, have neglected this dual nature, leading to reduced controllability. Addressing this issue, we implement cross-attention to synchronize these representations with various emotions. Additionally, a style autoencoder is developed for t
    

