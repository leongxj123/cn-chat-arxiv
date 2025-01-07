# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prompt-Singer: Controllable Singing-Voice-Synthesis with Natural Language Prompt](https://arxiv.org/abs/2403.11780) | 提出了Prompt-Singer，这是第一个能够用自然语言控制歌手性别、音域和音量的唱歌声音合成方法，采用了基于解码器的变压器模型架构和范围旋律解耦的音高表示方法。 |

# 详细

[^1]: Prompt-Singer: 带自然语言提示的可控唱歌声音合成

    Prompt-Singer: Controllable Singing-Voice-Synthesis with Natural Language Prompt

    [https://arxiv.org/abs/2403.11780](https://arxiv.org/abs/2403.11780)

    提出了Prompt-Singer，这是第一个能够用自然语言控制歌手性别、音域和音量的唱歌声音合成方法，采用了基于解码器的变压器模型架构和范围旋律解耦的音高表示方法。

    

    近期的唱歌声音合成(SVS)方法取得了显著的音频质量和自然度，然而它们缺乏显式控制合成唱歌风格属性的能力。我们提出Prompt-Singer，这是第一个能够用自然语言控制歌手性别、音域和音量的SVS方法。我们采用基于仅解码器的变压器模型架构，具有多尺度层次结构，并设计了一个分离音高表示的范围旋律解耦的方法，从而实现了基于文本的音域控制同时保持了旋律准确性。此外，我们探索了各种实验设置，包括不同类型的文本表示，文本编码器微调，以及引入语音数据以减轻数据稀缺性，旨在促进进一步研究。实验证明，我们的模型具有良好的控制能力和音频质量。音频示例可访问 http://prompt-singer.

    arXiv:2403.11780v1 Announce Type: cross  Abstract: Recent singing-voice-synthesis (SVS) methods have achieved remarkable audio quality and naturalness, yet they lack the capability to control the style attributes of the synthesized singing explicitly. We propose Prompt-Singer, the first SVS method that enables attribute controlling on singer gender, vocal range and volume with natural language. We adopt a model architecture based on a decoder-only transformer with a multi-scale hierarchy, and design a range-melody decoupled pitch representation that enables text-conditioned vocal range control while keeping melodic accuracy. Furthermore, we explore various experiment settings, including different types of text representations, text encoder fine-tuning, and introducing speech data to alleviate data scarcity, aiming to facilitate further research. Experiments show that our model achieves favorable controlling ability and audio quality. Audio samples are available at http://prompt-singer.
    

