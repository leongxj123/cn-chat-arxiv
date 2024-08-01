# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models.](http://arxiv.org/abs/2310.08753) | CompA提出了由两个专家注释的音频-语言模型组合推理基准数据集，用于评估ALMs在理解音频中声音事件的顺序和属性绑定方面的表现。 |

# 详细

[^1]: CompA: 解决音频-语言模型中的组合推理差距

    CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models. (arXiv:2310.08753v1 [cs.SD])

    [http://arxiv.org/abs/2310.08753](http://arxiv.org/abs/2310.08753)

    CompA提出了由两个专家注释的音频-语言模型组合推理基准数据集，用于评估ALMs在理解音频中声音事件的顺序和属性绑定方面的表现。

    

    音频的基本特性是其组合性。使用对比方法（例如CLAP）训练的音频-语言模型（ALMs）能够学习音频和语言模态之间的共享表示，从而在许多下游应用中提高性能，包括零样本音频分类、音频检索等。然而，这些模型在有效执行组合推理方面的能力还很少被探索，需要进一步的研究。本文提出了CompA，这是一个由两个专家注释的基准数据集，其中大多数是真实世界的音频样本，用于评估ALMs的组合推理能力。我们的CompA-order评估ALMs在理解音频中声音事件的顺序或发生时的表现如何，而CompA-attribute评估声音事件的属性绑定。每个基准数据集中的实例包含两个音频-标题对，其中两个音频具有相同的声音事件，但组合方式不同。

    A fundamental characteristic of audio is its compositional nature. Audio-language models (ALMs) trained using a contrastive approach (e.g., CLAP) that learns a shared representation between audio and language modalities have improved performance in many downstream applications, including zero-shot audio classification, audio retrieval, etc. However, the ability of these models to effectively perform compositional reasoning remains largely unexplored and necessitates additional research. In this paper, we propose CompA, a collection of two expert-annotated benchmarks with a majority of real-world audio samples, to evaluate compositional reasoning in ALMs. Our proposed CompA-order evaluates how well an ALM understands the order or occurrence of acoustic events in audio, and CompA-attribute evaluates attribute binding of acoustic events. An instance from either benchmark consists of two audio-caption pairs, where both audios have the same acoustic events but with different compositions. A
    

