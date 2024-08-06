# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Furnishing Sound Event Detection with Language Model Abilities.](http://arxiv.org/abs/2308.11530) | 本文提出了一种增强声音事件检测的方法，通过对齐音频特征和文本特征来实现声音事件分类和时间定位。该方法利用语言模型的语义能力直接生成序列，相比传统方法更简洁全面，并通过实验证明了其在时间戳捕获和事件分类方面的有效性。 |

# 详细

[^1]: 增强声音事件检测的语言模型能力

    Furnishing Sound Event Detection with Language Model Abilities. (arXiv:2308.11530v1 [cs.SD])

    [http://arxiv.org/abs/2308.11530](http://arxiv.org/abs/2308.11530)

    本文提出了一种增强声音事件检测的方法，通过对齐音频特征和文本特征来实现声音事件分类和时间定位。该方法利用语言模型的语义能力直接生成序列，相比传统方法更简洁全面，并通过实验证明了其在时间戳捕获和事件分类方面的有效性。

    

    最近，语言模型（LMs）在视觉跨模态中的能力引起了越来越多的关注。在本文中，我们进一步探索了LMs在声音事件检测（SED）中的生成能力，超越了视觉领域。具体而言，我们提出了一种优雅的方法，通过对齐音频特征和文本特征来完成声音事件分类和时间定位。该框架由一个声学编码器、一个对应的文本和音频表示对齐的对比模块，以及一个解耦的语言解码器组成，用于从音频特征中生成时间和事件序列。与需要复杂处理并几乎不使用有限音频特征的传统方法相比，我们的模型更简洁全面，因为语言模型直接利用其语义能力生成序列。我们研究了不同的解耦模块，以展示其对时间戳捕捉和事件分类的有效性。

    Recently, the ability of language models (LMs) has attracted increasing attention in visual cross-modality. In this paper, we further explore the generation capacity of LMs for sound event detection (SED), beyond the visual domain. Specifically, we propose an elegant method that aligns audio features and text features to accomplish sound event classification and temporal location. The framework consists of an acoustic encoder, a contrastive module that align the corresponding representations of the text and audio, and a decoupled language decoder that generates temporal and event sequences from the audio characteristic. Compared with conventional works that require complicated processing and barely utilize limited audio features, our model is more concise and comprehensive since language model directly leverage its semantic capabilities to generate the sequences. We investigate different decoupling modules to demonstrate the effectiveness for timestamps capture and event classification
    

