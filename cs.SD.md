# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Speech-to-Speech Translation with Discrete-Unit-Based Style Transfer.](http://arxiv.org/abs/2309.07566) | 本研究提出了一种基于离散单元的语音到语音翻译框架，通过自监督学习和神经编解码器实现风格转换，解决了数据稀缺和音色保留的问题。实验结果表明，我们的模型在之前未见的语言上实现了高质量的跨语言风格转换。 |
| [^2] | [WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research.](http://arxiv.org/abs/2303.17395) | 本文介绍了第一个大规模的弱标注音频字幕数据集WavCaps，含约40万条带有配对字幕的音频剪辑。为克服噪声标注的问题，提出了基于ChatGPT的三阶段字幕生成流程。 |

# 详细

[^1]: 基于离散单元的风格转换的语音到语音翻译

    Speech-to-Speech Translation with Discrete-Unit-Based Style Transfer. (arXiv:2309.07566v1 [cs.SD])

    [http://arxiv.org/abs/2309.07566](http://arxiv.org/abs/2309.07566)

    本研究提出了一种基于离散单元的语音到语音翻译框架，通过自监督学习和神经编解码器实现风格转换，解决了数据稀缺和音色保留的问题。实验结果表明，我们的模型在之前未见的语言上实现了高质量的跨语言风格转换。

    

    直接的语音到语音翻译（S2ST）通过离散的自监督表示实现了显著的准确性，但在翻译过程中无法保留源语音的说话人音色。与此同时，高质量说话人平行数据的稀缺性对于学习源语音和目标语音之间的风格转换构成了挑战。我们提出了一个基于自监督模型的离散单元的声学语言模型和风格转换的神经编解码器的S2ST框架。声学语言模型通过自监督上下文学习获得了风格转换的能力，无需依赖于任何说话人平行数据，从而克服了数据稀缺性问题。通过使用大量的训练数据，我们的模型可以在之前未见过的源语言上实现零-shot跨语言风格转换。实验证明，我们的模型生成的翻译语音具有高度的保真度和风格相似性。

    Direct speech-to-speech translation (S2ST) with discrete self-supervised representations has achieved remarkable accuracy, but is unable to preserve the speaker timbre of the source speech during translation. Meanwhile, the scarcity of high-quality speaker-parallel data poses a challenge for learning style transfer between source and target speech. We propose an S2ST framework with an acoustic language model based on discrete units from a self-supervised model and a neural codec for style transfer. The acoustic language model leverages self-supervised in-context learning, acquiring the ability for style transfer without relying on any speaker-parallel data, thereby overcoming the issue of data scarcity. By using extensive training data, our model achieves zero-shot cross-lingual style transfer on previously unseen source languages. Experiments show that our model generates translated speeches with high fidelity and style similarity. Audio samples are available at this http URL .
    
[^2]: WavCaps: 一种ChatGPT辅助的弱标注音频字幕数据集，用于音频-语言多模态研究

    WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research. (arXiv:2303.17395v1 [eess.AS])

    [http://arxiv.org/abs/2303.17395](http://arxiv.org/abs/2303.17395)

    本文介绍了第一个大规模的弱标注音频字幕数据集WavCaps，含约40万条带有配对字幕的音频剪辑。为克服噪声标注的问题，提出了基于ChatGPT的三阶段字幕生成流程。

    

    近年来，音频-语言（AL）多模态学习任务的发展非常显著。然而，现有的AL数据集收集过程昂贵费时，规模有限，给研究者带来了挑战。为解决这个数据稀缺问题，我们介绍了WavCaps，这是第一个包含大约40万条带有配对字幕的大规模弱标注音频字幕数据集。我们从Web资源和声音事件检测数据集中获取音频剪辑及原始描述。但是，在线收集到的原始描述非常嘈杂，不适合用于自动化音频字幕等任务。为了克服这个问题，我们提出了一个三阶段的处理流程，以过滤嘈杂数据并生成高质量字幕，在其中利用了ChatGPT，一种大型语言模型，来自动过滤和转换原始描述。我们对WavCaps的特征进行了全面的分析。

    The advancement of audio-language (AL) multimodal learning tasks has been significant in recent years. However, researchers face challenges due to the costly and time-consuming collection process of existing audio-language datasets, which are limited in size. To address this data scarcity issue, we introduce WavCaps, the first large-scale weakly-labelled audio captioning dataset, comprising approximately 400k audio clips with paired captions. We sourced audio clips and their raw descriptions from web sources and a sound event detection dataset. However, the online-harvested raw descriptions are highly noisy and unsuitable for direct use in tasks such as automated audio captioning. To overcome this issue, we propose a three-stage processing pipeline for filtering noisy data and generating high-quality captions, where ChatGPT, a large language model, is leveraged to filter and transform raw descriptions automatically. We conduct a comprehensive analysis of the characteristics of WavCaps 
    

