# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BAT: Learning to Reason about Spatial Sounds with Large Language Models](https://rss.arxiv.org/abs/2402.01591) | 本文提出了BAT，它结合了双耳声音场景分析模型的空间声音感知能力和大规模语言模型的自然语言推理能力，以复制人类的空间声音推理能力。通过使用合成的双耳音频数据集和基于空间声音的问答数据集进行训练，BAT在空间声音感知和推理方面取得了强大的性能。 |
| [^2] | [Streaming Sequence Transduction through Dynamic Compression](https://rss.arxiv.org/abs/2402.01172) | STAR是一种新型的Transformer模型，通过动态压缩和优化延迟、内存占用和质量，实现对流的高效序列转导，并在自动语音识别领域表现出色。 |
| [^3] | [WikiMT++ Dataset Card.](http://arxiv.org/abs/2309.13259) | WikiMT++是一个扩展和精细版本的WikiMusicText数据集，包含了1010个经过策划的ABC记谱法的主题曲。它添加了客观属性和主观情感属性，增强了数据集的应用场景和可用性，并通过CLaMP来纠正属性，提高准确性和完整性。 |

# 详细

[^1]: BAT: 使用大规模语言模型学习关于空间声音的推理能力

    BAT: Learning to Reason about Spatial Sounds with Large Language Models

    [https://rss.arxiv.org/abs/2402.01591](https://rss.arxiv.org/abs/2402.01591)

    本文提出了BAT，它结合了双耳声音场景分析模型的空间声音感知能力和大规模语言模型的自然语言推理能力，以复制人类的空间声音推理能力。通过使用合成的双耳音频数据集和基于空间声音的问答数据集进行训练，BAT在空间声音感知和推理方面取得了强大的性能。

    

    空间声音推理是一种基本的人类技能，它使我们能够根据声音来导航和解释我们的周围环境。本文提出了BAT，它将双耳声音场景分析模型的空间声音感知能力与大规模语言模型（LLM）的自然语言推理能力相结合，以复制这种固有能力。为了解决现有野外空间声音数据集的缺乏，我们使用AudioSet和SoundSpaces 2.0合成了一个双耳音频数据集。接下来，我们开发了一种基于空间声音的问答数据集SpatialSoundQA，提供了一系列QA任务，以训练BAT在空间声音感知和推理的各个方面。BAT的声学前端编码器是一种名为Spatial Audio Spectrogram Transformer（Spatial-AST）的创新空间音频编码器，它本身在声音事件检测、空间定位和距离估计等方面具有强大的性能。通过将Spatial-AST与LLaMA-2 7B集成，

    Spatial sound reasoning is a fundamental human skill, enabling us to navigate and interpret our surroundings based on sound. In this paper we present BAT, which combines the spatial sound perception ability of a binaural acoustic scene analysis model with the natural language reasoning capabilities of a large language model (LLM) to replicate this innate ability. To address the lack of existing datasets of in-the-wild spatial sounds, we synthesized a binaural audio dataset using AudioSet and SoundSpaces 2.0. Next, we developed SpatialSoundQA, a spatial sound-based question-answering dataset, offering a range of QA tasks that train BAT in various aspects of spatial sound perception and reasoning. The acoustic front end encoder of BAT is a novel spatial audio encoder named Spatial Audio Spectrogram Transformer, or Spatial-AST, which by itself achieves strong performance across sound event detection, spatial localization, and distance estimation. By integrating Spatial-AST with LLaMA-2 7B
    
[^2]: 流式序列转导通过动态压缩

    Streaming Sequence Transduction through Dynamic Compression

    [https://rss.arxiv.org/abs/2402.01172](https://rss.arxiv.org/abs/2402.01172)

    STAR是一种新型的Transformer模型，通过动态压缩和优化延迟、内存占用和质量，实现对流的高效序列转导，并在自动语音识别领域表现出色。

    

    我们引入了STAR（带有锚定表示的流式转导），这是一种基于Transformer的新型模型，旨在实现对流的高效序列转导。STAR动态地对输入流进行分段，创建压缩的锚定表示，实现近乎无损的压缩（12倍）在自动语音识别（ASR）中，并优于现有方法。此外，STAR在同时进行语音到文本任务中展示出优越的分割和延迟-质量折衷，优化延迟、内存占用和质量。

    We introduce STAR (Stream Transduction with Anchor Representations), a novel Transformer-based model designed for efficient sequence-to-sequence transduction over streams. STAR dynamically segments input streams to create compressed anchor representations, achieving nearly lossless compression (12x) in Automatic Speech Recognition (ASR) and outperforming existing methods. Moreover, STAR demonstrates superior segmentation and latency-quality trade-offs in simultaneous speech-to-text tasks, optimizing latency, memory footprint, and quality.
    
[^3]: WikiMT++数据集卡片

    WikiMT++ Dataset Card. (arXiv:2309.13259v1 [cs.IR])

    [http://arxiv.org/abs/2309.13259](http://arxiv.org/abs/2309.13259)

    WikiMT++是一个扩展和精细版本的WikiMusicText数据集，包含了1010个经过策划的ABC记谱法的主题曲。它添加了客观属性和主观情感属性，增强了数据集的应用场景和可用性，并通过CLaMP来纠正属性，提高准确性和完整性。

    

    WikiMT++是WikiMusicText（WikiMT）的扩展和精细版本，包含了1010个经过策划的ABC记谱法的主题曲。为了扩展WikiMT的应用场景，我们添加了客观属性（专辑、歌词、视频）和主观情感属性（12个情感形容词）和情感4Q（Russell 4Q），增强了其在音乐信息检索、条件音乐生成、自动作曲和情感分类等方面的可用性。此外，我们还实现了CLaMP来纠正从WikiMT继承的属性，以减少原始数据收集过程中引入的错误，增强了数据集的准确性和完整性。

    WikiMT++ is an expanded and refined version of WikiMusicText (WikiMT), featuring 1010 curated lead sheets in ABC notation. To expand application scenarios of WikiMT, we add both objective (album, lyrics, video) and subjective emotion (12 emotion adjectives) and emo\_4q (Russell 4Q) attributes, enhancing its usability for music information retrieval, conditional music generation, automatic composition, and emotion classification, etc. Additionally, CLaMP is implemented to correct the attributes inherited from WikiMT to reduce errors introduced during original data collection and enhance the accuracy and completeness of our dataset.
    

