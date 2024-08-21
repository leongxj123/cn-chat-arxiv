# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The FruitShell French synthesis system at the Blizzard 2023 Challenge.](http://arxiv.org/abs/2309.00223) | 本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统，通过对数据的筛选和增强，以及添加词边界和起始/结束符号的方式，提高了语音质量并进行了标准化转录。 |

# 详细

[^1]: FruitShell法语合成系统在Blizzard 2023挑战赛中的应用

    The FruitShell French synthesis system at the Blizzard 2023 Challenge. (arXiv:2309.00223v1 [eess.AS])

    [http://arxiv.org/abs/2309.00223](http://arxiv.org/abs/2309.00223)

    本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统，通过对数据的筛选和增强，以及添加词边界和起始/结束符号的方式，提高了语音质量并进行了标准化转录。

    

    本文介绍了一个用于Blizzard Challenge 2023的法语文本到语音合成系统。该挑战包括两个任务：从女性演讲者生成高质量的语音和生成与特定个体相似的语音。关于比赛数据，我们进行了筛选过程，去除了缺失或错误的文本数据。我们对除音素以外的所有符号进行了整理，并消除了没有发音或持续时间为零的符号。此外，我们还在文本中添加了词边界和起始/结束符号，根据我们之前的经验，我们发现这样可以提高语音质量。对于Spoke任务，我们根据比赛规则进行了数据增强。我们使用了一个开源的G2P模型将法语文本转录为音素。由于G2P模型使用国际音标（IPA），我们对提供的比赛数据应用了相同的转录过程，以进行标准化。然而，由于编译器对某些技术限制的识别能力有限，所以我们为了保持竞争的公正，将数据按音标划分为不同的片段进行评估。

    This paper presents a French text-to-speech synthesis system for the Blizzard Challenge 2023. The challenge consists of two tasks: generating high-quality speech from female speakers and generating speech that closely resembles specific individuals. Regarding the competition data, we conducted a screening process to remove missing or erroneous text data. We organized all symbols except for phonemes and eliminated symbols that had no pronunciation or zero duration. Additionally, we added word boundary and start/end symbols to the text, which we have found to improve speech quality based on our previous experience. For the Spoke task, we performed data augmentation according to the competition rules. We used an open-source G2P model to transcribe the French texts into phonemes. As the G2P model uses the International Phonetic Alphabet (IPA), we applied the same transcription process to the provided competition data for standardization. However, due to compiler limitations in recognizing 
    

