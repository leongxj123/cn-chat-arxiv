# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Human-Centered Dynamic Scene Understanding via Multiple LLMs Collaborated Reasoning](https://arxiv.org/abs/2403.10107) | 通过多个大型预训练语言模型的合作推理，本研究提出了V-HOI Multi-LLMs Collaborated Reasoning（V-HOI MLCR）框架，用于增强当前V-HOI检测模型的性能。 |
| [^2] | [WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research.](http://arxiv.org/abs/2303.17395) | 本文介绍了第一个大规模的弱标注音频字幕数据集WavCaps，含约40万条带有配对字幕的音频剪辑。为克服噪声标注的问题，提出了基于ChatGPT的三阶段字幕生成流程。 |

# 详细

[^1]: 通过多个LLM合作推理提升人类中心动态场景理解

    Enhancing Human-Centered Dynamic Scene Understanding via Multiple LLMs Collaborated Reasoning

    [https://arxiv.org/abs/2403.10107](https://arxiv.org/abs/2403.10107)

    通过多个大型预训练语言模型的合作推理，本研究提出了V-HOI Multi-LLMs Collaborated Reasoning（V-HOI MLCR）框架，用于增强当前V-HOI检测模型的性能。

    

    人类中心的动态场景理解在增强机器人和自主系统的能力中起着至关重要的作用，其中视频人-物交互（V-HOI）检测是语义场景理解中的关键任务，旨在全面理解视频中的HOI关系，以使移动机器人和自动驾驶系统的行为决策受益。虽然先前的V-HOI检测模型在特定数据集上取得了显著进展，但它们仍然缺乏像人类一样的通用推理能力，无法有效引导HOI关系。在本研究中，我们提出了V-HOI多LLM协同推理（V-HOI MLCR），这是一个新颖的框架，由一系列即插即用的模块组成，可以通过利用不同现成大型预训练语言模型（LLMs）的强大推理能力，促进当前V-HOI检测模型的性能。

    arXiv:2403.10107v1 Announce Type: cross  Abstract: Human-centered dynamic scene understanding plays a pivotal role in enhancing the capability of robotic and autonomous systems, in which Video-based Human-Object Interaction (V-HOI) detection is a crucial task in semantic scene understanding, aimed at comprehensively understanding HOI relationships within a video to benefit the behavioral decisions of mobile robots and autonomous driving systems. Although previous V-HOI detection models have made significant strides in accurate detection on specific datasets, they still lack the general reasoning ability like human beings to effectively induce HOI relationships. In this study, we propose V-HOI Multi-LLMs Collaborated Reasoning (V-HOI MLCR), a novel framework consisting of a series of plug-and-play modules that could facilitate the performance of current V-HOI detection models by leveraging the strong reasoning ability of different off-the-shelf pre-trained large language models (LLMs). 
    
[^2]: WavCaps: 一种ChatGPT辅助的弱标注音频字幕数据集，用于音频-语言多模态研究

    WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research. (arXiv:2303.17395v1 [eess.AS])

    [http://arxiv.org/abs/2303.17395](http://arxiv.org/abs/2303.17395)

    本文介绍了第一个大规模的弱标注音频字幕数据集WavCaps，含约40万条带有配对字幕的音频剪辑。为克服噪声标注的问题，提出了基于ChatGPT的三阶段字幕生成流程。

    

    近年来，音频-语言（AL）多模态学习任务的发展非常显著。然而，现有的AL数据集收集过程昂贵费时，规模有限，给研究者带来了挑战。为解决这个数据稀缺问题，我们介绍了WavCaps，这是第一个包含大约40万条带有配对字幕的大规模弱标注音频字幕数据集。我们从Web资源和声音事件检测数据集中获取音频剪辑及原始描述。但是，在线收集到的原始描述非常嘈杂，不适合用于自动化音频字幕等任务。为了克服这个问题，我们提出了一个三阶段的处理流程，以过滤嘈杂数据并生成高质量字幕，在其中利用了ChatGPT，一种大型语言模型，来自动过滤和转换原始描述。我们对WavCaps的特征进行了全面的分析。

    The advancement of audio-language (AL) multimodal learning tasks has been significant in recent years. However, researchers face challenges due to the costly and time-consuming collection process of existing audio-language datasets, which are limited in size. To address this data scarcity issue, we introduce WavCaps, the first large-scale weakly-labelled audio captioning dataset, comprising approximately 400k audio clips with paired captions. We sourced audio clips and their raw descriptions from web sources and a sound event detection dataset. However, the online-harvested raw descriptions are highly noisy and unsuitable for direct use in tasks such as automated audio captioning. To overcome this issue, we propose a three-stage processing pipeline for filtering noisy data and generating high-quality captions, where ChatGPT, a large language model, is leveraged to filter and transform raw descriptions automatically. We conduct a comprehensive analysis of the characteristics of WavCaps 
    

