# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Computational Analysis of Lyric Similarity Perception](https://arxiv.org/abs/2404.02342) | 该研究通过比较分析计算方法对模拟歌词相似度与人类感知的关联，发现基于BERT模型嵌入、歌词音频和音素组件相似性的计算模型对感知上的歌词相似度具有指示作用。 |
| [^2] | [OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification](https://arxiv.org/abs/2402.12654) | 提出了OWSM-CTC，这是一种基于Connectionist Temporal Classification的新型仅编码器语音基础模型，训练有180k小时的公共音频数据，用于多语言自动语音识别（ASR）、语音翻译（ST）和语言识别。 |
| [^3] | [Unified Speech-Text Pretraining for Spoken Dialog Modeling](https://arxiv.org/abs/2402.05706) | 本研究提出了一个名为统一口语对话模型（USDM）的广泛语音文本模型框架，用于生成与输入语音相关的连贯口语回复。通过使用多步骤的语音文本推理方式和广义语音文本预训练方案，该方法能够有效捕捉跨模态语义，并生成自然流畅的口语回复。 |

# 详细

[^1]: 歌词相似度感知的计算分析

    A Computational Analysis of Lyric Similarity Perception

    [https://arxiv.org/abs/2404.02342](https://arxiv.org/abs/2404.02342)

    该研究通过比较分析计算方法对模拟歌词相似度与人类感知的关联，发现基于BERT模型嵌入、歌词音频和音素组件相似性的计算模型对感知上的歌词相似度具有指示作用。

    

    在包含人声的音乐作品中，歌词对艺术表达起着重要作用。因此，先前的研究引入了推荐系统的概念，该系统建议类似于用户喜爱或个性化偏好的歌词，有助于在数百万音轨中发现歌词。然而，许多系统并未充分考虑人类对歌词相似度的感知，主要是由于该领域的研究有限。为弥补这一差距，我们进行了对计算方法建模歌词相似度与人类感知进行了比较分析。结果表明，基于预训练的BERT模型嵌入之间的相似性、歌词来源的音频以及音素组件的计算模型指示了感知上的歌词相似度。该发现强调了语义、风格和音韵相似性在人类感知中的重要性。

    arXiv:2404.02342v1 Announce Type: new  Abstract: In musical compositions that include vocals, lyrics significantly contribute to artistic expression. Consequently, previous studies have introduced the concept of a recommendation system that suggests lyrics similar to a user's favorites or personalized preferences, aiding in the discovery of lyrics among millions of tracks. However, many of these systems do not fully consider human perceptions of lyric similarity, primarily due to limited research in this area. To bridge this gap, we conducted a comparative analysis of computational methods for modeling lyric similarity with human perception. Results indicated that computational models based on similarities between embeddings from pre-trained BERT-based models, the audio from which the lyrics are derived, and phonetic components are indicative of perceptual lyric similarity. This finding underscores the importance of semantic, stylistic, and phonetic similarities in human perception abo
    
[^2]: OWSM-CTC:一种用于语音识别、翻译和语言识别的开放编码器基础模型

    OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification

    [https://arxiv.org/abs/2402.12654](https://arxiv.org/abs/2402.12654)

    提出了OWSM-CTC，这是一种基于Connectionist Temporal Classification的新型仅编码器语音基础模型，训练有180k小时的公共音频数据，用于多语言自动语音识别（ASR）、语音翻译（ST）和语言识别。

    

    近来对能够在单个模型中执行多个语音处理任务的大型语音模型越来越感兴趣。这些模型通常采用编码器-解码器或仅解码器架构，因为它们在许多领域中非常流行且性能良好。然而，与非自回归模型相比，自回归模型在推断时可能会比较慢，并且还存在幻觉的潜在风险。尽管先前的研究观察到非自回归模型在小规模任务中产生了令人满意的结果，但尚不清楚它们是否可以扩展到不同语言和任务的语音转文本生成中。受Open Whisper-style Speech Model (OWSM)项目的启发，我们提出了OWSM-CTC，这是一种基于Connectionist Temporal Classification (CTC)的新型仅编码器的语音基础模型。它使用18万小时的公共音频数据进行训练，用于多语言自动语音识别（ASR）、语音翻译（ST）和语言识别。

    arXiv:2402.12654v1 Announce Type: new  Abstract: There has been an increasing interest in large speech models that can perform multiple speech processing tasks in a single model. Such models usually adopt the encoder-decoder or decoder-only architecture due to their popularity and good performance in many domains. However, autoregressive models can be slower during inference compared to non-autoregressive models and also have potential risks of hallucination. Though prior studies observed promising results of non-autoregressive models for certain tasks at small scales, it remains unclear if they can be scaled to speech-to-text generation in diverse languages and tasks. Inspired by the Open Whisper-style Speech Model (OWSM) project, we propose OWSM-CTC, a novel encoder-only speech foundation model based on Connectionist Temporal Classification (CTC). It is trained on 180k hours of public audio data for multilingual automatic speech recognition (ASR), speech translation (ST), and languag
    
[^3]: 面向口语对话建模的统一语音文本预训练方法

    Unified Speech-Text Pretraining for Spoken Dialog Modeling

    [https://arxiv.org/abs/2402.05706](https://arxiv.org/abs/2402.05706)

    本研究提出了一个名为统一口语对话模型（USDM）的广泛语音文本模型框架，用于生成与输入语音相关的连贯口语回复。通过使用多步骤的语音文本推理方式和广义语音文本预训练方案，该方法能够有效捕捉跨模态语义，并生成自然流畅的口语回复。

    

    近期的研究表明，扩展大型语言模型（LLM）以直接理解和合成语音具有良好的结果，但用于口语对话建模的基于LLM的策略仍然难以实现，需要进一步研究。本文提出了一个广泛的语音文本LLM框架，命名为统一口语对话模型（USDM），以在不依赖于自动语音识别（ASR）或文本到语音（TTS）解决方案的情况下生成与给定输入语音相关的连贯口语回复和有机的韵律特征。我们的方法采用了一种多步骤的语音文本推理方式，利用了底层LLM所展示的推理链能力。我们还提出了一种广义的语音文本预训练方案，有助于捕捉跨模态语义。自动和人工评估结果表明，所提出的方法能够有效生成自然流畅的口语回复，并且优于之前的和级联的基线模型。详细的比较研究

    While recent work shows promising results in expanding the capabilities of large language models (LLM) to directly understand and synthesize speech, an LLM-based strategy for modeling spoken dialogs remains elusive and calls for further investigation. This work proposes an extensive speech-text LLM framework, named the Unified Spoken Dialog Model (USDM), to generate coherent spoken responses with organic prosodic features relevant to the given input speech without relying on automatic speech recognition (ASR) or text-to-speech (TTS) solutions. Our approach employs a multi-step speech-text inference scheme that leverages chain-of-reasoning capabilities exhibited by the underlying LLM. We also propose a generalized speech-text pretraining scheme that helps with capturing cross-modal semantics. Automatic and human evaluations show that the proposed approach is effective in generating natural-sounding spoken responses, outperforming both prior and cascaded baselines. Detailed comparative s
    

