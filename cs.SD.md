# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unified Speech-Text Pretraining for Spoken Dialog Modeling](https://arxiv.org/abs/2402.05706) | 本研究提出了一个名为统一口语对话模型（USDM）的广泛语音文本模型框架，用于生成与输入语音相关的连贯口语回复。通过使用多步骤的语音文本推理方式和广义语音文本预训练方案，该方法能够有效捕捉跨模态语义，并生成自然流畅的口语回复。 |

# 详细

[^1]: 面向口语对话建模的统一语音文本预训练方法

    Unified Speech-Text Pretraining for Spoken Dialog Modeling

    [https://arxiv.org/abs/2402.05706](https://arxiv.org/abs/2402.05706)

    本研究提出了一个名为统一口语对话模型（USDM）的广泛语音文本模型框架，用于生成与输入语音相关的连贯口语回复。通过使用多步骤的语音文本推理方式和广义语音文本预训练方案，该方法能够有效捕捉跨模态语义，并生成自然流畅的口语回复。

    

    近期的研究表明，扩展大型语言模型（LLM）以直接理解和合成语音具有良好的结果，但用于口语对话建模的基于LLM的策略仍然难以实现，需要进一步研究。本文提出了一个广泛的语音文本LLM框架，命名为统一口语对话模型（USDM），以在不依赖于自动语音识别（ASR）或文本到语音（TTS）解决方案的情况下生成与给定输入语音相关的连贯口语回复和有机的韵律特征。我们的方法采用了一种多步骤的语音文本推理方式，利用了底层LLM所展示的推理链能力。我们还提出了一种广义的语音文本预训练方案，有助于捕捉跨模态语义。自动和人工评估结果表明，所提出的方法能够有效生成自然流畅的口语回复，并且优于之前的和级联的基线模型。详细的比较研究

    While recent work shows promising results in expanding the capabilities of large language models (LLM) to directly understand and synthesize speech, an LLM-based strategy for modeling spoken dialogs remains elusive and calls for further investigation. This work proposes an extensive speech-text LLM framework, named the Unified Spoken Dialog Model (USDM), to generate coherent spoken responses with organic prosodic features relevant to the given input speech without relying on automatic speech recognition (ASR) or text-to-speech (TTS) solutions. Our approach employs a multi-step speech-text inference scheme that leverages chain-of-reasoning capabilities exhibited by the underlying LLM. We also propose a generalized speech-text pretraining scheme that helps with capturing cross-modal semantics. Automatic and human evaluations show that the proposed approach is effective in generating natural-sounding spoken responses, outperforming both prior and cascaded baselines. Detailed comparative s
    

