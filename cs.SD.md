# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [WavLLM: Towards Robust and Adaptive Speech Large Language Model](https://arxiv.org/abs/2404.00656) | WavLLM是一个稳健和自适应语音大语言模型，引入了双编码器和Prompt-aware LoRA权重适配器，通过两阶段课程学习方法优化，解耦不同类型的语音信息，为处理语义内容和说话者身份的独特特征提供了新思路 |

# 详细

[^1]: WavLLM：面向稳健和自适应语音大语言模型

    WavLLM: Towards Robust and Adaptive Speech Large Language Model

    [https://arxiv.org/abs/2404.00656](https://arxiv.org/abs/2404.00656)

    WavLLM是一个稳健和自适应语音大语言模型，引入了双编码器和Prompt-aware LoRA权重适配器，通过两阶段课程学习方法优化，解耦不同类型的语音信息，为处理语义内容和说话者身份的独特特征提供了新思路

    

    近年来，大型语言模型(LLMs)的最新进展彻底改变了自然语言处理领域，逐渐拓宽了它们的范围到多模态感知和生成。然而，有效地将听觉能力整合到LLMs中会带来显著挑战，特别是在泛化跨不同语境和执行复杂听觉任务方面。在这项工作中，我们引入了WavLLM，一个具有双编码器和Prompt-aware LoRA权重适配器的稳健和自适应语音大语言模型，通过两阶段课程学习方法进行优化。利用双编码器，我们解耦不同类型的语音信息，利用Whisper编码器处理语音的语义内容，利用WavLM编码器捕捉说话者身份的独特特征。在课程学习框架内，WavLLM首先通过混合要素进行优化来建立其基础能力

    arXiv:2404.00656v1 Announce Type: cross  Abstract: The recent advancements in large language models (LLMs) have revolutionized the field of natural language processing, progressively broadening their scope to multimodal perception and generation. However, effectively integrating listening capabilities into LLMs poses significant challenges, particularly with respect to generalizing across varied contexts and executing complex auditory tasks. In this work, we introduce WavLLM, a robust and adaptive speech large language model with dual encoders, and a prompt-aware LoRA weight adapter, optimized by a two-stage curriculum learning approach. Leveraging dual encoders, we decouple different types of speech information, utilizing a Whisper encoder to process the semantic content of speech, and a WavLM encoder to capture the unique characteristics of the speaker's identity. Within the curriculum learning framework, WavLLM first builds its foundational capabilities by optimizing on mixed elemen
    

