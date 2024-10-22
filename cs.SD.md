# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SpiRit-LM: Interleaved Spoken and Written Language Model](https://arxiv.org/abs/2402.05755) | SPIRIT-LM是一个基于预训练文本语言模型的多模态语言模型，通过将文本和语音连续训练，实现了口语和书面语言的混合模型。它展示了文本模型的语义能力和语音模型的表现能力。此外，SPIRIT-LM还能以少量样本的方式学习新任务。 |

# 详细

[^1]: SpiRit-LM: 交织的口语和书面语言模型

    SpiRit-LM: Interleaved Spoken and Written Language Model

    [https://arxiv.org/abs/2402.05755](https://arxiv.org/abs/2402.05755)

    SPIRIT-LM是一个基于预训练文本语言模型的多模态语言模型，通过将文本和语音连续训练，实现了口语和书面语言的混合模型。它展示了文本模型的语义能力和语音模型的表现能力。此外，SPIRIT-LM还能以少量样本的方式学习新任务。

    

    我们引入了SPIRIT-LM，这是一个基于文本和语音自由混合的多模态语言模型。我们的模型基于预训练的文本语言模型，并通过连续在文本和语音单元上进行训练将其扩展到语音模态。语音和文本序列被连接为一组单词，并使用一个小型自动筛选的语音-文本平行语料库来进行词级交织的训练方法。SPIRIT-LM有两个版本：一个是使用语音语义单元的BASE版本，另一个是在语义单元之外还使用了音高和风格单元来模拟表现力的EXPRESSIVE版本。对于这两个版本，文本是用子词BPE标记编码的。结果模型展示了文本模型的语义能力和语音模型的表现能力。此外，我们还证明了SPIRIT-LM能够在跨模态（即ASR、TTS、语音分类）中以少量样本的方式学习新任务。

    We introduce SPIRIT-LM, a foundation multimodal language model that freely mixes text and speech. Our model is based on a pretrained text language model that we extend to the speech modality by continuously training it on text and speech units. Speech and text sequences are concatenated as a single set of tokens, and trained with a word-level interleaving method using a small automatically-curated speech-text parallel corpus. SPIRIT-LM comes in two versions: a BASE version that uses speech semantic units and an EXPRESSIVE version that models expressivity using pitch and style units in addition to the semantic units. For both versions, the text is encoded with subword BPE tokens. The resulting model displays both the semantic abilities of text models and the expressive abilities of speech models. Additionally, we demonstrate that SPIRIT-LM is able to learn new tasks in a few-shot fashion across modalities (i.e. ASR, TTS, Speech Classification).
    

