# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Whisper-MCE: Whisper Model Finetuned for Better Performance with Mixed Languages.](http://arxiv.org/abs/2310.17953) | Whisper-MCE是使用自己收集的混合粤语和英语音频数据集（MCE）进行训练的Whisper模型微调，相较于基准模型，其在准确捕捉原始音频内容、提高识别准确性和加快识别速度方面具有更优越的能力，尤其在混合语言识别任务中表现出色。 |

# 详细

[^1]: Whisper-MCE: 针对混合语言实现更好性能的Whisper模型微调

    Whisper-MCE: Whisper Model Finetuned for Better Performance with Mixed Languages. (arXiv:2310.17953v1 [cs.SD])

    [http://arxiv.org/abs/2310.17953](http://arxiv.org/abs/2310.17953)

    Whisper-MCE是使用自己收集的混合粤语和英语音频数据集（MCE）进行训练的Whisper模型微调，相较于基准模型，其在准确捕捉原始音频内容、提高识别准确性和加快识别速度方面具有更优越的能力，尤其在混合语言识别任务中表现出色。

    

    最近，Whisper在英语自动语音识别（ASR）领域已经接近于人类级别的鲁棒性和准确性，但在较小语种和混合语言的语音识别中，仍然需要进一步改进。本文介绍了我们细调的Whisper模型Whisper-MCE的令人瞩目的结果，该模型使用了我们自己收集的混合粤语和英语音频数据集（MCE）进行训练。同时，考虑到词错误率（WER）在较小语种和混合语言环境中评估其有效性时存在挑战，我们提出了一种新颖的评估机制。通过将我们的模型与基准的whisper-large-v2模型进行比较，我们展示了它准确捕捉原始音频内容的能力更强、识别准确性更高、识别速度更快。值得注意的是，我们的模型在识别混合语言的特定任务中胜过其他现有模型。

    Recently Whisper has approached human-level robustness and accuracy in English automatic speech recognition (ASR), while in minor language and mixed language speech recognition, there remains a compelling need for further improvement. In this work, we present the impressive results of Whisper-MCE, our finetuned Whisper model, which was trained using our self-collected dataset, Mixed Cantonese and English audio dataset (MCE). Meanwhile, considering word error rate (WER) poses challenges when it comes to evaluating its effectiveness in minor language and mixed-language contexts, we present a novel rating mechanism. By comparing our model to the baseline whisper-large-v2 model, we demonstrate its superior ability to accurately capture the content of the original audio, achieve higher recognition accuracy, and exhibit faster recognition speed. Notably, our model outperforms other existing models in the specific task of recognizing mixed language.
    

