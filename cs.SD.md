# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hybrid Attention-based Encoder-decoder Model for Efficient Language Model Adaptation.](http://arxiv.org/abs/2309.07369) | 半混合注意力编码器-解码器模型通过分离声学模型和语言模型，以实现对传统文本语言模型适应技术的利用。在使用域外文本数据进行语言模型适应时，相对于传统模型，该模型可获得21\%的词错误率改进。 |

# 详细

[^1]: 半混合注意力编码器-解码器模型用于高效语言模型适应

    Hybrid Attention-based Encoder-decoder Model for Efficient Language Model Adaptation. (arXiv:2309.07369v1 [eess.AS])

    [http://arxiv.org/abs/2309.07369](http://arxiv.org/abs/2309.07369)

    半混合注意力编码器-解码器模型通过分离声学模型和语言模型，以实现对传统文本语言模型适应技术的利用。在使用域外文本数据进行语言模型适应时，相对于传统模型，该模型可获得21\%的词错误率改进。

    

    基于注意力的编码器-解码器语音识别模型近年来取得了广泛的成功。然而，在端到端方式中联合优化声学模型和语言模型对于文本适应性提出了挑战。特别是，有效、快速和廉价地适应文本已成为在工业中部署注意力编码器-解码器系统的主要关注点。为了解决这个问题，我们提出了一种新颖的模型，即半混合注意力编码器-解码器语音识别模型，保留了传统混合自动语音识别系统的模块化特性。我们的半混合注意力编码器-解码器模型将声学模型和语言模型分离，使得可以使用传统的基于文本的语言模型适应技术。我们证明了在使用域外文本数据进行语言模型适应时，所提出的半混合注意力编码器-解码器模型相对于传统的基于注意力的模型在词错误率上实现了21\%的改进，并且在常规测试集上的词错误率只有轻微的降低。

    Attention-based encoder-decoder (AED) speech recognition model has been widely successful in recent years. However, the joint optimization of acoustic model and language model in end-to-end manner has created challenges for text adaptation. In particular, effectively, quickly and inexpensively adapting text has become a primary concern for deploying AED systems in industry. To address this issue, we propose a novel model, the hybrid attention-based encoder-decoder (HAED) speech recognition model that preserves the modularity of conventional hybrid automatic speech recognition systems. Our HAED model separates the acoustic and language models, allowing for the use of conventional text-based language model adaptation techniques. We demonstrate that the proposed HAED model yields 21\% Word Error Rate (WER) improvements in relative when out-of-domain text data is used for language model adaptation, and with only a minor degradation in WER on a general test set compared with conventional AE
    

