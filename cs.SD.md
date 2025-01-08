# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition.](http://arxiv.org/abs/2309.10524) | 本论文结合指导调整的大语言模型（LLM）和端到端自动语音识别（ASR），利用LLM的零-shot能力来改善语音识别性能。 |

# 详细

[^1]: 发挥指导调整的大语言模型在端到端语音识别中的零-shot能力

    Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition. (arXiv:2309.10524v1 [eess.AS])

    [http://arxiv.org/abs/2309.10524](http://arxiv.org/abs/2309.10524)

    本论文结合指导调整的大语言模型（LLM）和端到端自动语音识别（ASR），利用LLM的零-shot能力来改善语音识别性能。

    

    我们提出了一种将指导调整的大语言模型和端到端自动语音识别相结合的新方法。现代大语言模型在零-shot学习中可以执行各种语言任务，只要提供明确的指导或提示来指导文本生成过程。我们探索使用这种零-shot能力的大语言模型来提取语言信息，以改善语音识别性能。具体来说，我们将大语言模型引导去纠正语音识别假设中的语法错误，并利用嵌入的语言知识进行端到端语音识别。所提出的模型基于混合连接主义时间分类和注意力架构，其中指导调整的大语言模型（即Llama2）被用作解码器的前端。通过CTC解码从编码器获得一个需要纠正的语音识别假设，然后将其与指导一起输入大语言模型。解码器随后采取...

    We present a novel integration of an instruction-tuned large language model (LLM) and end-to-end automatic speech recognition (ASR). Modern LLMs can perform a wide range of linguistic tasks within zero-shot learning when provided with a precise instruction or a prompt to guide the text generation process towards the desired task. We explore using this zero-shot capability of LLMs to extract linguistic information that can contribute to improving ASR performance. Specifically, we direct an LLM to correct grammatical errors in an ASR hypothesis and harness the embedded linguistic knowledge to conduct end-to-end ASR. The proposed model is built on the hybrid connectionist temporal classification (CTC) and attention architecture, where an instruction-tuned LLM (i.e., Llama2) is employed as a front-end of the decoder. An ASR hypothesis, subject to correction, is obtained from the encoder via CTC decoding, which is then fed into the LLM along with an instruction. The decoder subsequently tak
    

