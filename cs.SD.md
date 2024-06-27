# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Compact Speech Translation Models via Discrete Speech Units Pretraining](https://arxiv.org/abs/2402.19333) | 通过在离散语音单元上预训练较小模型，以蒸馏SSL模型的知识，实现了紧凑的语音翻译模型，具有短推理管道和适用于低资源环境等优点 |
| [^2] | [Wavelet Scattering Transform for Bioacustics: Application to Watkins Marine Mammal Sound Database](https://arxiv.org/abs/2402.17775) | 本研究提出了在Watkins海洋哺乳动物声音数据库上应用Wavelet散射变换（WST）和Mel频谱图预处理的方法，在分类任务中取得了较高的准确率。 |
| [^3] | [MFAS: Emotion Recognition through Multiple Perspectives Fusion Architecture Search Emulating Human Cognition.](http://arxiv.org/abs/2306.09361) | 该论文提出了一种基于多角度融合结构搜索的情感识别框架，模拟人类的认知过程，能够从连续的角度捕捉更全面的情感信息。 |
| [^4] | [On the Impact of Voice Anonymization on Speech-Based COVID-19 Detection.](http://arxiv.org/abs/2304.02181) | 研究探讨了语音匿名化在 COVID-19 检测应用中的影响。研究发现，匿名化方法可能会对语音诊断系统的准确性产生显著影响。 |

# 详细

[^1]: 通过离散语音单元预训练实现紧凑的语音翻译模型

    Compact Speech Translation Models via Discrete Speech Units Pretraining

    [https://arxiv.org/abs/2402.19333](https://arxiv.org/abs/2402.19333)

    通过在离散语音单元上预训练较小模型，以蒸馏SSL模型的知识，实现了紧凑的语音翻译模型，具有短推理管道和适用于低资源环境等优点

    

    使用自监督学习（SSL）作为模型初始化如今在语音翻译（ST）中获得强大结果是常见的。然而，它们也会占用大量内存，阻碍了设备部署。本文利用SSL模型通过在其离散语音单元（DSU）上预训练较小模型。我们在1）Filterbank-to-DSU和2）DSU-to-Translation数据上预训练编码器-解码器模型，然后取自1）的编码器和来自2）的解码器来初始化一个新模型，在有限的语音翻译数据上微调。通过使用DSU预训练来提炼SSL模型的知识，最终模型变得紧凑。我们的方法相比于使用DSU作为模型输入有几个优点，比如推理管道更短和对（DSU）标记化的鲁棒性。与ASR预训练相比，它不需要转录，使其适用于资源匮乏的环境。在CoVoST-2 X-En上的评估显示我们的方法是

    arXiv:2402.19333v1 Announce Type: new  Abstract: Using Self-Supervised Learning (SSL) as model initialization is now common to obtain strong results in Speech Translation (ST). However, they also impose a large memory footprint, hindering on-device deployment. In this paper, we leverage the SSL models by pretraining smaller models on their Discrete Speech Units (DSU). We pretrain encoder-decoder models on 1) Filterbank-to-DSU and 2) DSU-to-Translation data, and take the encoder from 1) and the decoder from 2) to initialise a new model, finetuning this on limited speech-translation data. The final model becomes compact by using the DSU pretraining to distil the knowledge of the SSL model. Our method has several benefits over using DSU as model inputs, such as shorter inference pipeline and robustness over (DSU) tokenization. In contrast to ASR pretraining, it does not require transcripts, making it applicable to low-resource settings. Evaluation on CoVoST-2 X-En shows that our method is
    
[^2]: Wavelet散射变换在生物声学中的应用：以Watkins海洋哺乳动物声音数据库为例

    Wavelet Scattering Transform for Bioacustics: Application to Watkins Marine Mammal Sound Database

    [https://arxiv.org/abs/2402.17775](https://arxiv.org/abs/2402.17775)

    本研究提出了在Watkins海洋哺乳动物声音数据库上应用Wavelet散射变换（WST）和Mel频谱图预处理的方法，在分类任务中取得了较高的准确率。

    

    海洋哺乳动物的交流是一个复杂的领域，受到鸣叫的多样性和环境因素的影响。Watkins海洋哺乳动物声音数据库（WMMD）是一个广泛应用于机器学习中的标记数据集。本研究首先重点介绍了该数据集上最新的基准记录，着重澄清数据准备和预处理方法。随后，我们提出了在STFT基础上应用Wavelet散射变换（WST）的方法。研究还探讨了使用自适应深层架构和残差层进行分类任务。我们在准确率上使用WST比现有分类架构提高了6％，使用Mel频谱图预处理提高了8％，从而有效地减少了

    arXiv:2402.17775v1 Announce Type: cross  Abstract: Marine mammal communication is a complex field, hindered by the diversity of vocalizations and environmental factors. The Watkins Marine Mammal Sound Database (WMMD) is an extensive labeled dataset used in machine learning applications. However, the methods for data preparation, preprocessing, and classification found in the literature are quite disparate. This study first focuses on a brief review of the state-of-the-art benchmarks on the dataset, with an emphasis on clarifying data preparation and preprocessing methods. Subsequently, we propose the application of the Wavelet Scattering Transform (WST) in place of standard methods based on the Short-Time Fourier Transform (STFT). The study also tackles a classification task using an ad-hoc deep architecture with residual layers. We outperform the existing classification architecture by $6\%$ in accuracy using WST and $8\%$ using Mel spectrogram preprocessing, effectively reducing by h
    
[^3]: MFAS: 基于多角度融合结构搜索的情感识别，模拟人类认知

    MFAS: Emotion Recognition through Multiple Perspectives Fusion Architecture Search Emulating Human Cognition. (arXiv:2306.09361v1 [eess.AS])

    [http://arxiv.org/abs/2306.09361](http://arxiv.org/abs/2306.09361)

    该论文提出了一种基于多角度融合结构搜索的情感识别框架，模拟人类的认知过程，能够从连续的角度捕捉更全面的情感信息。

    

    语音情感识别旨在识别和分析与人类类似的情绪状态。完美的情感识别可以极大地改善各种人机交互任务。受人类理解情感的过程的启发，我们证明了与量化建模相比，从连续的角度理解语音内容，类似于人类的理解，能够使模型捕捉更全面的情感信息。此外，考虑到人类根据语音中存在的某些线索调整情感单词的文本语义的感知，我们设计了一个新的搜索空间并搜索两种信息的最佳融合策略。实验结果进一步验证了调整感知的重要性。基于这些观察结果，我们提出了一种新的框架，称为Multiple perspectives Fusion Architecture Search(MFAS)。

    Speech emotion recognition aims to identify and analyze emotional states in target speech similar to humans. Perfect emotion recognition can greatly benefit a wide range of human-machine interaction tasks. Inspired by the human process of understanding emotions, we demonstrate that compared to quantized modeling, understanding speech content from a continuous perspective, akin to human-like comprehension, enables the model to capture more comprehensive emotional information. Additionally, considering that humans adjust their perception of emotional words in textual semantic based on certain cues present in speech, we design a novel search space and search for the optimal fusion strategy for the two types of information. Experimental results further validate the significance of this perception adjustment. Building on these observations, we propose a novel framework called Multiple perspectives Fusion Architecture Search (MFAS). Specifically, we utilize continuous-based knowledge to capt
    
[^4]: 关于声音匿名化对基于语音的COVID-19检测的影响研究

    On the Impact of Voice Anonymization on Speech-Based COVID-19 Detection. (arXiv:2304.02181v1 [cs.CL])

    [http://arxiv.org/abs/2304.02181](http://arxiv.org/abs/2304.02181)

    研究探讨了语音匿名化在 COVID-19 检测应用中的影响。研究发现，匿名化方法可能会对语音诊断系统的准确性产生显著影响。

    

    随着深度学习的发展，基于语音的应用正蓬勃发展，从个人助理、情感计算到远程疾病诊断。由于声音同时包含语言和语用信息（如语音音调、语调、语速、声音大小），因此保护说话者的隐私和身份的声音匿名化引起了广泛的关注。近年来，声音隐私问题已经出现，重点是去除说话者身份，同时保留语言内容。然而，对于情感计算和疾病监测应用而言，语用内容可能更为关键。遗憾的是，匿名化可能对这些系统产生的影响仍然不明确。在本文中，我们填补了这个空白，并专注于一个特定的健康监测应用：基于语音的COVID-19诊断。我们测试了两种流行的匿名化方法及其对五种最先进的COVID-19诊断系统的影响。

    With advances seen in deep learning, voice-based applications are burgeoning, ranging from personal assistants, affective computing, to remote disease diagnostics. As the voice contains both linguistic and paralinguistic information (e.g., vocal pitch, intonation, speech rate, loudness), there is growing interest in voice anonymization to preserve speaker privacy and identity. Voice privacy challenges have emerged over the last few years and focus has been placed on removing speaker identity while keeping linguistic content intact. For affective computing and disease monitoring applications, however, the paralinguistic content may be more critical. Unfortunately, the effects that anonymization may have on these systems are still largely unknown. In this paper, we fill this gap and focus on one particular health monitoring application: speech-based COVID-19 diagnosis. We test two popular anonymization methods and their impact on five different state-of-the-art COVID-19 diagnostic system
    

