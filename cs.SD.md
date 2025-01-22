# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DurFlex-EVC: Duration-Flexible Emotional Voice Conversion with Parallel Generation](https://arxiv.org/abs/2401.08095) | DurFlex-EVC通过引入风格自编码器和交叉注意力，解决了传统情绪语音转换模型中对语言和语音信息的同步问题。 |
| [^2] | [Communication-Efficient Personalized Federated Learning for Speech-to-Text Tasks.](http://arxiv.org/abs/2401.10070) | 该论文提出了一种通信高效的个性化联邦学习框架，通过引入轻量级的LoRA模块进行客户端调整和与服务器的交互，以最小化通信开销，以及使用K最近邻分类器的全局模型来实现个性化并克服数据异构问题。 |

# 详细

[^1]: DurFlex-EVC: 具有并行生成的持续灵活情绪语音转换

    DurFlex-EVC: Duration-Flexible Emotional Voice Conversion with Parallel Generation

    [https://arxiv.org/abs/2401.08095](https://arxiv.org/abs/2401.08095)

    DurFlex-EVC通过引入风格自编码器和交叉注意力，解决了传统情绪语音转换模型中对语言和语音信息的同步问题。

    

    情绪语音转换（EVC）旨在修改说话者声音的情绪色彩，同时保留原始的语言内容和说话者独特的声音特征。最近EVC的进展涉及同时建模音高和持续时间，利用序列到序列（seq2seq）模型的潜力。为了增强转换的可靠性和效率，本研究将重点转向并行语音生成。我们介绍了DurFlex-EVC，它集成了风格自编码器和单元对齐器。传统模型虽然融入了包含语言和语音信息的自监督学习（SSL）表示，但却忽视了这种双重性质，导致了可控性的降低。为了解决这个问题，我们实现了交叉注意力以将这些表示与不同情绪进行同步。此外，我们还开发了一个风格自编码器。

    arXiv:2401.08095v2 Announce Type: replace-cross  Abstract: Emotional voice conversion (EVC) seeks to modify the emotional tone of a speaker's voice while preserving the original linguistic content and the speaker's unique vocal characteristics. Recent advancements in EVC have involved the simultaneous modeling of pitch and duration, utilizing the potential of sequence-to-sequence (seq2seq) models. To enhance reliability and efficiency in conversion, this study shifts focus towards parallel speech generation. We introduce Duration-Flexible EVC (DurFlex-EVC), which integrates a style autoencoder and unit aligner. Traditional models, while incorporating self-supervised learning (SSL) representations that contain both linguistic and paralinguistic information, have neglected this dual nature, leading to reduced controllability. Addressing this issue, we implement cross-attention to synchronize these representations with various emotions. Additionally, a style autoencoder is developed for t
    
[^2]: 通信高效的个性化联邦学习在语音转文本任务中的应用

    Communication-Efficient Personalized Federated Learning for Speech-to-Text Tasks. (arXiv:2401.10070v1 [cs.CL])

    [http://arxiv.org/abs/2401.10070](http://arxiv.org/abs/2401.10070)

    该论文提出了一种通信高效的个性化联邦学习框架，通过引入轻量级的LoRA模块进行客户端调整和与服务器的交互，以最小化通信开销，以及使用K最近邻分类器的全局模型来实现个性化并克服数据异构问题。

    

    为了保护隐私并满足法规要求，联邦学习在训练语音转文本系统（包括自动语音识别和语音翻译）方面引起了广泛关注。然而，在语音转文本任务中常用的联邦学习方法（即FedAvg）通常面临着大量的通信开销和数据异构导致的性能下降问题。为了解决这些问题，我们提出了一种个性化的联邦语音转文本框架，引入了轻量级的LoRA模块（FedLoRA）用于客户端调整和与服务器进行交互以最小化通信开销，以及全局模型（FedMem）配备了K最近邻分类器，以捕捉客户特定的分布变化以实现个性化并克服数据异构。在CoVoST和GigaSp数据集上基于Conformer和Whisper主干模型进行了大量实验。

    To protect privacy and meet legal regulations, federated learning (FL) has gained significant attention for training speech-to-text (S2T) systems, including automatic speech recognition (ASR) and speech translation (ST). However, the commonly used FL approach (i.e., \textsc{FedAvg}) in S2T tasks typically suffers from extensive communication overhead due to multi-round interactions based on the whole model and performance degradation caused by data heterogeneity among clients.To address these issues, we propose a personalized federated S2T framework that introduces \textsc{FedLoRA}, a lightweight LoRA module for client-side tuning and interaction with the server to minimize communication overhead, and \textsc{FedMem}, a global model equipped with a $k$-nearest-neighbor ($k$NN) classifier that captures client-specific distributional shifts to achieve personalization and overcome data heterogeneity. Extensive experiments based on Conformer and Whisper backbone models on CoVoST and GigaSp
    

