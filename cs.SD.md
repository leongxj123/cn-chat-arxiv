# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cross-Utterance Conditioned VAE for Speech Generation.](http://arxiv.org/abs/2309.04156) | 该论文提出了一种跨发言条件化变分自编码器框架，利用预训练语言模型和变分自编码器来增强语音合成的韵律，并确保自然语音生成。该框架的核心组件是跨发言CVAE，通过提取周围句子的声学、说话人和文本特征来生成上下文敏感的韵律特征，有效模拟人类韵律生成。同时，该论文还提出了两个实用算法：CUC-VAE TTS用于文本到语音合成和CUC-VAE SE用于语音编辑。 |

# 详细

[^1]: 跨发言条件化VAE语音生成

    Cross-Utterance Conditioned VAE for Speech Generation. (arXiv:2309.04156v1 [cs.SD])

    [http://arxiv.org/abs/2309.04156](http://arxiv.org/abs/2309.04156)

    该论文提出了一种跨发言条件化变分自编码器框架，利用预训练语言模型和变分自编码器来增强语音合成的韵律，并确保自然语音生成。该框架的核心组件是跨发言CVAE，通过提取周围句子的声学、说话人和文本特征来生成上下文敏感的韵律特征，有效模拟人类韵律生成。同时，该论文还提出了两个实用算法：CUC-VAE TTS用于文本到语音合成和CUC-VAE SE用于语音编辑。

    

    由神经网络驱动的语音合成系统在多媒体制作中有着潜力，但常常面临产生有表现力的语音和无缝编辑的问题。为此，我们提出了跨发言条件化变分自编码器语音合成(CUC-VAE S2)框架，以增强韵律并确保自然语音生成。该框架利用预训练语言模型的强大表现能力和变分自编码器(VAEs)的再表达能力。CUC-VAE S2框架的核心组件是跨发言CVAE，它从周围的句子中提取声学、说话人和文本特征，以生成上下文敏感的韵律特征，更准确地模拟人类韵律生成。我们进一步提出了两个针对不同语音合成应用的实用算法：CUC-VAE TTS以进行文本到语音合成和CUC-VAE SE以进行语音编辑。CUC-VAE TTS是该框架的直接应用，使得能够将任意文本转成语音。

    Speech synthesis systems powered by neural networks hold promise for multimedia production, but frequently face issues with producing expressive speech and seamless editing. In response, we present the Cross-Utterance Conditioned Variational Autoencoder speech synthesis (CUC-VAE S2) framework to enhance prosody and ensure natural speech generation. This framework leverages the powerful representational capabilities of pre-trained language models and the re-expression abilities of variational autoencoders (VAEs). The core component of the CUC-VAE S2 framework is the cross-utterance CVAE, which extracts acoustic, speaker, and textual features from surrounding sentences to generate context-sensitive prosodic features, more accurately emulating human prosody generation. We further propose two practical algorithms tailored for distinct speech synthesis applications: CUC-VAE TTS for text-to-speech and CUC-VAE SE for speech editing. The CUC-VAE TTS is a direct application of the framework, de
    

