# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent-Domain Predictive Neural Speech Coding.](http://arxiv.org/abs/2207.08363) | 本文提出的TF-Codec为一种适用于低延迟的端到端神经语音编码器，通过潜域预测编码完全消除了时间冗余，采用可学习的时间频率输入压缩和基于距离到软映射和Gumbel-Softmax的可微量化方案，相比现有最先进的神经语音编解码器在客观和主观指标上均有显著提升。 |

# 详细

[^1]: 潜域预测神经语音编码

    Latent-Domain Predictive Neural Speech Coding. (arXiv:2207.08363v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2207.08363](http://arxiv.org/abs/2207.08363)

    本文提出的TF-Codec为一种适用于低延迟的端到端神经语音编码器，通过潜域预测编码完全消除了时间冗余，采用可学习的时间频率输入压缩和基于距离到软映射和Gumbel-Softmax的可微量化方案，相比现有最先进的神经语音编解码器在客观和主观指标上均有显著提升。

    

    近期，神经音频/语音编码展现出在远低于传统方法比特率下实现高质量的能力。然而，现有的神经音频/语音编解码器采用声学特征或卷积神经网络学习到的盲目特征进行编码，仍存在编码特征中的时间冗余，本文将潜域预测编码引入VQ-VAE框架中，以完全消除这些冗余，并提出了适用于低延迟的端到端神经语音编码器TF-Codec。具体而言，根据过去量化潜态帧的预测，对提取的特征进行编码，从而进一步消除时间相关性。此外，我们引入一种可学习的时间频率输入压缩，以适应不同比特率下对主要频率和细节的关注。基于距离到软映射和Gumbel-Softmax的可微量化方案用于量化/解码潜域特征。实验结果表明，我们提出的TF-Codec在客观和主观指标上均优于现有的最先进神经语音编解码器。

    Neural audio/speech coding has recently demonstrated its capability to deliver high quality at much lower bitrates than traditional methods. However, existing neural audio/speech codecs employ either acoustic features or learned blind features with a convolutional neural network for encoding, by which there are still temporal redundancies within encoded features. This paper introduces latent-domain predictive coding into the VQ-VAE framework to fully remove such redundancies and proposes the TF-Codec for low-latency neural speech coding in an end-to-end manner. Specifically, the extracted features are encoded conditioned on a prediction from past quantized latent frames so that temporal correlations are further removed. Moreover, we introduce a learnable compression on the time-frequency input to adaptively adjust the attention paid to main frequencies and details at different bitrates. A differentiable vector quantization scheme based on distance-to-soft mapping and Gumbel-Softmax is 
    

