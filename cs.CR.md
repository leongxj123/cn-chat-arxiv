# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Resilience of Entropy Model in Distributed Neural Networks](https://arxiv.org/abs/2403.00942) | 本文研究了分布式神经网络中熵模型对有意干扰和无意干扰的韧性，通过实验证明了熵模型的韧性。 |

# 详细

[^1]: 分布式神经网络中熵模型的韧性

    Resilience of Entropy Model in Distributed Neural Networks

    [https://arxiv.org/abs/2403.00942](https://arxiv.org/abs/2403.00942)

    本文研究了分布式神经网络中熵模型对有意干扰和无意干扰的韧性，通过实验证明了熵模型的韧性。

    

    分布式深度神经网络（DNNs）已经成为边缘计算系统中减少通信开销而不降低性能的关键技术。最近，熵编码被引入以进一步减少通信开销。其关键思想是将分布式DNN与熵模型联合训练，该模型在推断时间用作边信息，以自适应地将潜在表示编码为具有可变长度的比特流。据我们所知，熵模型的韧性尚未得到研究。因此，在本文中，我们制定并调查了熵模型对有意干扰（例如，对抗性攻击）和无意干扰（例如，天气变化和运动模糊）的韧性。通过对3种不同DNN架构、2个熵模型和4个速率失真权衡因子进行广泛的实验，我们证明了熵模型的韧性

    arXiv:2403.00942v1 Announce Type: cross  Abstract: Distributed deep neural networks (DNNs) have emerged as a key technique to reduce communication overhead without sacrificing performance in edge computing systems. Recently, entropy coding has been introduced to further reduce the communication overhead. The key idea is to train the distributed DNN jointly with an entropy model, which is used as side information during inference time to adaptively encode latent representations into bit streams with variable length. To the best of our knowledge, the resilience of entropy models is yet to be investigated. As such, in this paper we formulate and investigate the resilience of entropy models to intentional interference (e.g., adversarial attacks) and unintentional interference (e.g., weather changes and motion blur). Through an extensive experimental campaign with 3 different DNN architectures, 2 entropy models and 4 rate-distortion trade-off factors, we demonstrate that the entropy attacks
    

