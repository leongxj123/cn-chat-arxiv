# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamic Relative Representations for Goal-Oriented Semantic Communications](https://arxiv.org/abs/2403.16986) | 本文提出了一个新颖的面向目标的语义通信框架，利用相对表示通过潜在空间对齐来缓解语义不匹配，并实现了能效高、延迟低的目标导向语义通信。 |
| [^2] | [Integrating Pre-Trained Language Model with Physical Layer Communications](https://arxiv.org/abs/2402.11656) | 提出了一个集成了物理层通信功能的实用设备间人工智能通信框架，通过端到端训练结合信道噪声以增强韧性，采用VQ-VAE实现高效稳健的通信，利用预训练Transformer提升通用性能 |
| [^3] | [Generalization Error of Graph Neural Networks in the Mean-field Regime](https://arxiv.org/abs/2402.07025) | 该论文在均场极限下提供了一个理论框架，评估了图神经网络在过参数化情况下的泛化误差，通过推导出收敛速度为$O(1/n)$的上界，为我们对网络在未见数据上的性能提供了理论保证。 |
| [^4] | [Neural Distributed Source Coding.](http://arxiv.org/abs/2106.02797) | 这项研究提出了一种神经分布式源编码的框架，可以处理复杂的相关性并实现最先进的峰值信噪比。 |

# 详细

[^1]: 面向目标导向语义通信的动态相对表示

    Dynamic Relative Representations for Goal-Oriented Semantic Communications

    [https://arxiv.org/abs/2403.16986](https://arxiv.org/abs/2403.16986)

    本文提出了一个新颖的面向目标的语义通信框架，利用相对表示通过潜在空间对齐来缓解语义不匹配，并实现了能效高、延迟低的目标导向语义通信。

    

    在未来的6G无线网络中，通信的语义和有效性方面将发挥基础作用，将含义和相关性纳入传输。然而，当设备使用不同的语言、逻辑或内部表示时，会出现障碍，导致语义不匹配，可能危及理解。在潜在空间通信中，这一挑战表现为深度神经网络对数据进行编码时高维表示不匹配。本文提出了一个新颖的面向目标的语义通信框架，利用相对表示来通过潜在空间对齐缓解语义不匹配。我们提出了一种动态优化策略，以调整相对表示、通信参数和计算资源，实现能效高、延迟低的目标导向语义通信。数值结果证明了我们的方法在缓解中起作用的有效性。

    arXiv:2403.16986v1 Announce Type: cross  Abstract: In future 6G wireless networks, semantic and effectiveness aspects of communications will play a fundamental role, incorporating meaning and relevance into transmissions. However, obstacles arise when devices employ diverse languages, logic, or internal representations, leading to semantic mismatches that might jeopardize understanding. In latent space communication, this challenge manifests as misalignment within high-dimensional representations where deep neural networks encode data. This paper presents a novel framework for goal-oriented semantic communication, leveraging relative representations to mitigate semantic mismatches via latent space alignment. We propose a dynamic optimization strategy that adapts relative representations, communication parameters, and computation resources for energy-efficient, low-latency, goal-oriented semantic communications. Numerical results demonstrate our methodology's effectiveness in mitigating
    
[^2]: 将预训练语言模型与物理层通信集成

    Integrating Pre-Trained Language Model with Physical Layer Communications

    [https://arxiv.org/abs/2402.11656](https://arxiv.org/abs/2402.11656)

    提出了一个集成了物理层通信功能的实用设备间人工智能通信框架，通过端到端训练结合信道噪声以增强韧性，采用VQ-VAE实现高效稳健的通信，利用预训练Transformer提升通用性能

    

    在设备间人工智能通信的新兴领域中，设备直接通过嵌入式基础模型（如语言模型）交换信息，需要强大、高效且通用的通信框架。然而，将这些框架与现有无线系统集成并有效管理噪声和比特误差都面临着重大挑战。在本研究中，我们介绍了一个实用的设备间人工智能通信框架，集成了物理层通信功能，并通过链路级模拟器展示了其性能。我们的框架通过端到端训练结合信道噪声以增强韧性，采用向量量化变分自动编码器（VQ-VAE）实现高效稳健的通信，利用预训练编码-解码Transformer提升通用性能。在各种通信场景的模拟中，我们的框架展现出

    arXiv:2402.11656v1 Announce Type: cross  Abstract: The burgeoning field of on-device AI communication, where devices exchange information directly through embedded foundation models, such as language models (LMs), requires robust, efficient, and generalizable communication frameworks. However, integrating these frameworks with existing wireless systems and effectively managing noise and bit errors pose significant challenges. In this work, we introduce a practical on-device AI communication framework, integrated with physical layer (PHY) communication functions, demonstrated through its performance on a link-level simulator. Our framework incorporates end-to-end training with channel noise to enhance resilience, incorporates vector quantized variational autoencoders (VQ-VAE) for efficient and robust communication, and utilizes pre-trained encoder-decoder transformers for improved generalization capabilities. Simulations, across various communication scenarios, reveal that our framework
    
[^3]: 均场极限下图神经网络的泛化误差

    Generalization Error of Graph Neural Networks in the Mean-field Regime

    [https://arxiv.org/abs/2402.07025](https://arxiv.org/abs/2402.07025)

    该论文在均场极限下提供了一个理论框架，评估了图神经网络在过参数化情况下的泛化误差，通过推导出收敛速度为$O(1/n)$的上界，为我们对网络在未见数据上的性能提供了理论保证。

    

    该工作提供了一个理论框架，用于评估在过参数化的情况下通过图神经网络进行图分类任务的泛化误差，即参数数量超过数据点数量的情况。我们研究了两种广泛使用的图神经网络类型：图卷积神经网络和消息传递图神经网络。在本研究之前，关于过参数化情况下泛化误差的现有界限缺乏信息，限制了我们对过参数化网络性能的理解。我们的创新方法是在均场极限下推导出上界，以评估这些图神经网络的泛化误差。我们建立了以$O(1/n)$收敛速度的上界，其中$n$是图样本的数量。这些上界为在具有挑战性的过参数化情况下网络在未见数据上的性能提供了理论上的保证，从而对我们的理解做出了贡献。

    This work provides a theoretical framework for assessing the generalization error of graph classification tasks via graph neural networks in the over-parameterized regime, where the number of parameters surpasses the quantity of data points. We explore two widely utilized types of graph neural networks: graph convolutional neural networks and message passing graph neural networks. Prior to this study, existing bounds on the generalization error in the over-parametrized regime were uninformative, limiting our understanding of over-parameterized network performance. Our novel approach involves deriving upper bounds within the mean-field regime for evaluating the generalization error of these graph neural networks. We establish upper bounds with a convergence rate of $O(1/n)$, where $n$ is the number of graph samples. These upper bounds offer a theoretical assurance of the networks' performance on unseen data in the challenging over-parameterized regime and overall contribute to our under
    
[^4]: 神经分布式源编码

    Neural Distributed Source Coding. (arXiv:2106.02797v3 [cs.IT] UPDATED)

    [http://arxiv.org/abs/2106.02797](http://arxiv.org/abs/2106.02797)

    这项研究提出了一种神经分布式源编码的框架，可以处理复杂的相关性并实现最先进的峰值信噪比。

    

    分布式源编码(DSC)是在没有相互关联的边际信息可供解码器使用的情况下对输入进行编码的任务。值得注意的是，Slepian和Wolf在1973年证明，没有访问边际信息的编码器可以渐近地实现与边际信息可用情况下相同的压缩率。虽然在这个领域有广泛的先前工作，但实践中的DSC一直局限于合成数据集和特定的相关结构。在这里，我们提出了一个对相关结构不可知且能够扩展到高维度的有损DSC框架。我们的方法不依赖于手工设计的源模型，而是利用条件向量量化变分自动编码器(VQ-VAE)来学习分布式编码器和解码器。我们在多个数据集上评估了我们的方法，并展示了我们的方法可以处理复杂的相关性，并实现了最先进的峰值信噪比(PSNR)。

    Distributed source coding (DSC) is the task of encoding an input in the absence of correlated side information that is only available to the decoder. Remarkably, Slepian and Wolf showed in 1973 that an encoder without access to the side information can asymptotically achieve the same compression rate as when the side information is available to it. While there is vast prior work on this topic, practical DSC has been limited to synthetic datasets and specific correlation structures. Here we present a framework for lossy DSC that is agnostic to the correlation structure and can scale to high dimensions. Rather than relying on hand-crafted source modeling, our method utilizes a conditional Vector-Quantized Variational Autoencoder (VQ-VAE) to learn the distributed encoder and decoder. We evaluate our method on multiple datasets and show that our method can handle complex correlations and achieves state-of-the-art PSNR.
    

