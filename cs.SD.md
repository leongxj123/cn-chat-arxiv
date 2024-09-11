# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SPMamba: State-space model is all you need in speech separation](https://arxiv.org/abs/2404.02063) | SPMamba提出了一种利用状态空间模型进行语音分离的新网络架构，通过替换Transformer组件为Mamba模块，旨在提高性能并减少计算需求。 |

# 详细

[^1]: SPMamba：状态空间模型是语音分离中所需的一切

    SPMamba: State-space model is all you need in speech separation

    [https://arxiv.org/abs/2404.02063](https://arxiv.org/abs/2404.02063)

    SPMamba提出了一种利用状态空间模型进行语音分离的新网络架构，通过替换Transformer组件为Mamba模块，旨在提高性能并减少计算需求。

    

    在语音分离领域，CNN和Transformer模型都展示了稳健的分离能力，引起了研究社区的广泛关注。然而，基于CNN的方法对于长序列音频的建模能力有限，导致分离性能不佳。相反，基于Transformer的方法在实际应用中受到计算复杂性的限制。本文提出了一种利用状态空间模型进行语音分离的网络架构，即SPMamba。我们采用TF-GridNet模型作为基础框架，并将其Transformer组件替换为一个双向Mamba模块，旨在捕获更广泛的上下文信息。我们的实验结果揭示了Mamba基于方法在提高性能和减少计算需求方面的重要作用。

    arXiv:2404.02063v1 Announce Type: cross  Abstract: In speech separation, both CNN- and Transformer-based models have demonstrated robust separation capabilities, garnering significant attention within the research community. However, CNN-based methods have limited modelling capability for long-sequence audio, leading to suboptimal separation performance. Conversely, Transformer-based methods are limited in practical applications due to their high computational complexity. Notably, within computer vision, Mamba-based methods have been celebrated for their formidable performance and reduced computational requirements. In this paper, we propose a network architecture for speech separation using a state-space model, namely SPMamba. We adopt the TF-GridNet model as the foundational framework and substitute its Transformer component with a bidirectional Mamba module, aiming to capture a broader range of contextual information. Our experimental results reveal an important role in the performa
    

