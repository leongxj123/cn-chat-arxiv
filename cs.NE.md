# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks](https://arxiv.org/abs/2403.12574) | 提出了一种利用循环脉冲神经网络的自适应采样模块，通过将脉冲神经元的神经动力学与理想的时间事件采样器的行为相结合，实现了端到端可学习的事件检测框架 |
| [^2] | [Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data.](http://arxiv.org/abs/2306.13840) | 本论文提出使用多样性系数作为LLM预训练数据质量的指标，研究表明公开可用的LLM数据集的多样性系数很高。 |

# 详细

[^1]: EAS-SNN：端到端自适应采样和表示，用于循环脉冲神经网络的事件检测

    EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks

    [https://arxiv.org/abs/2403.12574](https://arxiv.org/abs/2403.12574)

    提出了一种利用循环脉冲神经网络的自适应采样模块，通过将脉冲神经元的神经动力学与理想的时间事件采样器的行为相结合，实现了端到端可学习的事件检测框架

    

    事件摄像头以其高动态范围和时间分辨率，特别适用于物体检测，尤其是在存在动态模糊和具有挑战性的光照条件的情况下。然而，大多数现有方法更注重优化具有先进检测骨干和早期聚合功能的时空表示，而自适应事件采样的关键问题仍未得到解决。脉冲神经网络（SNN），通过稀疏脉冲通信运行的事件驱动范式，成为解决这一挑战的天然选择。在这项研究中，我们发现脉冲神经元的神经动力学与理想的时间事件采样器的行为密切相符。在这一启发下，我们提出了一个新颖的自适应采样模块，利用具有时间记忆的循环卷积SNN增强，为基于事件检测的完全端到端可学习框架提供支持。

    arXiv:2403.12574v1 Announce Type: cross  Abstract: Event cameras, with their high dynamic range and temporal resolution, are ideally suited for object detection, especially under scenarios with motion blur and challenging lighting conditions. However, while most existing approaches prioritize optimizing spatiotemporal representations with advanced detection backbones and early aggregation functions, the crucial issue of adaptive event sampling remains largely unaddressed. Spiking Neural Networks (SNNs), which operate on an event-driven paradigm through sparse spike communication, emerge as a natural fit for addressing this challenge. In this study, we discover that the neural dynamics of spiking neurons align closely with the behavior of an ideal temporal event sampler. Motivated by this insight, we propose a novel adaptive sampling module that leverages recurrent convolutional SNNs enhanced with temporal memory, facilitating a fully end-to-end learnable framework for event-based detec
    
[^2]: 超越规模：多样性系数作为数据质量指标证明了LLMs是在形式多样的数据上预先训练的

    Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data. (arXiv:2306.13840v1 [cs.CL])

    [http://arxiv.org/abs/2306.13840](http://arxiv.org/abs/2306.13840)

    本论文提出使用多样性系数作为LLM预训练数据质量的指标，研究表明公开可用的LLM数据集的多样性系数很高。

    

    当前，预先训练强大的大语言模型(LLMs)的趋势主要集中在模型和数据集规模的扩大。然而，预先训练数据的质量对于训练强大的LLMs来说是一个重要因素，但它是一个模糊的概念，尚未完全表征。因此，我们使用最近提出的Task2Vec多样性系数来基于数据质量的形式方面，超越规模本身。具体而言，我们测量公开可用的预先训练数据集的多样性系数，以证明它们的形式多样性高于理论的下限和上限。此外，为了建立对多样性系数的信心，我们进行可解释性实验，并发现该系数与多样性的直观属性相吻合，例如，随着潜在概念数量的增加，它增加。我们得出结论，多样性系数是可靠的，表明公开可用的LLM数据集的多样性系数很高，并推测它可以作为预训练LLMs模型的数据质量指标。

    Current trends to pre-train capable Large Language Models (LLMs) mostly focus on scaling of model and dataset size. However, the quality of pre-training data is an important factor for training powerful LLMs, yet it is a nebulous concept that has not been fully characterized. Therefore, we use the recently proposed Task2Vec diversity coefficient to ground and understand formal aspects of data quality, to go beyond scale alone. Specifically, we measure the diversity coefficient of publicly available pre-training datasets to demonstrate that their formal diversity is high when compared to theoretical lower and upper bounds. In addition, to build confidence in the diversity coefficient, we conduct interpretability experiments and find that the coefficient aligns with intuitive properties of diversity, e.g., it increases as the number of latent concepts increases. We conclude the diversity coefficient is reliable, show it's high for publicly available LLM datasets, and conjecture it can be
    

