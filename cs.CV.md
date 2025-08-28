# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NPHardEval4V: A Dynamic Reasoning Benchmark of Multimodal Large Language Models](https://arxiv.org/abs/2403.01777) | 这项研究介绍了一个旨在评估多模态大型语言模型推理能力的动态基准NPHardEval4V，发现在推理能力方面不同模型存在显著差异，并揭示了相对于LLMs，MLLMs的推理性能较弱。 |
| [^2] | [Variational Bayes image restoration with compressive autoencoders](https://arxiv.org/abs/2311.17744) | 使用压缩自动编码器代替最先进的生成模型，提出了一种在图像恢复中的新方法。 |

# 详细

[^1]: NPHardEval4V: 多模态大型语言模型的动态推理基准

    NPHardEval4V: A Dynamic Reasoning Benchmark of Multimodal Large Language Models

    [https://arxiv.org/abs/2403.01777](https://arxiv.org/abs/2403.01777)

    这项研究介绍了一个旨在评估多模态大型语言模型推理能力的动态基准NPHardEval4V，发现在推理能力方面不同模型存在显著差异，并揭示了相对于LLMs，MLLMs的推理性能较弱。

    

    理解多模态大型语言模型（MLLMs）的推理能力是一个重要的研究领域。在这项研究中，我们引入了一个动态基准，NPHardEval4V，旨在解决评估MLLM纯粹推理能力方面的现有差距。我们的基准旨在提供一个平台，以解开诸多因素（如图像识别和指令遵循）对模型整体性能的影响，从而专注于评估它们的推理能力。我们的研究发现不同模型在推理能力方面存在显著差异，并突出了相较于LLMs，MLLMs在推理方面表现相对较弱。我们还研究了不同提示样式（包括视觉、文本和结合视觉与文本提示）对MLLM推理能力的影响，展示了多模态输入在模型性能中的不同影响。

    arXiv:2403.01777v1 Announce Type: new  Abstract: Understanding the reasoning capabilities of Multimodal Large Language Models (MLLMs) is an important area of research. In this study, we introduce a dynamic benchmark, NPHardEval4V, aimed at addressing the existing gaps in evaluating the pure reasoning abilities of MLLMs. Our benchmark aims to provide a venue to disentangle the effect of various factors such as image recognition and instruction following, from the overall performance of the models, allowing us to focus solely on evaluating their reasoning abilities. Our findings reveal significant discrepancies in reasoning abilities across different models and highlight the relatively weak performance of MLLMs compared to LLMs in terms of reasoning. We also investigate the impact of different prompting styles, including visual, text, and combined vision and text prompts, on the reasoning abilities of MLLMs, demonstrating the different impacts of multimodal inputs in model performance. U
    
[^2]: 使用压缩自动编码器的变分贝叶斯图像恢复

    Variational Bayes image restoration with compressive autoencoders

    [https://arxiv.org/abs/2311.17744](https://arxiv.org/abs/2311.17744)

    使用压缩自动编码器代替最先进的生成模型，提出了一种在图像恢复中的新方法。

    

    逆问题的正则化在计算成像中至关重要。近年来，神经网络学习有效图像表示的能力已被利用来设计强大的数据驱动正则化器。本文首先提出使用压缩自动编码器。这些网络可以被看作具有灵活潜在先验的变分自动编码器，比起最先进的生成模型更小更容易训练。

    arXiv:2311.17744v2 Announce Type: replace-cross  Abstract: Regularization of inverse problems is of paramount importance in computational imaging. The ability of neural networks to learn efficient image representations has been recently exploited to design powerful data-driven regularizers. While state-of-the-art plug-and-play methods rely on an implicit regularization provided by neural denoisers, alternative Bayesian approaches consider Maximum A Posteriori (MAP) estimation in the latent space of a generative model, thus with an explicit regularization. However, state-of-the-art deep generative models require a huge amount of training data compared to denoisers. Besides, their complexity hampers the optimization involved in latent MAP derivation. In this work, we first propose to use compressive autoencoders instead. These networks, which can be seen as variational autoencoders with a flexible latent prior, are smaller and easier to train than state-of-the-art generative models. As a
    

