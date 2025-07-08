# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem](https://arxiv.org/abs/2403.03593) | 介绍了MaleficNet 2.0，一种在神经网络中嵌入恶意软件的新技术，其注入技术具有隐蔽性，不会降低模型性能，并且对神经网络参数中的恶意有效负载进行注入 |
| [^2] | [VGMShield: Mitigating Misuse of Video Generative Models](https://arxiv.org/abs/2402.13126) | VGMShield提出了三项简单但开创性的措施，通过检测虚假视频、溯源问题和利用预训练的空间-时间动态模型，防范视频生成模型的误用。 |

# 详细

[^1]: 您信任您的模型吗？深度学习生态系统中新兴的恶意软件威胁

    Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem

    [https://arxiv.org/abs/2403.03593](https://arxiv.org/abs/2403.03593)

    介绍了MaleficNet 2.0，一种在神经网络中嵌入恶意软件的新技术，其注入技术具有隐蔽性，不会降低模型性能，并且对神经网络参数中的恶意有效负载进行注入

    

    训练高质量的深度学习模型是一项具有挑战性的任务，这是因为需要计算和技术要求。越来越多的个人、机构和公司越来越多地依赖于在公共代码库中提供的预训练的第三方模型。这些模型通常直接使用或集成到产品管道中而没有特殊的预防措施，因为它们实际上只是以张量形式的数据，被认为是安全的。在本文中，我们提出了一种针对神经网络的新的机器学习供应链威胁。我们介绍了MaleficNet 2.0，一种在神经网络中嵌入自解压自执行恶意软件的新技术。MaleficNet 2.0使用扩频信道编码结合纠错技术在深度神经网络的参数中注入恶意有效载荷。MaleficNet 2.0注入技术具有隐蔽性，不会降低模型的性能，并且对...

    arXiv:2403.03593v1 Announce Type: cross  Abstract: Training high-quality deep learning models is a challenging task due to computational and technical requirements. A growing number of individuals, institutions, and companies increasingly rely on pre-trained, third-party models made available in public repositories. These models are often used directly or integrated in product pipelines with no particular precautions, since they are effectively just data in tensor form and considered safe. In this paper, we raise awareness of a new machine learning supply chain threat targeting neural networks. We introduce MaleficNet 2.0, a novel technique to embed self-extracting, self-executing malware in neural networks. MaleficNet 2.0 uses spread-spectrum channel coding combined with error correction techniques to inject malicious payloads in the parameters of deep neural networks. MaleficNet 2.0 injection technique is stealthy, does not degrade the performance of the model, and is robust against 
    
[^2]: VGMShield：缓解视频生成模型的误用

    VGMShield: Mitigating Misuse of Video Generative Models

    [https://arxiv.org/abs/2402.13126](https://arxiv.org/abs/2402.13126)

    VGMShield提出了三项简单但开创性的措施，通过检测虚假视频、溯源问题和利用预训练的空间-时间动态模型，防范视频生成模型的误用。

    

    随着视频生成技术的快速发展，人们可以方便地利用视频生成模型创建符合其特定需求的视频。然而，人们也越来越担心这些技术被用于创作和传播虚假信息。在这项工作中，我们介绍了VGMShield：一套包含三项直接但开创性的措施，用于防范虚假视频生成过程中可能出现的问题。我们首先从“虚假视频检测”开始，尝试理解生成的视频中是否存在独特性，以及我们是否能够区分它们与真实视频的不同；然后，我们探讨“溯源”问题，即将一段虚假视频追溯回生成它的模型。为此，我们提出利用预训练的关注“时空动态”的模型作为骨干，以识别视频中的不一致性。通过对七个最先进的开源模型进行实验，我们证明了...

    arXiv:2402.13126v1 Announce Type: cross  Abstract: With the rapid advancement in video generation, people can conveniently utilize video generation models to create videos tailored to their specific desires. Nevertheless, there are also growing concerns about their potential misuse in creating and disseminating false information.   In this work, we introduce VGMShield: a set of three straightforward but pioneering mitigations through the lifecycle of fake video generation. We start from \textit{fake video detection} trying to understand whether there is uniqueness in generated videos and whether we can differentiate them from real videos; then, we investigate the \textit{tracing} problem, which maps a fake video back to a model that generates it. Towards these, we propose to leverage pre-trained models that focus on {\it spatial-temporal dynamics} as the backbone to identify inconsistencies in videos. Through experiments on seven state-of-the-art open-source models, we demonstrate that
    

