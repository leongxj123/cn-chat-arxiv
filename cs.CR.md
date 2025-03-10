# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-Aware Semantic Cache for Large Language Models](https://arxiv.org/abs/2403.02694) | MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。 |
| [^2] | [Universal Post-Training Reverse-Engineering Defense Against Backdoors in Deep Neural Networks](https://arxiv.org/abs/2402.02034) | 本文提出了一种针对深度神经网络中后门攻击的通用后训练反向工程防御方法，通过依赖内部特征图来检测和反向工程后门，并识别其目标类别，具有广泛适用性和低计算开销。 |

# 详细

[^1]: 面向大型语言模型的隐私感知语义缓存

    Privacy-Aware Semantic Cache for Large Language Models

    [https://arxiv.org/abs/2403.02694](https://arxiv.org/abs/2403.02694)

    MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。

    

    大型语言模型（LLMs）如ChatGPT、Google Bard、Claude和Llama 2彻底改变了自然语言处理和搜索引擎动态。然而，这些模型造成了异常高的计算成本。本文介绍了MeanCache，一种用于LLMs的语义缓存，它能够识别语义上相似的查询以确定缓存命中或未命中。

    arXiv:2403.02694v1 Announce Type: cross  Abstract: Large Language Models (LLMs) like ChatGPT, Google Bard, Claude, and Llama 2 have revolutionized natural language processing and search engine dynamics. However, these models incur exceptionally high computational costs. For instance, GPT-3 consists of 175 billion parameters and inference on these models also demands billions of floating-point operations. Caching is a natural solution to reduce LLM inference costs on repeated queries. However, existing caching methods are incapable of finding semantic similarities among LLM queries, leading to unacceptable false hit-and-miss rates.   This paper introduces MeanCache, a semantic cache for LLMs that identifies semantically similar queries to determine cache hit or miss. Using MeanCache, the response to a user's semantically similar query can be retrieved from a local cache rather than re-querying the LLM, thus reducing costs, service provider load, and environmental impact. MeanCache lever
    
[^2]: 深度神经网络中针对后门攻击的通用后训练反向工程防御方法

    Universal Post-Training Reverse-Engineering Defense Against Backdoors in Deep Neural Networks

    [https://arxiv.org/abs/2402.02034](https://arxiv.org/abs/2402.02034)

    本文提出了一种针对深度神经网络中后门攻击的通用后训练反向工程防御方法，通过依赖内部特征图来检测和反向工程后门，并识别其目标类别，具有广泛适用性和低计算开销。

    

    针对深度神经网络分类器的后门攻击，提出了各种防御方法。通用方法旨在可靠地检测和/或减轻后门攻击，而反向工程方法通常明确假设其中一种。本文提出了一种新的检测器，它依赖于被防守的DNN的内部特征图来检测和反向工程后门，并识别其目标类别；它可以在后训练时操作（无需访问训练数据集）；对于不同的嵌入机制（即通用的）非常有效；并且具有低计算开销，因此可扩展。我们对基准CIFAR-10图像分类器的不同攻击进行了检测方法的评估。

    A variety of defenses have been proposed against backdoors attacks on deep neural network (DNN) classifiers. Universal methods seek to reliably detect and/or mitigate backdoors irrespective of the incorporation mechanism used by the attacker, while reverse-engineering methods often explicitly assume one. In this paper, we describe a new detector that: relies on internal feature map of the defended DNN to detect and reverse-engineer the backdoor and identify its target class; can operate post-training (without access to the training dataset); is highly effective for various incorporation mechanisms (i.e., is universal); and which has low computational overhead and so is scalable. Our detection approach is evaluated for different attacks on a benchmark CIFAR-10 image classifier.
    

