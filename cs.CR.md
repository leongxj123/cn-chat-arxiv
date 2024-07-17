# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond the Request: Harnessing HTTP Response Headers for Cross-Browser Web Tracker Classification in an Imbalanced Setting](https://rss.arxiv.org/abs/2402.01240) | 本研究通过利用HTTP响应头设计了机器学习分类器，在跨浏览器环境下有效检测Web追踪器，结果在Chrome和Firefox上表现出较高的准确性和性能。 |
| [^2] | [JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models](https://arxiv.org/abs/2404.01318) | JailbreakBench是一个用于对抗大型语言模型越狱的开放基准，提供新的数据集、对抗提示和评估框架。 |
| [^3] | [Privacy-Aware Semantic Cache for Large Language Models](https://arxiv.org/abs/2403.02694) | MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。 |
| [^4] | [FairProof : Confidential and Certifiable Fairness for Neural Networks](https://arxiv.org/abs/2402.12572) | FairProof提出了一种使用零知识证明来公开验证神经网络模型公平性的系统，同时保持机密性，并提出了适用于ZKPs的全连接神经网络的公平性认证算法。 |
| [^5] | [Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD.](http://arxiv.org/abs/2307.00310) | 本文开发了一种新的DP-SGD分析方法，可以在训练过程中对许多数据点的隐私泄漏进行更准确的评估。 |

# 详细

[^1]: 超越请求：利用HTTP响应头在不平衡环境中进行跨浏览器Web追踪器分类

    Beyond the Request: Harnessing HTTP Response Headers for Cross-Browser Web Tracker Classification in an Imbalanced Setting

    [https://rss.arxiv.org/abs/2402.01240](https://rss.arxiv.org/abs/2402.01240)

    本研究通过利用HTTP响应头设计了机器学习分类器，在跨浏览器环境下有效检测Web追踪器，结果在Chrome和Firefox上表现出较高的准确性和性能。

    

    万维网的连通性主要归因于HTTP协议，其中的HTTP消息提供了有关网络安全和隐私的信息头字段，特别是关于Web追踪。尽管已有研究利用HTTP/S请求消息来识别Web追踪器，但往往忽视了HTTP/S响应头。本研究旨在设计使用HTTP/S响应头进行Web追踪器检测的有效机器学习分类器。通过浏览器扩展程序T.EX获取的Chrome、Firefox和Brave浏览器的数据作为我们的数据集。在Chrome数据上训练了11个监督模型，并在所有浏览器上进行了测试。结果表明，在Chrome和Firefox上具有高准确性、F1分数、精确度、召回率和最小对数损失误差的性能，但在Brave浏览器上表现不佳，可能是由于其不同的数据分布和特征集。研究表明，这些分类器可以用于检测Web追踪器。

    The World Wide Web's connectivity is greatly attributed to the HTTP protocol, with HTTP messages offering informative header fields that appeal to disciplines like web security and privacy, especially concerning web tracking. Despite existing research employing HTTP/S request messages to identify web trackers, HTTP/S response headers are often overlooked. This study endeavors to design effective machine learning classifiers for web tracker detection using HTTP/S response headers. Data from the Chrome, Firefox, and Brave browsers, obtained through the traffic monitoring browser extension T.EX, serves as our data set. Eleven supervised models were trained on Chrome data and tested across all browsers. The results demonstrated high accuracy, F1-score, precision, recall, and minimal log-loss error for Chrome and Firefox, but subpar performance on Brave, potentially due to its distinct data distribution and feature set. The research suggests that these classifiers are viable for detecting w
    
[^2]: JailbreakBench: 一个用于对抗大型语言模型越狱的开放鲁棒性基准

    JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models

    [https://arxiv.org/abs/2404.01318](https://arxiv.org/abs/2404.01318)

    JailbreakBench是一个用于对抗大型语言模型越狱的开放基准，提供新的数据集、对抗提示和评估框架。

    

    越狱攻击会导致大型语言模型生成有害、不道德或令人反感的内容。评估这些攻击存在许多挑战，当前的基准和评估技术并未充分解决。为了解决这些挑战，我们引入了JailbreakBench，一个开源基准，包括具有100个独特行为的新越狱数据集（称为JBB-Behaviors）、一组最先进的对抗提示（称为越狱工件）和一个标准化评估框架。

    arXiv:2404.01318v1 Announce Type: cross  Abstract: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) a new jailbreaking dataset containing 100 unique behaviors, which we call JBB-Behaviors; (2) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (3) a standardized evaluation framework that i
    
[^3]: 面向大型语言模型的隐私感知语义缓存

    Privacy-Aware Semantic Cache for Large Language Models

    [https://arxiv.org/abs/2403.02694](https://arxiv.org/abs/2403.02694)

    MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。

    

    大型语言模型（LLMs）如ChatGPT、Google Bard、Claude和Llama 2彻底改变了自然语言处理和搜索引擎动态。然而，这些模型造成了异常高的计算成本。本文介绍了MeanCache，一种用于LLMs的语义缓存，它能够识别语义上相似的查询以确定缓存命中或未命中。

    arXiv:2403.02694v1 Announce Type: cross  Abstract: Large Language Models (LLMs) like ChatGPT, Google Bard, Claude, and Llama 2 have revolutionized natural language processing and search engine dynamics. However, these models incur exceptionally high computational costs. For instance, GPT-3 consists of 175 billion parameters and inference on these models also demands billions of floating-point operations. Caching is a natural solution to reduce LLM inference costs on repeated queries. However, existing caching methods are incapable of finding semantic similarities among LLM queries, leading to unacceptable false hit-and-miss rates.   This paper introduces MeanCache, a semantic cache for LLMs that identifies semantically similar queries to determine cache hit or miss. Using MeanCache, the response to a user's semantically similar query can be retrieved from a local cache rather than re-querying the LLM, thus reducing costs, service provider load, and environmental impact. MeanCache lever
    
[^4]: FairProof：神经网络的机密和可认证公平性

    FairProof : Confidential and Certifiable Fairness for Neural Networks

    [https://arxiv.org/abs/2402.12572](https://arxiv.org/abs/2402.12572)

    FairProof提出了一种使用零知识证明来公开验证神经网络模型公平性的系统，同时保持机密性，并提出了适用于ZKPs的全连接神经网络的公平性认证算法。

    

    机器学习模型在社会应用中的使用越来越普遍，然而法律和隐私问题要求这些模型往往需要保密。因此，消费者对这些模型的公平性属性越来越不信任，消费者通常是模型预测的接收者。为此，我们提出了FairProof - 一种系统，使用零知识证明（一种密码原语）来公开验证模型的公平性，同时保持机密性。我们还提出了一个适合于ZKPs的全连接神经网络的公平性认证算法，并在该系统中使用。我们在Gnark中实现了FairProof，并通过实证证明了我们的系统是实际可行的。

    arXiv:2402.12572v1 Announce Type: cross  Abstract: Machine learning models are increasingly used in societal applications, yet legal and privacy concerns demand that they very often be kept confidential. Consequently, there is a growing distrust about the fairness properties of these models in the minds of consumers, who are often at the receiving end of model predictions. To this end, we propose FairProof - a system that uses Zero-Knowledge Proofs (a cryptographic primitive) to publicly verify the fairness of a model, while maintaining confidentiality. We also propose a fairness certification algorithm for fully-connected neural networks which is befitting to ZKPs and is used in this system. We implement FairProof in Gnark and demonstrate empirically that our system is practically feasible.
    
[^5]: 梯度相似：敏感度经常被过高估计在DP-SGD中

    Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD. (arXiv:2307.00310v1 [cs.LG])

    [http://arxiv.org/abs/2307.00310](http://arxiv.org/abs/2307.00310)

    本文开发了一种新的DP-SGD分析方法，可以在训练过程中对许多数据点的隐私泄漏进行更准确的评估。

    

    差分隐私随机梯度下降（DP-SGD）是私有深度学习的标准算法。虽然已知其隐私分析在最坏情况下是紧密的，但是一些实证结果表明，在常见的基准数据集上训练时，所得到的模型对许多数据点的隐私泄漏显著减少。在本文中，我们为DP-SGD开发了一种新的分析方法，捕捉到在数据集中具有相似邻居的点享受更好隐私性的直觉。形式上来说，这是通过修改从训练数据集计算得到的模型更新的每步隐私性分析来实现的。我们进一步开发了一个新的组合定理，以有效地利用这个新的每步分析来推理整个训练过程。总而言之，我们的评估结果表明，这种新颖的DP-SGD分析使我们能够正式地显示DP-SGD对许多数据点的隐私泄漏显著减少。

    Differentially private stochastic gradient descent (DP-SGD) is the canonical algorithm for private deep learning. While it is known that its privacy analysis is tight in the worst-case, several empirical results suggest that when training on common benchmark datasets, the models obtained leak significantly less privacy for many datapoints. In this paper, we develop a new analysis for DP-SGD that captures the intuition that points with similar neighbors in the dataset enjoy better privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints. In particular, we ob
    

