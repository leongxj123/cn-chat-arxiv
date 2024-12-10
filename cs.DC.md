# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SPEAR:Exact Gradient Inversion of Batches in Federated Learning](https://arxiv.org/abs/2403.03945) | 该论文提出了第一个能够精确重构批量$b >1$的算法，在联邦学习中解决了梯度反演攻击的问题。 |
| [^2] | [IoT in the Era of Generative AI: Vision and Challenges.](http://arxiv.org/abs/2401.01923) | 在生成式人工智能时代的物联网，Generative AI的进展带来了巨大的希望，同时也面临着高资源需求、及时工程、设备端推理、安全等关键挑战。 |

# 详细

[^1]: SPEAR：联邦学习中批量精确梯度反演

    SPEAR:Exact Gradient Inversion of Batches in Federated Learning

    [https://arxiv.org/abs/2403.03945](https://arxiv.org/abs/2403.03945)

    该论文提出了第一个能够精确重构批量$b >1$的算法，在联邦学习中解决了梯度反演攻击的问题。

    

    联邦学习是一种流行的协作机器学习框架，在这个框架中，多个客户端仅与服务器共享他们本地数据的梯度更新，而不是实际数据。不幸的是，最近发现梯度反演攻击可以从这些共享的梯度中重构出数据。现有的攻击只能在重要的诚实但好奇设置中对批量大小为$b=1$的数据进行精确重构，对于更大的批量只能进行近似重构。在这项工作中，我们提出了\emph{第一个准确重建批量$b >1$的算法}。这种方法结合了对梯度显式低秩结构的数学见解和基于采样的算法。关键的是，我们利用ReLU诱导的梯度稀疏性，精确地过滤掉大量错误的样本，使最终的重建步骤可行。我们为全连接提供了高效的GPU实现

    arXiv:2403.03945v1 Announce Type: new  Abstract: Federated learning is a popular framework for collaborative machine learning where multiple clients only share gradient updates on their local data with the server and not the actual data. Unfortunately, it was recently shown that gradient inversion attacks can reconstruct this data from these shared gradients. Existing attacks enable exact reconstruction only for a batch size of $b=1$ in the important honest-but-curious setting, with larger batches permitting only approximate reconstruction. In this work, we propose \emph{the first algorithm reconstructing whole batches with $b >1$ exactly}. This approach combines mathematical insights into the explicit low-rank structure of gradients with a sampling-based algorithm. Crucially, we leverage ReLU-induced gradient sparsity to precisely filter out large numbers of incorrect samples, making a final reconstruction step tractable. We provide an efficient GPU implementation for fully connected 
    
[^2]: 在生成式人工智能时代的物联网: 视野与挑战

    IoT in the Era of Generative AI: Vision and Challenges. (arXiv:2401.01923v1 [cs.DC])

    [http://arxiv.org/abs/2401.01923](http://arxiv.org/abs/2401.01923)

    在生成式人工智能时代的物联网，Generative AI的进展带来了巨大的希望，同时也面临着高资源需求、及时工程、设备端推理、安全等关键挑战。

    

    带有感知、网络和计算能力的物联网设备，如智能手机、可穿戴设备、智能音箱和家庭机器人，已经无缝地融入到我们的日常生活中。最近生成式人工智能（Generative AI）的进展，如GPT、LLaMA、DALL-E和稳定扩散等，给物联网的发展带来了巨大的希望。本文分享了我们对Generative AI在物联网中带来的好处的看法和愿景，并讨论了Generative AI在物联网相关领域的一些重要应用。充分利用Generative AI在物联网中是一个复杂的挑战。我们确定了一些最关键的挑战，包括Generative AI模型的高资源需求、及时工程、设备端推理、卸载、设备端微调、联邦学习、安全以及开发工具和基准，并讨论了当前存在的差距以及使Generative AI在物联网中实现的有希望的机会。我们希望这篇文章能够激发新的研究和创新。

    Equipped with sensing, networking, and computing capabilities, Internet of Things (IoT) such as smartphones, wearables, smart speakers, and household robots have been seamlessly weaved into our daily lives. Recent advancements in Generative AI exemplified by GPT, LLaMA, DALL-E, and Stable Difussion hold immense promise to push IoT to the next level. In this article, we share our vision and views on the benefits that Generative AI brings to IoT, and discuss some of the most important applications of Generative AI in IoT-related domains. Fully harnessing Generative AI in IoT is a complex challenge. We identify some of the most critical challenges including high resource demands of the Generative AI models, prompt engineering, on-device inference, offloading, on-device fine-tuning, federated learning, security, as well as development tools and benchmarks, and discuss current gaps as well as promising opportunities on enabling Generative AI for IoT. We hope this article can inspire new res
    

