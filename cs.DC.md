# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EdgeOL: Efficient in-situ Online Learning on Edge Devices.](http://arxiv.org/abs/2401.16694) | 本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。 |
| [^2] | [IoT in the Era of Generative AI: Vision and Challenges.](http://arxiv.org/abs/2401.01923) | 在生成式人工智能时代的物联网，Generative AI的进展带来了巨大的希望，同时也面临着高资源需求、及时工程、设备端推理、安全等关键挑战。 |
| [^3] | [An Experimental Comparison of Partitioning Strategies for Distributed Graph Neural Network Training.](http://arxiv.org/abs/2308.15602) | 本文研究了分布式图神经网络训练中分区策略的有效性，并探究了不同因素对分区效果的影响。 |
| [^4] | [SE-MoE: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System.](http://arxiv.org/abs/2205.10034) | SE-MoE提出了一种弹性的训练方式和不同优化措施，以提高混合专家模型的分布式训练和推理效率，解决了负载平衡、通信/计算效率和内存限制等问题。 |

# 详细

[^1]: EdgeOL: 边缘设备上高效的原位在线学习

    EdgeOL: Efficient in-situ Online Learning on Edge Devices. (arXiv:2401.16694v1 [cs.LG])

    [http://arxiv.org/abs/2401.16694](http://arxiv.org/abs/2401.16694)

    本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。

    

    新兴应用，如机器人辅助养老和物体识别，通常采用深度学习神经网络模型，并且自然需要：i) 处理实时推理请求和ii) 适应可能的部署场景变化。在线模型微调被广泛采用以满足这些需求。然而，微调会导致显著的能量消耗，使其难以部署在边缘设备上。在本文中，我们提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率。实验结果显示，EdgeOL平均减少了82%的微调执行时间，74%的能量消耗，并提高了平均推理准确率1.70%，相对于即时在线学习策略。

    Emerging applications, such as robot-assisted eldercare and object recognition, generally employ deep learning neural networks (DNNs) models and naturally require: i) handling streaming-in inference requests and ii) adapting to possible deployment scenario changes. Online model fine-tuning is widely adopted to satisfy these needs. However, fine-tuning involves significant energy consumption, making it challenging to deploy on edge devices. In this paper, we propose EdgeOL, an edge online learning framework that optimizes inference accuracy, fine-tuning execution time, and energy efficiency through both inter-tuning and intra-tuning optimizations. Experimental results show that, on average, EdgeOL reduces overall fine-tuning execution time by 82%, energy consumption by 74%, and improves average inference accuracy by 1.70% over the immediate online learning strategy.
    
[^2]: 在生成式人工智能时代的物联网: 视野与挑战

    IoT in the Era of Generative AI: Vision and Challenges. (arXiv:2401.01923v1 [cs.DC])

    [http://arxiv.org/abs/2401.01923](http://arxiv.org/abs/2401.01923)

    在生成式人工智能时代的物联网，Generative AI的进展带来了巨大的希望，同时也面临着高资源需求、及时工程、设备端推理、安全等关键挑战。

    

    带有感知、网络和计算能力的物联网设备，如智能手机、可穿戴设备、智能音箱和家庭机器人，已经无缝地融入到我们的日常生活中。最近生成式人工智能（Generative AI）的进展，如GPT、LLaMA、DALL-E和稳定扩散等，给物联网的发展带来了巨大的希望。本文分享了我们对Generative AI在物联网中带来的好处的看法和愿景，并讨论了Generative AI在物联网相关领域的一些重要应用。充分利用Generative AI在物联网中是一个复杂的挑战。我们确定了一些最关键的挑战，包括Generative AI模型的高资源需求、及时工程、设备端推理、卸载、设备端微调、联邦学习、安全以及开发工具和基准，并讨论了当前存在的差距以及使Generative AI在物联网中实现的有希望的机会。我们希望这篇文章能够激发新的研究和创新。

    Equipped with sensing, networking, and computing capabilities, Internet of Things (IoT) such as smartphones, wearables, smart speakers, and household robots have been seamlessly weaved into our daily lives. Recent advancements in Generative AI exemplified by GPT, LLaMA, DALL-E, and Stable Difussion hold immense promise to push IoT to the next level. In this article, we share our vision and views on the benefits that Generative AI brings to IoT, and discuss some of the most important applications of Generative AI in IoT-related domains. Fully harnessing Generative AI in IoT is a complex challenge. We identify some of the most critical challenges including high resource demands of the Generative AI models, prompt engineering, on-device inference, offloading, on-device fine-tuning, federated learning, security, as well as development tools and benchmarks, and discuss current gaps as well as promising opportunities on enabling Generative AI for IoT. We hope this article can inspire new res
    
[^3]: 分布式图神经网络训练的分区策略的实验比较

    An Experimental Comparison of Partitioning Strategies for Distributed Graph Neural Network Training. (arXiv:2308.15602v1 [cs.DC])

    [http://arxiv.org/abs/2308.15602](http://arxiv.org/abs/2308.15602)

    本文研究了分布式图神经网络训练中分区策略的有效性，并探究了不同因素对分区效果的影响。

    

    最近，图神经网络（GNNs）作为一种能够在图结构化数据上学习的深度学习领域，受到了广泛关注。然而，对于大规模图上的GNN训练，计算和内存要求可能超过单台机器或GPU的能力，因此分布式GNN训练成为大规模GNN训练的有前途的方向。分布式GNN训练的先决条件是将输入图分割成较小的部分，这些部分分布在计算集群的多台机器间。虽然图分区在图分析和图数据库方面已经得到了广泛研究，但其对GNN训练性能的影响尚未得到深入探索。在本文中，我们研究了分区对分布式GNN训练的效果。我们的研究旨在了解不同因素（如GNN参数、小批量大小、图类型、特征大小和扩展因子）对分区效果的影响。

    Recently, graph neural networks (GNNs) have gained much attention as a growing area of deep learning capable of learning on graph-structured data. However, the computational and memory requirements for training GNNs on large-scale graphs can exceed the capabilities of single machines or GPUs, making distributed GNN training a promising direction for large-scale GNN training. A prerequisite for distributed GNN training is to partition the input graph into smaller parts that are distributed among multiple machines of a compute cluster. Although graph partitioning has been extensively studied with regard to graph analytics and graph databases, its effect on GNN training performance is largely unexplored.  In this paper, we study the effectiveness of graph partitioning for distributed GNN training. Our study aims to understand how different factors such as GNN parameters, mini-batch size, graph type, features size, and scale-out factor influence the effectiveness of graph partitioning. We 
    
[^4]: SE-MoE: 一种可扩展高效的分布式训练和推理混合专家系统

    SE-MoE: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System. (arXiv:2205.10034v2 [cs.DC] UPDATED)

    [http://arxiv.org/abs/2205.10034](http://arxiv.org/abs/2205.10034)

    SE-MoE提出了一种弹性的训练方式和不同优化措施，以提高混合专家模型的分布式训练和推理效率，解决了负载平衡、通信/计算效率和内存限制等问题。

    

    随着机器学习架构的多样性增加，将模型分布式训练在异构计算系统上以方便生成大模型成为了一种趋势。混合专家模型利用分治策略通过门控和并行处理方式来降低训练成本并控制总体模型/数据大小。虽然 DeepSpeed 在进行大规模 MoE 训练上进行了尝试，但训练和推理的效率仍有提升的空间，包括负载平衡、通信/计算效率和内存限制等方面。本工作提出了 SE-MoE，使用弹性 MoE 训练、基于层次存储的 2D 预取和融合通信等方法，以便在不同的类型中获得高效的并行处理。针对单节点的可扩展推理，特别是当模型大小大于 GPU 内存时，SE-MoE 将 CPU-GPU 存储联合形成一个更大的存储空间。

    With the increasing diversity of ML infrastructures nowadays, distributed training over heterogeneous computing systems is desired to facilitate the production of big models. Mixture-of-Experts (MoE) models have been proposed to lower the cost of training subject to the overall size of models/data through gating and parallelism in a divide-and-conquer fashion. While DeepSpeed has made efforts in carrying out large-scale MoE training over heterogeneous infrastructures, the efficiency of training and inference could be further improved from several system aspects, including load balancing, communication/computation efficiency, and memory footprint limits. In this work, we present SE-MoE that proposes Elastic MoE training with 2D prefetch and Fusion communication over Hierarchical storage, so as to enjoy efficient parallelisms in various types. For scalable inference in a single node, especially when the model size is larger than GPU memory, SE-MoE forms the CPU-GPU memory jointly into a 
    

