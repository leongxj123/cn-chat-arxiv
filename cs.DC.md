# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LoCoDL: Communication-Efficient Distributed Learning with Local Training and Compression](https://arxiv.org/abs/2403.04348) | LoCoDL是一种通信高效的分布式学习算法，结合了本地训练和压缩技术，具有双倍加速的通信复杂度优势，特别适用于一般异构条件下的强凸函数。 |
| [^2] | [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) | 本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。 |

# 详细

[^1]: LoCoDL: 具有本地训练和压缩的通信高效分布式学习

    LoCoDL: Communication-Efficient Distributed Learning with Local Training and Compression

    [https://arxiv.org/abs/2403.04348](https://arxiv.org/abs/2403.04348)

    LoCoDL是一种通信高效的分布式学习算法，结合了本地训练和压缩技术，具有双倍加速的通信复杂度优势，特别适用于一般异构条件下的强凸函数。

    

    在分布式优化和学习中，甚至在现代联邦学习框架中，由于通信速度慢且成本高，通信至关重要。我们介绍了LoCoDL，这是一种通信高效的算法，它利用了本地训练和压缩这两种流行且有效的技术，本地训练降低了通信频率，压缩则是发送短的比特流而不是完整的浮点数向量。LoCoDL适用于大类别的无偏压缩器，其中包括广泛使用的稀疏化和量化方法。LoCoDL在一般异构条件下具有双倍加速的通信复杂度优势，这取决于函数的条件数和模型维度，特别是在强凸函数的情况下。在实践中得到了验证，LoCoDL胜过了现有的算法。

    arXiv:2403.04348v1 Announce Type: cross  Abstract: In Distributed optimization and Learning, and even more in the modern framework of federated learning, communication, which is slow and costly, is critical. We introduce LoCoDL, a communication-efficient algorithm that leverages the two popular and effective techniques of Local training, which reduces the communication frequency, and Compression, in which short bitstreams are sent instead of full-dimensional vectors of floats. LoCoDL works with a large class of unbiased compressors that includes widely-used sparsification and quantization methods. LoCoDL provably benefits from local training and compression and enjoys a doubly-accelerated communication complexity, with respect to the condition number of the functions and the model dimension, in the general heterogenous regime with strongly convex functions. This is confirmed in practice, with LoCoDL outperforming existing algorithms.
    
[^2]: Fiddler：用于Mixture-of-Experts模型快速推断的CPU-GPU编排

    Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

    [https://arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033)

    本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。

    

    基于Mixture-of-Experts（MoE）架构的大型语言模型（LLM）在各种任务上表现出了很好的性能。然而，在资源受限的环境下运行这些模型，即GPU内存资源不丰富的情况下，由于模型规模庞大，存在挑战。现有的将模型权重卸载到CPU内存的系统，由于频繁地在CPU和GPU之间移动数据而导致显著的开销。在本文中，我们提出了Fiddler，一种用于MoE模型的资源高效推断引擎，实现了CPU-GPU编排。Fiddler的核心思想是利用CPU的计算能力来最小化CPU和GPU之间的数据传输。我们的评估结果表明，Fiddler能够在单个具有24GB内存的GPU上运行未压缩的Mixtral-8x7B模型（参数超过90GB），每秒生成超过3个token，相比现有方法提高一个数量级。Fiddler的代码可以公开访问，网址为\url{https://github.com/efeslab/fiddler}

    Large Language Models (LLMs) based on Mixture-of-Experts (MoE) architecture are showing promising performance on various tasks. However, running them on resource-constrained settings, where GPU memory resources are not abundant, is challenging due to huge model sizes. Existing systems that offload model weights to CPU memory suffer from the significant overhead of frequently moving data between CPU and GPU. In this paper, we propose Fiddler, a resource-efficient inference engine with CPU-GPU orchestration for MoE models. The key idea of Fiddler is to use the computation ability of the CPU to minimize the data movement between the CPU and GPU. Our evaluation shows that Fiddler can run the uncompressed Mixtral-8x7B model, which exceeds 90GB in parameters, to generate over $3$ tokens per second on a single GPU with 24GB memory, showing an order of magnitude improvement over existing methods. The code of Fiddler is publicly available at \url{https://github.com/efeslab/fiddler}
    

