# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression](https://arxiv.org/abs/2403.16677) | FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。 |
| [^2] | [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) | 本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。 |

# 详细

[^1]: FOOL: 用神经特征压缩解决卫星计算中的下行瓶颈问题

    FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression

    [https://arxiv.org/abs/2403.16677](https://arxiv.org/abs/2403.16677)

    FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。

    

    具有传感器的纳卫星星座捕获大范围地理区域，为地球观测提供了前所未有的机会。随着星座规模的增加，网络争用形成了下行瓶颈。轨道边缘计算（OEC）利用有限的机载计算资源通过在源头处理原始捕获来减少传输成本。然而，由于依赖粗糙的过滤方法或过分优先考虑特定下游任务，目前的解决方案具有有限的实用性。本文提出了FOOL，一种OEC本地和任务不可知的特征压缩方法，可保留预测性能。FOOL将高分辨率卫星图像进行分区，以最大化吞吐量。此外，它嵌入上下文并利用瓷砖间的依赖关系，以较低的开销降低传输成本。虽然FOOL是一种特征压缩器，但它可以在低

    arXiv:2403.16677v1 Announce Type: new  Abstract: Nanosatellite constellations equipped with sensors capturing large geographic regions provide unprecedented opportunities for Earth observation. As constellation sizes increase, network contention poses a downlink bottleneck. Orbital Edge Computing (OEC) leverages limited onboard compute resources to reduce transfer costs by processing the raw captures at the source. However, current solutions have limited practicability due to reliance on crude filtering methods or over-prioritizing particular downstream tasks.   This work presents FOOL, an OEC-native and task-agnostic feature compression method that preserves prediction performance. FOOL partitions high-resolution satellite imagery to maximize throughput. Further, it embeds context and leverages inter-tile dependencies to lower transfer costs with negligible overhead. While FOOL is a feature compressor, it can recover images with competitive scores on perceptual quality measures at low
    
[^2]: Fiddler：用于Mixture-of-Experts模型快速推断的CPU-GPU编排

    Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

    [https://arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033)

    本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。

    

    基于Mixture-of-Experts（MoE）架构的大型语言模型（LLM）在各种任务上表现出了很好的性能。然而，在资源受限的环境下运行这些模型，即GPU内存资源不丰富的情况下，由于模型规模庞大，存在挑战。现有的将模型权重卸载到CPU内存的系统，由于频繁地在CPU和GPU之间移动数据而导致显著的开销。在本文中，我们提出了Fiddler，一种用于MoE模型的资源高效推断引擎，实现了CPU-GPU编排。Fiddler的核心思想是利用CPU的计算能力来最小化CPU和GPU之间的数据传输。我们的评估结果表明，Fiddler能够在单个具有24GB内存的GPU上运行未压缩的Mixtral-8x7B模型（参数超过90GB），每秒生成超过3个token，相比现有方法提高一个数量级。Fiddler的代码可以公开访问，网址为\url{https://github.com/efeslab/fiddler}

    Large Language Models (LLMs) based on Mixture-of-Experts (MoE) architecture are showing promising performance on various tasks. However, running them on resource-constrained settings, where GPU memory resources are not abundant, is challenging due to huge model sizes. Existing systems that offload model weights to CPU memory suffer from the significant overhead of frequently moving data between CPU and GPU. In this paper, we propose Fiddler, a resource-efficient inference engine with CPU-GPU orchestration for MoE models. The key idea of Fiddler is to use the computation ability of the CPU to minimize the data movement between the CPU and GPU. Our evaluation shows that Fiddler can run the uncompressed Mixtral-8x7B model, which exceeds 90GB in parameters, to generate over $3$ tokens per second on a single GPU with 24GB memory, showing an order of magnitude improvement over existing methods. The code of Fiddler is publicly available at \url{https://github.com/efeslab/fiddler}
    

