# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression](https://arxiv.org/abs/2403.16677) | FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。 |

# 详细

[^1]: FOOL: 用神经特征压缩解决卫星计算中的下行瓶颈问题

    FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression

    [https://arxiv.org/abs/2403.16677](https://arxiv.org/abs/2403.16677)

    FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。

    

    具有传感器的纳卫星星座捕获大范围地理区域，为地球观测提供了前所未有的机会。随着星座规模的增加，网络争用形成了下行瓶颈。轨道边缘计算（OEC）利用有限的机载计算资源通过在源头处理原始捕获来减少传输成本。然而，由于依赖粗糙的过滤方法或过分优先考虑特定下游任务，目前的解决方案具有有限的实用性。本文提出了FOOL，一种OEC本地和任务不可知的特征压缩方法，可保留预测性能。FOOL将高分辨率卫星图像进行分区，以最大化吞吐量。此外，它嵌入上下文并利用瓷砖间的依赖关系，以较低的开销降低传输成本。虽然FOOL是一种特征压缩器，但它可以在低

    arXiv:2403.16677v1 Announce Type: new  Abstract: Nanosatellite constellations equipped with sensors capturing large geographic regions provide unprecedented opportunities for Earth observation. As constellation sizes increase, network contention poses a downlink bottleneck. Orbital Edge Computing (OEC) leverages limited onboard compute resources to reduce transfer costs by processing the raw captures at the source. However, current solutions have limited practicability due to reliance on crude filtering methods or over-prioritizing particular downstream tasks.   This work presents FOOL, an OEC-native and task-agnostic feature compression method that preserves prediction performance. FOOL partitions high-resolution satellite imagery to maximize throughput. Further, it embeds context and leverages inter-tile dependencies to lower transfer costs with negligible overhead. While FOOL is a feature compressor, it can recover images with competitive scores on perceptual quality measures at low
    

