# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression](https://arxiv.org/abs/2403.16677) | FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。 |
| [^2] | [EyePreserve: Identity-Preserving Iris Synthesis.](http://arxiv.org/abs/2312.12028) | 本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法，能够合成不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在保持身份的同时进行非线性纹理变形。 |

# 详细

[^1]: FOOL: 用神经特征压缩解决卫星计算中的下行瓶颈问题

    FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression

    [https://arxiv.org/abs/2403.16677](https://arxiv.org/abs/2403.16677)

    FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。

    

    具有传感器的纳卫星星座捕获大范围地理区域，为地球观测提供了前所未有的机会。随着星座规模的增加，网络争用形成了下行瓶颈。轨道边缘计算（OEC）利用有限的机载计算资源通过在源头处理原始捕获来减少传输成本。然而，由于依赖粗糙的过滤方法或过分优先考虑特定下游任务，目前的解决方案具有有限的实用性。本文提出了FOOL，一种OEC本地和任务不可知的特征压缩方法，可保留预测性能。FOOL将高分辨率卫星图像进行分区，以最大化吞吐量。此外，它嵌入上下文并利用瓷砖间的依赖关系，以较低的开销降低传输成本。虽然FOOL是一种特征压缩器，但它可以在低

    arXiv:2403.16677v1 Announce Type: new  Abstract: Nanosatellite constellations equipped with sensors capturing large geographic regions provide unprecedented opportunities for Earth observation. As constellation sizes increase, network contention poses a downlink bottleneck. Orbital Edge Computing (OEC) leverages limited onboard compute resources to reduce transfer costs by processing the raw captures at the source. However, current solutions have limited practicability due to reliance on crude filtering methods or over-prioritizing particular downstream tasks.   This work presents FOOL, an OEC-native and task-agnostic feature compression method that preserves prediction performance. FOOL partitions high-resolution satellite imagery to maximize throughput. Further, it embeds context and leverages inter-tile dependencies to lower transfer costs with negligible overhead. While FOOL is a feature compressor, it can recover images with competitive scores on perceptual quality measures at low
    
[^2]: EyePreserve: 保持身份的虹膜合成

    EyePreserve: Identity-Preserving Iris Synthesis. (arXiv:2312.12028v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.12028](http://arxiv.org/abs/2312.12028)

    本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法，能够合成不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在保持身份的同时进行非线性纹理变形。

    

    在广泛的瞳孔尺寸范围内保持身份的同身份生物特征虹膜图像的合成是复杂的，因为它涉及到虹膜肌肉收缩机制，需要将虹膜非线性纹理变形模型嵌入到合成流程中。本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法。这种方法能够合成具有不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在给定目标虹膜图像的分割掩膜下非线性地变形现有主体的虹膜图像纹理。虹膜识别实验表明，所提出的变形模型不仅在改变瞳孔尺寸时保持身份，而且在瞳孔尺寸有显著差异的同身份虹膜样本之间提供更好的相似度，与最先进的线性方法相比。

    Synthesis of same-identity biometric iris images, both for existing and non-existing identities while preserving the identity across a wide range of pupil sizes, is complex due to intricate iris muscle constriction mechanism, requiring a precise model of iris non-linear texture deformations to be embedded into the synthesis pipeline. This paper presents the first method of fully data-driven, identity-preserving, pupil size-varying s ynthesis of iris images. This approach is capable of synthesizing images of irises with different pupil sizes representing non-existing identities as well as non-linearly deforming the texture of iris images of existing subjects given the segmentation mask of the target iris image. Iris recognition experiments suggest that the proposed deformation model not only preserves the identity when changing the pupil size but offers better similarity between same-identity iris samples with significant differences in pupil size, compared to state-of-the-art linear an
    

