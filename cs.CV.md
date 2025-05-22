# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalizing Medical Image Representations via Quaternion Wavelet Networks.](http://arxiv.org/abs/2310.10224) | 本文提出了一种名为QUAVE的四元数小波网络，可以从医学图像中提取显著特征。该网络可以与现有的医学图像分析或综合任务结合使用，并推广了对单通道数据的采用。通过四元数小波变换和加权处理，QUAVE能够处理具有较大变化的医学数据。 |
| [^2] | [SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network.](http://arxiv.org/abs/2310.06488) | 本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。 |

# 详细

[^1]: 通过四元数小波网络推广医学图像表示

    Generalizing Medical Image Representations via Quaternion Wavelet Networks. (arXiv:2310.10224v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2310.10224](http://arxiv.org/abs/2310.10224)

    本文提出了一种名为QUAVE的四元数小波网络，可以从医学图像中提取显著特征。该网络可以与现有的医学图像分析或综合任务结合使用，并推广了对单通道数据的采用。通过四元数小波变换和加权处理，QUAVE能够处理具有较大变化的医学数据。

    

    鉴于来自不同来源和各种任务的数据集日益增加，神经网络的普适性成为一个广泛研究的领域。当处理医学数据时，这个问题尤为广泛，因为缺乏方法论标准导致不同的成像中心或使用不同设备和辅助因素获取的数据存在较大变化。为了克服这些限制，我们引入了一种新颖的、普适的、数据-和任务不可知的框架，能够从医学图像中提取显著特征。所提出的四元数小波网络（QUAVE）可以很容易地与任何现有的医学图像分析或综合任务相结合，并且可以结合实际、四元数或超复值模型，推广它们对单通道数据的采用。QUAVE首先通过四元数小波变换提取不同的子带，得到低频/近似频带和高频/细粒度特征。然后，它对最有代表性的特征进行加权处理，从而减少了特征重要性不均匀性。

    Neural network generalizability is becoming a broad research field due to the increasing availability of datasets from different sources and for various tasks. This issue is even wider when processing medical data, where a lack of methodological standards causes large variations being provided by different imaging centers or acquired with various devices and cofactors. To overcome these limitations, we introduce a novel, generalizable, data- and task-agnostic framework able to extract salient features from medical images. The proposed quaternion wavelet network (QUAVE) can be easily integrated with any pre-existing medical image analysis or synthesis task, and it can be involved with real, quaternion, or hypercomplex-valued models, generalizing their adoption to single-channel data. QUAVE first extracts different sub-bands through the quaternion wavelet transform, resulting in both low-frequency/approximation bands and high-frequency/fine-grained features. Then, it weighs the most repr
    
[^2]: SpikeCLIP：一种对比语言-图像预训练脉冲神经网络

    SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network. (arXiv:2310.06488v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2310.06488](http://arxiv.org/abs/2310.06488)

    本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。

    

    脉冲神经网络（SNNs）已经证明其在视觉和语言领域中能够实现与深度神经网络（DNNs）相当的性能，同时具有能效提高和符合生物合理性的优势。然而，将这种单模态的SNNs扩展到多模态的情景仍然是一个未开发的领域。受到对比语言-图像预训练（CLIP）概念的启发，我们引入了一个名为SpikeCLIP的新框架，通过“对齐预训练+双损失微调”的两步骤配方，来解决脉冲计算背景下两种模态之间的差距。广泛的实验证明，在常用的用于多模态模型评估的各种数据集上，SNNs取得了与其DNNs对应物相当的结果，同时显著降低了能源消耗。此外，SpikeCLIP在图像分类方面保持了稳定的性能。

    Spiking neural networks (SNNs) have demonstrated the capability to achieve comparable performance to deep neural networks (DNNs) in both visual and linguistic domains while offering the advantages of improved energy efficiency and adherence to biological plausibility. However, the extension of such single-modality SNNs into the realm of multimodal scenarios remains an unexplored territory. Drawing inspiration from the concept of contrastive language-image pre-training (CLIP), we introduce a novel framework, named SpikeCLIP, to address the gap between two modalities within the context of spike-based computing through a two-step recipe involving ``Alignment Pre-training + Dual-Loss Fine-tuning". Extensive experiments demonstrate that SNNs achieve comparable results to their DNN counterparts while significantly reducing energy consumption across a variety of datasets commonly used for multimodal model evaluation. Furthermore, SpikeCLIP maintains robust performance in image classification 
    

