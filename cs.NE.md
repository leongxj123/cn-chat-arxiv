# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network.](http://arxiv.org/abs/2310.06488) | 本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。 |

# 详细

[^1]: SpikeCLIP：一种对比语言-图像预训练脉冲神经网络

    SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network. (arXiv:2310.06488v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2310.06488](http://arxiv.org/abs/2310.06488)

    本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。

    

    脉冲神经网络（SNNs）已经证明其在视觉和语言领域中能够实现与深度神经网络（DNNs）相当的性能，同时具有能效提高和符合生物合理性的优势。然而，将这种单模态的SNNs扩展到多模态的情景仍然是一个未开发的领域。受到对比语言-图像预训练（CLIP）概念的启发，我们引入了一个名为SpikeCLIP的新框架，通过“对齐预训练+双损失微调”的两步骤配方，来解决脉冲计算背景下两种模态之间的差距。广泛的实验证明，在常用的用于多模态模型评估的各种数据集上，SNNs取得了与其DNNs对应物相当的结果，同时显著降低了能源消耗。此外，SpikeCLIP在图像分类方面保持了稳定的性能。

    Spiking neural networks (SNNs) have demonstrated the capability to achieve comparable performance to deep neural networks (DNNs) in both visual and linguistic domains while offering the advantages of improved energy efficiency and adherence to biological plausibility. However, the extension of such single-modality SNNs into the realm of multimodal scenarios remains an unexplored territory. Drawing inspiration from the concept of contrastive language-image pre-training (CLIP), we introduce a novel framework, named SpikeCLIP, to address the gap between two modalities within the context of spike-based computing through a two-step recipe involving ``Alignment Pre-training + Dual-Loss Fine-tuning". Extensive experiments demonstrate that SNNs achieve comparable results to their DNN counterparts while significantly reducing energy consumption across a variety of datasets commonly used for multimodal model evaluation. Furthermore, SpikeCLIP maintains robust performance in image classification 
    

