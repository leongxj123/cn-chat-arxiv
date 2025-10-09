# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differential Privacy for Adaptive Weight Aggregation in Federated Tumor Segmentation.](http://arxiv.org/abs/2308.00856) | 本研究提出了一种针对联邦肿瘤分割中自适应权重聚合的差分隐私算法，通过扩展相似性权重聚合方法（SimAgg），提高了模型分割能力，并在保护隐私方面做出了额外改进。 |
| [^2] | [Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples.](http://arxiv.org/abs/2209.03358) | 这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。 |

# 详细

[^1]: 针对联邦肿瘤分割中自适应权重聚合的差分隐私研究

    Differential Privacy for Adaptive Weight Aggregation in Federated Tumor Segmentation. (arXiv:2308.00856v1 [cs.LG])

    [http://arxiv.org/abs/2308.00856](http://arxiv.org/abs/2308.00856)

    本研究提出了一种针对联邦肿瘤分割中自适应权重聚合的差分隐私算法，通过扩展相似性权重聚合方法（SimAgg），提高了模型分割能力，并在保护隐私方面做出了额外改进。

    

    联邦学习是一种分布式机器学习方法，通过创建一个公正的全局模型来保护个体客户数据的隐私。然而，传统的联邦学习方法在处理不同客户数据时可能引入安全风险，从而可能危及隐私和数据完整性。为了解决这些挑战，本文提出了一种差分隐私联邦深度学习框架，在医学图像分割中扩展了相似性权重聚合方法（SimAgg）到DP-SimAgg算法，这是一种针对多模态磁共振成像（MRI）中的脑肿瘤分割的差分隐私相似性加权聚合算法。我们的DP-SimAgg方法不仅提高了模型分割能力，还提供了额外的隐私保护层。通过广泛的基准测试和评估，以计算性能为主要考虑因素，证明了DP-SimAgg使..

    Federated Learning (FL) is a distributed machine learning approach that safeguards privacy by creating an impartial global model while respecting the privacy of individual client data. However, the conventional FL method can introduce security risks when dealing with diverse client data, potentially compromising privacy and data integrity. To address these challenges, we present a differential privacy (DP) federated deep learning framework in medical image segmentation. In this paper, we extend our similarity weight aggregation (SimAgg) method to DP-SimAgg algorithm, a differentially private similarity-weighted aggregation algorithm for brain tumor segmentation in multi-modal magnetic resonance imaging (MRI). Our DP-SimAgg method not only enhances model segmentation capabilities but also provides an additional layer of privacy preservation. Extensive benchmarking and evaluation of our framework, with computational performance as a key consideration, demonstrate that DP-SimAgg enables a
    
[^2]: 攻击脉冲：关于脉冲神经网络对抗性样本的可转移性与安全性的研究

    Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples. (arXiv:2209.03358v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2209.03358](http://arxiv.org/abs/2209.03358)

    这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。

    

    脉冲神经网络(SNNs)因其高能效和最近在分类性能上的进展而受到广泛关注。然而，与传统的深度学习方法不同，对SNNs对抗性样本的鲁棒性的分析和研究仍然相对不完善。在这项工作中，我们关注于推进SNNs的对抗攻击方面，并做出了三个主要贡献。首先，我们展示了成功的白盒对抗攻击SNNs在很大程度上依赖于底层的替代梯度技术，即使在对抗性训练SNNs的情况下也一样。其次，利用最佳的替代梯度技术，我们分析了对抗攻击在SNNs和其他最先进的架构如Vision Transformers(ViTs)和Big Transfer Convolutional Neural Networks(CNNs)之间的可转移性。我们证明了非SNN架构创建的对抗样本往往不被SNNs误分类。第三，由于缺乏一个共性

    Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remain relatively underdeveloped. In this work, we focus on advancing the adversarial attack side of SNNs and make three major contributions. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique, even in the case of adversarially trained SNNs. Second, using the best surrogate gradient technique, we analyze the transferability of adversarial attacks on SNNs and other state-of-the-art architectures like Vision Transformers (ViTs) and Big Transfer Convolutional Neural Networks (CNNs). We demonstrate that the adversarial examples created by non-SNN architectures are not misclassified often by SNNs. Third, due to the lack of an ubi
    

