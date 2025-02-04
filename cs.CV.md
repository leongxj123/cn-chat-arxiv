# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DSEG-LIME -- Improving Image Explanation by Hierarchical Data-Driven Segmentation](https://arxiv.org/abs/2403.07733) | 通过引入数据驱动分割和层次分割程序，DSEG-LIME改进了图像解释能力，提高了图像分类的可解释性。 |
| [^2] | [Group Distributionally Robust Dataset Distillation with Risk Minimization](https://arxiv.org/abs/2402.04676) | 这项研究关注数据集蒸馏与其泛化能力的关系，尤其是在面对不常见的子组的样本时，如何确保模型在合成数据集上的训练可以表现良好。 |
| [^3] | [Humans Beat Deep Networks at Recognizing Objects in Unusual Poses, Given Enough Time](https://arxiv.org/abs/2402.03973) | 人类在识别不寻常姿势中的物体上表现优于深度网络，当给予足够时间时。然而，随着图像曝光时间的限制，人类的表现降至深度网络的水平，这暗示人类在识别不寻常姿势中的物体时需要额外的心理过程。此外，人类与网络之间的错误模式也存在不同。因此，我们需要进一步研究，以提高计算机视觉系统的鲁棒性水平。 |
| [^4] | [Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets.](http://arxiv.org/abs/2401.14497) | 本文研究了DermaMNIST和Fitzpatrick17k皮肤科图像数据集的质量问题，对数据重复、数据泄漏、错误标记和缺乏测试分区等方面进行了详细分析，并提出纠正措施。 |
| [^5] | [Towards the Vulnerability of Watermarking Artificial Intelligence Generated Content.](http://arxiv.org/abs/2310.07726) | 该研究探讨了将水印技术应用于人工智能生成内容的漏洞，并证明了现有的水印机制容易被对手破解。 |

# 详细

[^1]: DSEG-LIME -- 通过层次化数据驱动分割提升图像解释能力

    DSEG-LIME -- Improving Image Explanation by Hierarchical Data-Driven Segmentation

    [https://arxiv.org/abs/2403.07733](https://arxiv.org/abs/2403.07733)

    通过引入数据驱动分割和层次分割程序，DSEG-LIME改进了图像解释能力，提高了图像分类的可解释性。

    

    可解释的人工智能在揭示复杂机器学习模型的决策过程中至关重要。LIME (Local Interpretable Model-agnostic Explanations) 是一个广为人知的用于图像分析的XAI框架。它利用图像分割来创建特征以识别相关的分类区域。然而，较差的分割可能会影响解释的一致性并削弱各个区域的重要性，从而影响整体的可解释性。针对这些挑战，我们引入了DSEG-LIME (Data-Driven Segmentation LIME)，具有: i) 用于生成人类可识别特征的数据驱动分割, 和 ii) 通过组合实现的层次分割程序。我们在预训练模型上使用来自ImageNet数据集的图像对DSEG-LIME进行基准测试-这些情景不包含特定领域的知识。分析包括使用已建立的XAI指标进行定量评估，以及进一步的定性评估。

    arXiv:2403.07733v1 Announce Type: cross  Abstract: Explainable Artificial Intelligence is critical in unraveling decision-making processes in complex machine learning models. LIME (Local Interpretable Model-agnostic Explanations) is a well-known XAI framework for image analysis. It utilizes image segmentation to create features to identify relevant areas for classification. Consequently, poor segmentation can compromise the consistency of the explanation and undermine the importance of the segments, affecting the overall interpretability. Addressing these challenges, we introduce DSEG-LIME (Data-Driven Segmentation LIME), featuring: i) a data-driven segmentation for human-recognized feature generation, and ii) a hierarchical segmentation procedure through composition. We benchmark DSEG-LIME on pre-trained models with images from the ImageNet dataset - scenarios without domain-specific knowledge. The analysis includes a quantitative evaluation using established XAI metrics, complemented
    
[^2]: 带风险最小化的分组分布鲁棒数据集蒸馏

    Group Distributionally Robust Dataset Distillation with Risk Minimization

    [https://arxiv.org/abs/2402.04676](https://arxiv.org/abs/2402.04676)

    这项研究关注数据集蒸馏与其泛化能力的关系，尤其是在面对不常见的子组的样本时，如何确保模型在合成数据集上的训练可以表现良好。

    

    数据集蒸馏（DD）已成为一种广泛采用的技术，用于构建一个合成数据集，该数据集在捕捉训练数据集的基本信息方面起到重要作用，从而方便准确训练神经模型。其应用涵盖了转移学习、联邦学习和神经架构搜索等各个领域。构建合成数据的最流行方法依赖于使模型在合成数据集和训练数据集上的收敛性能相匹配。然而，目标是将训练数据集视为辅助，就像训练集是人口分布的近似替代品一样，而后者才是我们感兴趣的数据。尽管其受欢迎程度很高，但尚未探索的一个方面是DD与其泛化能力的关系，特别是跨不常见的子组。也就是说，当面对来自罕见子组的样本时，我们如何确保在合成数据集上训练的模型表现良好。

    Dataset distillation (DD) has emerged as a widely adopted technique for crafting a synthetic dataset that captures the essential information of a training dataset, facilitating the training of accurate neural models. Its applications span various domains, including transfer learning, federated learning, and neural architecture search. The most popular methods for constructing the synthetic data rely on matching the convergence properties of training the model with the synthetic dataset and the training dataset. However, targeting the training dataset must be thought of as auxiliary in the same sense that the training set is an approximate substitute for the population distribution, and the latter is the data of interest. Yet despite its popularity, an aspect that remains unexplored is the relationship of DD to its generalization, particularly across uncommon subgroups. That is, how can we ensure that a model trained on the synthetic dataset performs well when faced with samples from re
    
[^3]: 给予足够时间，人类在识别不寻常姿势中的物体上击败了深度网络

    Humans Beat Deep Networks at Recognizing Objects in Unusual Poses, Given Enough Time

    [https://arxiv.org/abs/2402.03973](https://arxiv.org/abs/2402.03973)

    人类在识别不寻常姿势中的物体上表现优于深度网络，当给予足够时间时。然而，随着图像曝光时间的限制，人类的表现降至深度网络的水平，这暗示人类在识别不寻常姿势中的物体时需要额外的心理过程。此外，人类与网络之间的错误模式也存在不同。因此，我们需要进一步研究，以提高计算机视觉系统的鲁棒性水平。

    

    深度学习在几个物体识别基准上正在缩小与人类的差距。本文在涉及从不寻常视角观察物体的挑战性图像中对这一差距进行了研究。我们发现人类在识别不寻常姿势中的物体时表现出色，与先进的预训练网络（EfficientNet、SWAG、ViT、SWIN、BEiT、ConvNext）相比，这些网络在此情况下普遍脆弱。值得注意的是，随着我们限制图像曝光时间，人类的表现下降到深度网络的水平，这表明人类在识别不寻常姿势中的物体时需要额外的心理过程（需要额外的时间）。最后，我们分析了人类与网络的错误模式，发现即使在限制时间的情况下，人类与前馈深度网络也有不同。我们得出结论，需要更多的工作将计算机视觉系统带到人类视觉系统的鲁棒性水平。理解在外部情况下发生的心理过程的本质是必要的。

    Deep learning is closing the gap with humans on several object recognition benchmarks. Here we investigate this gap in the context of challenging images where objects are seen from unusual viewpoints. We find that humans excel at recognizing objects in unusual poses, in contrast with state-of-the-art pretrained networks (EfficientNet, SWAG, ViT, SWIN, BEiT, ConvNext) which are systematically brittle in this condition. Remarkably, as we limit image exposure time, human performance degrades to the level of deep networks, suggesting that additional mental processes (requiring additional time) take place when humans identify objects in unusual poses. Finally, our analysis of error patterns of humans vs. networks reveals that even time-limited humans are dissimilar to feed-forward deep networks. We conclude that more work is needed to bring computer vision systems to the level of robustness of the human visual system. Understanding the nature of the mental processes taking place during extr
    
[^4]: 研究DermaMNIST和Fitzpatrick17k皮肤科图像数据集的质量

    Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets. (arXiv:2401.14497v1 [cs.CV])

    [http://arxiv.org/abs/2401.14497](http://arxiv.org/abs/2401.14497)

    本文研究了DermaMNIST和Fitzpatrick17k皮肤科图像数据集的质量问题，对数据重复、数据泄漏、错误标记和缺乏测试分区等方面进行了详细分析，并提出纠正措施。

    

    深度学习在皮肤科任务中取得的显著进展使我们更接近于达到与人类专家相当的诊断准确性。然而，尽管大型数据集在可靠的深度神经网络模型的开发中起着关键作用，但数据集中的数据质量和其正确使用至关重要。多种因素可以影响数据质量，如重复数据的存在，训练-测试分区的数据泄漏，错误标记的图像以及缺乏明确定义的测试分区。在本文中，我们对两个流行的皮肤科图像数据集DermaMNIST和Fitzpatrick17k进行了详细分析，揭示了这些数据质量问题，测量了这些问题对基准结果的影响，并对数据集提出了纠正措施。通过公开我们的分析流程和配套代码，确保我们分析的可重复性，我们旨在鼓励类似的探索并促进这方面的研究发展。

    The remarkable progress of deep learning in dermatological tasks has brought us closer to achieving diagnostic accuracies comparable to those of human experts. However, while large datasets play a crucial role in the development of reliable deep neural network models, the quality of data therein and their correct usage are of paramount importance. Several factors can impact data quality, such as the presence of duplicates, data leakage across train-test partitions, mislabeled images, and the absence of a well-defined test partition. In this paper, we conduct meticulous analyses of two popular dermatological image datasets: DermaMNIST and Fitzpatrick17k, uncovering these data quality issues, measure the effects of these problems on the benchmark results, and propose corrections to the datasets. Besides ensuring the reproducibility of our analysis, by making our analysis pipeline and the accompanying code publicly available, we aim to encourage similar explorations and to facilitate the 
    
[^5]: 对水印技术应用于人工智能生成内容的漏洞研究

    Towards the Vulnerability of Watermarking Artificial Intelligence Generated Content. (arXiv:2310.07726v1 [cs.CV])

    [http://arxiv.org/abs/2310.07726](http://arxiv.org/abs/2310.07726)

    该研究探讨了将水印技术应用于人工智能生成内容的漏洞，并证明了现有的水印机制容易被对手破解。

    

    人工智能生成内容（AIGC）在社交媒体上越来越受欢迎，许多商业服务已经推出。这些服务利用先进的生成模型，如潜在扩散模型和大型语言模型，为用户生成创意内容（例如逼真的图像、流畅的句子）。对于此类生成内容的使用需要高度监管，因为服务提供商需要确保用户不违反使用政策（例如滥用商业化、生成和分发不安全的内容）。最近提出了许多水印技术，但是本文表明对手可以轻易破解这些水印机制。具体而言，我们考虑了两种可能的攻击方式：（1）水印去除：对手可以轻松地从生成内容中删除嵌入的水印，然后自由使用而不受服务提供商的限制；（2）水印伪造：对手可以创建非法的水印。

    Artificial Intelligence Generated Content (AIGC) is gaining great popularity in social media, with many commercial services available. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images, fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content).  Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely without the regulation of the service provider. (2) Watermark forge: the adversary can create illegal co
    

