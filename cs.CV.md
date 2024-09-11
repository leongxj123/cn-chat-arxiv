# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning](https://arxiv.org/abs/2404.00785) | 本研究利用图变分自动编码器和对比学习解开神经系统疾病中海马形状变异的关键潜变量，超越了其他先进方法在解开能力上的表现。 |
| [^2] | [Extending 6D Object Pose Estimators for Stereo Vision](https://arxiv.org/abs/2402.05610) | 这篇论文扩展了用于立体视觉的6D物体姿态估计器，通过使用立体视觉提供的额外视角和直接推测物体的距离，该方法在6D姿态估计方面优于最先进的算法，且可适用于其他基于密集特征的算法。 |
| [^3] | [A Unified Remote Sensing Anomaly Detector Across Modalities and Scenes via Deviation Relationship Learning.](http://arxiv.org/abs/2310.07511) | 通过学习偏差关系，构建了一个统一的遥感异常检测器，可以跨模态和场景进行检测，并且具有灵活性和成本效益。 |
| [^4] | [SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network.](http://arxiv.org/abs/2310.06488) | 本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。 |
| [^5] | [Feedback-guided Data Synthesis for Imbalanced Classification.](http://arxiv.org/abs/2310.00158) | 本论文介绍了一种反馈引导数据合成的方法，通过从分类器到生成模型的反馈来驱动采样，将静态数据集增强为包含有用的合成样本，以提高分类器的性能。 |

# 详细

[^1]: 解开海马形状变异之谜：利用对比学习的图变分自动编码器研究神经系统疾病

    Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning

    [https://arxiv.org/abs/2404.00785](https://arxiv.org/abs/2404.00785)

    本研究利用图变分自动编码器和对比学习解开神经系统疾病中海马形状变异的关键潜变量，超越了其他先进方法在解开能力上的表现。

    

    本文提出了一项综合研究，专注于在神经系统疾病背景下从扩散张量成像（DTI）数据集中解开海马形状变异。借助增强的监督对比学习图变分自动编码器（VAE），我们的方法旨在通过区分代表年龄和是否患病的两个不同潜变量来提高解释性。在我们的消融研究中，我们调查了一系列VAE架构和对比损失函数，展示了我们方法增强的解开能力。这个评估使用了来自DTI海马数据集的合成3D环形网格数据和真实的3D海马网格数据集。我们的监督解开模型在解开分数方面优于几种最先进的方法，如属性和引导VAE。我们的模型可以区分不同年龄组和疾病状况。

    arXiv:2404.00785v1 Announce Type: cross  Abstract: This paper presents a comprehensive study focused on disentangling hippocampal shape variations from diffusion tensor imaging (DTI) datasets within the context of neurological disorders. Leveraging a Graph Variational Autoencoder (VAE) enhanced with Supervised Contrastive Learning, our approach aims to improve interpretability by disentangling two distinct latent variables corresponding to age and the presence of diseases. In our ablation study, we investigate a range of VAE architectures and contrastive loss functions, showcasing the enhanced disentanglement capabilities of our approach. This evaluation uses synthetic 3D torus mesh data and real 3D hippocampal mesh datasets derived from the DTI hippocampal dataset. Our supervised disentanglement model outperforms several state-of-the-art (SOTA) methods like attribute and guided VAEs in terms of disentanglement scores. Our model distinguishes between age groups and disease status in pa
    
[^2]: 扩展用于立体视觉的6D物体姿态估计器

    Extending 6D Object Pose Estimators for Stereo Vision

    [https://arxiv.org/abs/2402.05610](https://arxiv.org/abs/2402.05610)

    这篇论文扩展了用于立体视觉的6D物体姿态估计器，通过使用立体视觉提供的额外视角和直接推测物体的距离，该方法在6D姿态估计方面优于最先进的算法，且可适用于其他基于密集特征的算法。

    

    准确、快速、稳健地估计物体的6D姿态仍然是一项困难的任务。然而，最近的直接从RGB图像中使用密集特征回归姿态的方法已经取得了最先进的结果。立体视觉提供了对物体的额外视角，可以帮助减少姿态歧义和遮挡。此外，立体图像可以直接推测物体的距离，而单目视觉则需要内置对象尺寸的知识。为了将最先进的6D物体姿态估计方法扩展到立体视觉，我们创建了一个与BOP兼容的YCB-V数据集的立体版本。我们的方法通过利用立体视觉优于最先进的6D姿态估计算法，并且可以轻松应用于其他基于密集特征的算法。

    Estimating the 6D pose of objects accurately, quickly, and robustly remains a difficult task. However, recent methods for directly regressing poses from RGB images using dense features have achieved state-of-the-art results. Stereo vision, which provides an additional perspective on the object, can help reduce pose ambiguity and occlusion. Moreover, stereo can directly infer the distance of an object, while mono-vision requires internalized knowledge of the object's size. To extend the state-of-the-art in 6D object pose estimation to stereo, we created a BOP compatible stereo version of the YCB-V dataset. Our method outperforms state-of-the-art 6D pose estimation algorithms by utilizing stereo vision and can easily be adopted for other dense feature-based algorithms.
    
[^3]: 通过偏差关系学习实现跨模态和场景的统一遥感异常检测

    A Unified Remote Sensing Anomaly Detector Across Modalities and Scenes via Deviation Relationship Learning. (arXiv:2310.07511v1 [cs.CV])

    [http://arxiv.org/abs/2310.07511](http://arxiv.org/abs/2310.07511)

    通过学习偏差关系，构建了一个统一的遥感异常检测器，可以跨模态和场景进行检测，并且具有灵活性和成本效益。

    

    遥感异常检测器可以找到与背景不符的目标作为潜在目标。鉴于地球异常类型的多样性，跨模态和场景的统一异常检测器应该具有成本效益，并且对于新的地球观测源和异常类型具有灵活性。然而，当前的异常检测器仅限于单一模态和单一场景，因为它们旨在学习不断变化的背景分布。在普遍的异常偏差模式的激发下，即异常展现出与其局部环境的偏差特点，我们利用这一特征构建了一个统一的异常检测器。首先，我们将异常检测任务重新定义为基于偏差关系的无向双层图，其中异常评分被建模为在背景和正常对象的模式给定下的条件概率。然后，学习目标被表示为一个条件概率排序问题。此外，我们设计了一种实例化方案。

    Remote sensing anomaly detector can find the objects deviating from the background as potential targets. Given the diversity in earth anomaly types, a unified anomaly detector across modalities and scenes should be cost-effective and flexible to new earth observation sources and anomaly types. However, the current anomaly detectors are limited to a single modality and single scene, since they aim to learn the varying background distribution. Motivated by the universal anomaly deviation pattern, in that anomalies exhibit deviations from their local context, we exploit this characteristic to build a unified anomaly detector. Firstly, we reformulate the anomaly detection task as an undirected bilayer graph based on the deviation relationship, where the anomaly score is modeled as the conditional probability, given the pattern of the background and normal objects. The learning objective is then expressed as a conditional probability ranking problem. Furthermore, we design an instantiation 
    
[^4]: SpikeCLIP：一种对比语言-图像预训练脉冲神经网络

    SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network. (arXiv:2310.06488v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2310.06488](http://arxiv.org/abs/2310.06488)

    本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。

    

    脉冲神经网络（SNNs）已经证明其在视觉和语言领域中能够实现与深度神经网络（DNNs）相当的性能，同时具有能效提高和符合生物合理性的优势。然而，将这种单模态的SNNs扩展到多模态的情景仍然是一个未开发的领域。受到对比语言-图像预训练（CLIP）概念的启发，我们引入了一个名为SpikeCLIP的新框架，通过“对齐预训练+双损失微调”的两步骤配方，来解决脉冲计算背景下两种模态之间的差距。广泛的实验证明，在常用的用于多模态模型评估的各种数据集上，SNNs取得了与其DNNs对应物相当的结果，同时显著降低了能源消耗。此外，SpikeCLIP在图像分类方面保持了稳定的性能。

    Spiking neural networks (SNNs) have demonstrated the capability to achieve comparable performance to deep neural networks (DNNs) in both visual and linguistic domains while offering the advantages of improved energy efficiency and adherence to biological plausibility. However, the extension of such single-modality SNNs into the realm of multimodal scenarios remains an unexplored territory. Drawing inspiration from the concept of contrastive language-image pre-training (CLIP), we introduce a novel framework, named SpikeCLIP, to address the gap between two modalities within the context of spike-based computing through a two-step recipe involving ``Alignment Pre-training + Dual-Loss Fine-tuning". Extensive experiments demonstrate that SNNs achieve comparable results to their DNN counterparts while significantly reducing energy consumption across a variety of datasets commonly used for multimodal model evaluation. Furthermore, SpikeCLIP maintains robust performance in image classification 
    
[^5]: 不平衡分类中的反馈引导数据合成

    Feedback-guided Data Synthesis for Imbalanced Classification. (arXiv:2310.00158v1 [cs.CV])

    [http://arxiv.org/abs/2310.00158](http://arxiv.org/abs/2310.00158)

    本论文介绍了一种反馈引导数据合成的方法，通过从分类器到生成模型的反馈来驱动采样，将静态数据集增强为包含有用的合成样本，以提高分类器的性能。

    

    当前机器学习中的现状是使用来自长尾分布的真实图像的静态数据集进行训练。最近生成模型的进展使研究人员开始用合成数据增强这些静态数据集，并在分类任务上报告了适度的性能改进。我们假设这些性能提升受到从分类器到生成模型的反馈不足的限制，这将促进生成样本的有用性以提高分类器的性能。在这项工作中，我们介绍了一种用有用的合成样本增强静态数据集的框架，该框架利用从分类器到生成模型的一次性反馈来驱动采样。为了使该框架有效，我们发现样本必须接近手头任务的真实数据支持，并且具有足够的多样性。我们在一个长尾数据集（ImageNe...上验证了三个反馈标准。

    Current status quo in machine learning is to use static datasets of real images for training, which often come from long-tailed distributions. With the recent advances in generative models, researchers have started augmenting these static datasets with synthetic data, reporting moderate performance improvements on classification tasks. We hypothesize that these performance gains are limited by the lack of feedback from the classifier to the generative model, which would promote the usefulness of the generated samples to improve the classifier's performance. In this work, we introduce a framework for augmenting static datasets with useful synthetic samples, which leverages one-shot feedback from the classifier to drive the sampling of the generative model. In order for the framework to be effective, we find that the samples must be close to the support of the real data of the task at hand, and be sufficiently diverse. We validate three feedback criteria on a long-tailed dataset (ImageNe
    

