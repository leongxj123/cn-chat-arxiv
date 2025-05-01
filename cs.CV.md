# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Redshift: Random Networks are not Random Functions](https://arxiv.org/abs/2403.02241) | 本论文研究了未经训练的随机权重网络，发现即使简单的MLPs也具有强烈的归纳偏见，不同于传统观点的是，NNs并不具有固有的“简单偏见”，而是依赖于组件的作用。 |
| [^2] | [EyePreserve: Identity-Preserving Iris Synthesis.](http://arxiv.org/abs/2312.12028) | 本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法，能够合成不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在保持身份的同时进行非线性纹理变形。 |
| [^3] | [LEyes: A Lightweight Framework for Deep Learning-Based Eye Tracking using Synthetic Eye Images.](http://arxiv.org/abs/2309.06129) | 本研究提出了一种名为LEyes的轻量级深度学习眼动跟踪框架，利用合成眼部图像进行训练，解决了由于训练数据集不足和眼部图像变异导致的模型泛化问题。实验结果表明，LEyes训练的模型在瞳孔和CR定位方面优于其他算法。 |
| [^4] | [Semantic Positive Pairs for Enhancing Contrastive Instance Discrimination.](http://arxiv.org/abs/2306.16122) | 提出了一种名为语义正向对集合（SPPS）的方法，可以在表示学习过程中识别具有相似语义内容的图像，并将它们视为正向实例，从而减少丢弃重要特征的风险。在多个实验数据集上的实验证明了该方法的有效性。 |

# 详细

[^1]: 神经红移：随机网络并非随机函数

    Neural Redshift: Random Networks are not Random Functions

    [https://arxiv.org/abs/2403.02241](https://arxiv.org/abs/2403.02241)

    本论文研究了未经训练的随机权重网络，发现即使简单的MLPs也具有强烈的归纳偏见，不同于传统观点的是，NNs并不具有固有的“简单偏见”，而是依赖于组件的作用。

    

    我们对神经网络（NNs）的泛化能力的理解仍不完整。目前的解释基于梯度下降（GD）的隐含偏见，但无法解释梯度自由方法中模型的能力，也无法解释最近观察到的未经训练网络的简单偏见。本文寻找NNs中的其他泛化源。为了独立于GD理解体系结构提供的归纳偏见，我们研究未经训练的随机权重网络。即使是简单的MLPs也表现出强烈的归纳偏见：在权重空间中进行均匀抽样会产生一个非常偏向于复杂性的函数分布。但与常规智慧不同，NNs并不具有固有的“简单偏见”。这一特性取决于组件，如ReLU、残差连接和层归一化。可利用替代体系结构构建偏向于任何复杂性水平的偏见。Transformers也具有这一特性。

    arXiv:2403.02241v1 Announce Type: cross  Abstract: Our understanding of the generalization capabilities of neural networks (NNs) is still incomplete. Prevailing explanations are based on implicit biases of gradient descent (GD) but they cannot account for the capabilities of models from gradient-free methods nor the simplicity bias recently observed in untrained networks. This paper seeks other sources of generalization in NNs.   Findings. To understand the inductive biases provided by architectures independently from GD, we examine untrained, random-weight networks. Even simple MLPs show strong inductive biases: uniform sampling in weight space yields a very biased distribution of functions in terms of complexity. But unlike common wisdom, NNs do not have an inherent "simplicity bias". This property depends on components such as ReLUs, residual connections, and layer normalizations. Alternative architectures can be built with a bias for any level of complexity. Transformers also inher
    
[^2]: EyePreserve: 保持身份的虹膜合成

    EyePreserve: Identity-Preserving Iris Synthesis. (arXiv:2312.12028v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.12028](http://arxiv.org/abs/2312.12028)

    本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法，能够合成不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在保持身份的同时进行非线性纹理变形。

    

    在广泛的瞳孔尺寸范围内保持身份的同身份生物特征虹膜图像的合成是复杂的，因为它涉及到虹膜肌肉收缩机制，需要将虹膜非线性纹理变形模型嵌入到合成流程中。本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法。这种方法能够合成具有不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在给定目标虹膜图像的分割掩膜下非线性地变形现有主体的虹膜图像纹理。虹膜识别实验表明，所提出的变形模型不仅在改变瞳孔尺寸时保持身份，而且在瞳孔尺寸有显著差异的同身份虹膜样本之间提供更好的相似度，与最先进的线性方法相比。

    Synthesis of same-identity biometric iris images, both for existing and non-existing identities while preserving the identity across a wide range of pupil sizes, is complex due to intricate iris muscle constriction mechanism, requiring a precise model of iris non-linear texture deformations to be embedded into the synthesis pipeline. This paper presents the first method of fully data-driven, identity-preserving, pupil size-varying s ynthesis of iris images. This approach is capable of synthesizing images of irises with different pupil sizes representing non-existing identities as well as non-linearly deforming the texture of iris images of existing subjects given the segmentation mask of the target iris image. Iris recognition experiments suggest that the proposed deformation model not only preserves the identity when changing the pupil size but offers better similarity between same-identity iris samples with significant differences in pupil size, compared to state-of-the-art linear an
    
[^3]: LEyes：一种轻量级深度学习眼动跟踪框架，使用合成眼部图像

    LEyes: A Lightweight Framework for Deep Learning-Based Eye Tracking using Synthetic Eye Images. (arXiv:2309.06129v1 [cs.CV])

    [http://arxiv.org/abs/2309.06129](http://arxiv.org/abs/2309.06129)

    本研究提出了一种名为LEyes的轻量级深度学习眼动跟踪框架，利用合成眼部图像进行训练，解决了由于训练数据集不足和眼部图像变异导致的模型泛化问题。实验结果表明，LEyes训练的模型在瞳孔和CR定位方面优于其他算法。

    

    深度学习已经加强了凝视估计技术，但实际部署受到不足的训练数据集的限制。眼部图像的硬件引起的变异以及记录的参与者之间固有的生物差异会导致特征和像素级别的差异，阻碍了在特定数据集上训练的模型的泛化能力。虚拟数据集可以是一个解决方案，但创建虚拟数据集既需要时间又需要资源。为了解决这个问题，我们提出了一个名为Light Eyes or "LEyes"的框架，与传统的逼真方法不同，LEyes仅模拟视频眼动跟踪所需的关键图像特征。LEyes便于在多样化的凝视估计任务上训练神经网络。我们证明，使用LEyes训练的模型在眼睛瞳孔和CR定位方面优于其他最先进的算法。

    Deep learning has bolstered gaze estimation techniques, but real-world deployment has been impeded by inadequate training datasets. This problem is exacerbated by both hardware-induced variations in eye images and inherent biological differences across the recorded participants, leading to both feature and pixel-level variance that hinders the generalizability of models trained on specific datasets. While synthetic datasets can be a solution, their creation is both time and resource-intensive. To address this problem, we present a framework called Light Eyes or "LEyes" which, unlike conventional photorealistic methods, only models key image features required for video-based eye tracking using simple light distributions. LEyes facilitates easy configuration for training neural networks across diverse gaze-estimation tasks. We demonstrate that models trained using LEyes outperform other state-of-the-art algorithms in terms of pupil and CR localization across well-known datasets. In addit
    
[^4]: 增强对比实例区分的语义正向对

    Semantic Positive Pairs for Enhancing Contrastive Instance Discrimination. (arXiv:2306.16122v1 [cs.CV])

    [http://arxiv.org/abs/2306.16122](http://arxiv.org/abs/2306.16122)

    提出了一种名为语义正向对集合（SPPS）的方法，可以在表示学习过程中识别具有相似语义内容的图像，并将它们视为正向实例，从而减少丢弃重要特征的风险。在多个实验数据集上的实验证明了该方法的有效性。

    

    基于实例区分的自监督学习算法有效地防止表示坍缩，并在表示学习中产生了有希望的结果。然而，在嵌入空间中吸引正向对（即相同实例的两个视图）并排斥所有其他实例（即负向对），无论它们的类别，可能导致丢弃重要的特征。为了解决这个问题，我们提出了一种方法来识别具有相似语义内容的图像，并将它们视为正向实例，命名为语义正向对集合（SPPS），从而在表示学习中减少了丢弃重要特征的风险。我们的方法可以与任何对比实例区分框架（如SimCLR或MOCO）一起使用。我们在三个数据集（ImageNet、STL-10和CIFAR-10）上进行实验证明了我们的方法的有效性。实验结果表明，我们的方法始终优于基线方法vanilla。

    Self-supervised learning algorithms based on instance discrimination effectively prevent representation collapse and produce promising results in representation learning. However, the process of attracting positive pairs (i.e., two views of the same instance) in the embedding space and repelling all other instances (i.e., negative pairs) irrespective of their categories could result in discarding important features. To address this issue, we propose an approach to identifying those images with similar semantic content and treating them as positive instances, named semantic positive pairs set (SPPS), thereby reducing the risk of discarding important features during representation learning. Our approach could work with any contrastive instance discrimination framework such as SimCLR or MOCO. We conduct experiments on three datasets: ImageNet, STL-10 and CIFAR-10 to evaluate our approach. The experimental results show that our approach consistently outperforms the baseline method vanilla 
    

