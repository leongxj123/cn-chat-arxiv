# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Interpretability of Vertebrae Fracture Grading using Human-interpretable Prototypes](https://arxiv.org/abs/2404.02830) | 使用ProtoVerse方法，我们提出了一种可解释的原型设计方法，可以可靠地解释深度学习模型对椎体骨折的分类决策，表现优于现有的基于原型的方法。 |
| [^2] | [Addressing Source Scale Bias via Image Warping for Domain Adaptation](https://arxiv.org/abs/2403.12712) | 通过在训练过程中对突出的对象区域进行过采样的自适应注意力处理，以及针对对象区域采样的实例级变形引导，有效减轻域自适应中的源尺度偏差。 |
| [^3] | [Noise Level Adaptive Diffusion Model for Robust Reconstruction of Accelerated MRI](https://arxiv.org/abs/2403.05245) | 提出了一种具有噪声水平自适应特性的后验采样策略，可用于解决MRI重建过程中因真实噪声水平变化导致的重建不准确问题。 |

# 详细

[^1]: 使用可解释的人类原型提升椎体骨折分级的解释性

    Enhancing Interpretability of Vertebrae Fracture Grading using Human-interpretable Prototypes

    [https://arxiv.org/abs/2404.02830](https://arxiv.org/abs/2404.02830)

    使用ProtoVerse方法，我们提出了一种可解释的原型设计方法，可以可靠地解释深度学习模型对椎体骨折的分类决策，表现优于现有的基于原型的方法。

    

    椎体骨折分级分类骨折严重程度，这在医学成像中是一项具有挑战性的任务，近年来吸引了深度学习（DL）模型。尽管DL辅助医学诊断等关键应用场景需要透明性和可信度，但只有少数工作尝试使这种模型具有人类可解释性。此外，这些模型要么依赖于事后方法，要么依赖于额外注释。在这项工作中，我们提出了一种新颖的可解释-by-design方法ProtoVerse，以在人类可理解的方式中找到相关的椎体骨折子部分（原型），可可靠地解释模型的决策。具体来说，我们引入了一种新颖的多样性促进损失，以减轻在具有复杂语义的小数据集中原型重复的问题。我们在VerSe'19数据集上进行了实验，并优于现有的基于原型的方法。此外，我们的模型在解释性方面表现更优秀。

    arXiv:2404.02830v1 Announce Type: cross  Abstract: Vertebral fracture grading classifies the severity of vertebral fractures, which is a challenging task in medical imaging and has recently attracted Deep Learning (DL) models. Only a few works attempted to make such models human-interpretable despite the need for transparency and trustworthiness in critical use cases like DL-assisted medical diagnosis. Moreover, such models either rely on post-hoc methods or additional annotations. In this work, we propose a novel interpretable-by-design method, ProtoVerse, to find relevant sub-parts of vertebral fractures (prototypes) that reliably explain the model's decision in a human-understandable way. Specifically, we introduce a novel diversity-promoting loss to mitigate prototype repetitions in small datasets with intricate semantics. We have experimented with the VerSe'19 dataset and outperformed the existing prototype-based method. Further, our model provides superior interpretability agains
    
[^2]: 通过图像变形解决域自适应中的源尺度偏差问题

    Addressing Source Scale Bias via Image Warping for Domain Adaptation

    [https://arxiv.org/abs/2403.12712](https://arxiv.org/abs/2403.12712)

    通过在训练过程中对突出的对象区域进行过采样的自适应注意力处理，以及针对对象区域采样的实例级变形引导，有效减轻域自适应中的源尺度偏差。

    

    在视觉识别中，由于真实场景数据集中对象和图像大小分布的不平衡，尺度偏差是一个关键挑战。传统解决方案包括注入尺度不变性先验、在训练过程中对数据集在不同尺度进行过采样，或者在推断时调整尺度。虽然这些策略在一定程度上减轻了尺度偏差，但它们在跨多样化数据集时的适应能力有限。此外，它们会增加训练过程的计算负载和推断过程的延迟。在这项工作中，我们使用自适应的注意力处理——通过在训练过程中就地扭曲图像来对突出的对象区域进行过采样。我们发现，通过改变源尺度分布可以改善主干特征，我们开发了一个面向对象区域采样的实例级变形引导，以减轻域自适应中的源尺度偏差。我们的方法提高了对地理、光照和天气条件的适应性。

    arXiv:2403.12712v1 Announce Type: cross  Abstract: In visual recognition, scale bias is a key challenge due to the imbalance of object and image size distribution inherent in real scene datasets. Conventional solutions involve injecting scale invariance priors, oversampling the dataset at different scales during training, or adjusting scale at inference. While these strategies mitigate scale bias to some extent, their ability to adapt across diverse datasets is limited. Besides, they increase computational load during training and latency during inference. In this work, we use adaptive attentional processing -- oversampling salient object regions by warping images in-place during training. Discovering that shifting the source scale distribution improves backbone features, we developed a instance-level warping guidance aimed at object region sampling to mitigate source scale bias in domain adaptation. Our approach improves adaptation across geographies, lighting and weather conditions, 
    
[^3]: 噪声水平自适应扩散模型用于加速MRI的稳健重建

    Noise Level Adaptive Diffusion Model for Robust Reconstruction of Accelerated MRI

    [https://arxiv.org/abs/2403.05245](https://arxiv.org/abs/2403.05245)

    提出了一种具有噪声水平自适应特性的后验采样策略，可用于解决MRI重建过程中因真实噪声水平变化导致的重建不准确问题。

    

    通常，基于扩散模型的MRI重建方法会逐步去除人为添加的噪声，并强调数据一致性以重建潜在图像。然而，现实世界中的MRI采集已经包含由热涨落引起的固有噪声。使用超快速、高分辨率成像序列进行高级研究，或者使用低场系统（受低收入和中等收入国家青睐）时，这种现象尤其明显。这些常见场景可能导致现有基于扩散模型的重建技术性能亚优或完全失败。具体而言，随着逐渐去除人为添加的噪声，固有的MRI噪声变得越来越明显，使实际噪声水平与预定义去噪时间表不一致，从而导致图像重建不准确。为解决这一问题，我们提出了一种具有新颖噪声水平自适应特性的后验采样策略。

    arXiv:2403.05245v1 Announce Type: cross  Abstract: In general, diffusion model-based MRI reconstruction methods incrementally remove artificially added noise while imposing data consistency to reconstruct the underlying images. However, real-world MRI acquisitions already contain inherent noise due to thermal fluctuations. This phenomenon is particularly notable when using ultra-fast, high-resolution imaging sequences for advanced research, or using low-field systems favored by low- and middle-income countries. These common scenarios can lead to sub-optimal performance or complete failure of existing diffusion model-based reconstruction techniques. Specifically, as the artificially added noise is gradually removed, the inherent MRI noise becomes increasingly pronounced, making the actual noise level inconsistent with the predefined denoising schedule and consequently inaccurate image reconstruction. To tackle this problem, we propose a posterior sampling strategy with a novel NoIse Lev
    

