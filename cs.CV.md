# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fully Differentiable Lagrangian Convolutional Neural Network for Continuity-Consistent Physics-Informed Precipitation Nowcasting](https://arxiv.org/abs/2402.10747) | 提出了一种完全可微的拉格朗日卷积神经网络模型，实现了物理信息与数据驱动学习相结合，在降水预报中表现优秀，为其他拉格朗日机器学习模型提供了新思路。 |
| [^2] | [Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift](https://arxiv.org/abs/2402.10665) | 本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性 |
| [^3] | [Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT.](http://arxiv.org/abs/2401.03302) | 本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。 |
| [^4] | [Towards Enhanced Controllability of Diffusion Models.](http://arxiv.org/abs/2302.14368) | 本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。 |

# 详细

[^1]: 完全可微的拉格朗日卷积神经网络用于连续一致物理信息降水预报

    Fully Differentiable Lagrangian Convolutional Neural Network for Continuity-Consistent Physics-Informed Precipitation Nowcasting

    [https://arxiv.org/abs/2402.10747](https://arxiv.org/abs/2402.10747)

    提出了一种完全可微的拉格朗日卷积神经网络模型，实现了物理信息与数据驱动学习相结合，在降水预报中表现优秀，为其他拉格朗日机器学习模型提供了新思路。

    

    本文提出了一种卷积神经网络模型，用于降水预报，结合了数据驱动学习和基于物理信息的领域知识。我们提出了LUPIN，即用于物理信息的拉格朗日双U-Net的现在预报，借鉴了现有的基于外推的预报方法，并以完全可微且GPU加速的方式实现了数据的拉格朗日坐标系转换，以允许实时端到端训练和推断。根据我们的评估，LUPIN与并超过了所选择基准的性能，为其他拉格朗日机器学习模型敞开了大门。

    arXiv:2402.10747v1 Announce Type: cross  Abstract: This paper presents a convolutional neural network model for precipitation nowcasting that combines data-driven learning with physics-informed domain knowledge. We propose LUPIN, a Lagrangian Double U-Net for Physics-Informed Nowcasting, that draws from existing extrapolation-based nowcasting methods and implements the Lagrangian coordinate system transformation of the data in a fully differentiable and GPU-accelerated manner to allow for real-time end-to-end training and inference. Based on our evaluation, LUPIN matches and exceeds the performance of the chosen benchmark, opening the door for other Lagrangian machine learning models.
    
[^2]: 使用事后置信度估计的选择性预测在语义分割中的性能及其在分布偏移下的表现

    Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift

    [https://arxiv.org/abs/2402.10665](https://arxiv.org/abs/2402.10665)

    本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性

    

    语义分割在各种计算机视觉应用中扮演着重要角色，然而其有效性常常受到高质量标记数据的缺乏所限。为了解决这一挑战，一个常见策略是利用在不同种群上训练的模型，如公开可用的数据集。然而，这种方法导致了分布偏移问题，在兴趣种群上表现出降低的性能。在模型错误可能带来重大后果的情况下，选择性预测方法提供了一种减轻风险、减少对专家监督依赖的手段。本文研究了在资源匮乏环境下语义分割的选择性预测，着重于应用于在分布偏移下运行的预训练模型的事后置信度估计器。我们提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性。

    arXiv:2402.10665v1 Announce Type: new  Abstract: Semantic segmentation plays a crucial role in various computer vision applications, yet its efficacy is often hindered by the lack of high-quality labeled data. To address this challenge, a common strategy is to leverage models trained on data from different populations, such as publicly available datasets. This approach, however, leads to the distribution shift problem, presenting a reduced performance on the population of interest. In scenarios where model errors can have significant consequences, selective prediction methods offer a means to mitigate risks and reduce reliance on expert supervision. This paper investigates selective prediction for semantic segmentation in low-resource settings, thus focusing on post-hoc confidence estimators applied to pre-trained models operating under distribution shift. We propose a novel image-level confidence measure tailored for semantic segmentation and demonstrate its effectiveness through expe
    
[^3]: 行动中的现实主义：使用YOLOv8和DeiT从医学图像中诊断脑肿瘤的异常感知

    Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT. (arXiv:2401.03302v1 [eess.IV])

    [http://arxiv.org/abs/2401.03302](http://arxiv.org/abs/2401.03302)

    本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。

    

    在医学科学领域，由于脑肿瘤在患者中的罕见程度，可靠地检测和分类脑肿瘤仍然是一个艰巨的挑战。因此，在异常情况下检测肿瘤的能力对于确保及时干预和改善患者结果至关重要。本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤。来自国家脑映射实验室（NBML）的精选数据集包括81名患者，其中包括30例肿瘤病例和51例正常病例。检测和分类流程被分为两个连续的任务。检测阶段包括全面的数据分析和预处理，以修改图像样本和每个类别的患者数量，以符合真实世界场景中的异常分布（9个正常样本对应1个肿瘤样本）。此外，在测试中除了常见的评估指标外，我们还采用了... [摘要长度已达到上限]

    In the field of medical sciences, reliable detection and classification of brain tumors from images remains a formidable challenge due to the rarity of tumors within the population of patients. Therefore, the ability to detect tumors in anomaly scenarios is paramount for ensuring timely interventions and improved patient outcomes. This study addresses the issue by leveraging deep learning (DL) techniques to detect and classify brain tumors in challenging situations. The curated data set from the National Brain Mapping Lab (NBML) comprises 81 patients, including 30 Tumor cases and 51 Normal cases. The detection and classification pipelines are separated into two consecutive tasks. The detection phase involved comprehensive data analysis and pre-processing to modify the number of image samples and the number of patients of each class to anomaly distribution (9 Normal per 1 Tumor) to comply with real world scenarios. Next, in addition to common evaluation metrics for the testing, we emplo
    
[^4]: 实现扩展扩展扩散模型的可控性

    Towards Enhanced Controllability of Diffusion Models. (arXiv:2302.14368v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.14368](http://arxiv.org/abs/2302.14368)

    本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。

    

    去噪扩散模型在生成逼真、高质量和多样化图像方面表现出卓越能力。然而，在生成过程中的可控程度尚未得到充分探讨。受基于GAN潜在空间的图像操纵技术启发，我们训练了一个条件于两个潜在编码、一个空间内容掩码和一个扁平的样式嵌入的扩散模型。我们依赖于扩散模型渐进去噪过程的感性偏置，在空间结构掩码中编码姿势/布局信息，在样式代码中编码语义/样式信息。我们提出了两种通用的采样技术来改善可控性。我们扩展了可组合的扩散模型，允许部分依赖于条件输入，以提高生成质量，同时还提供对每个潜在代码和它们的联合分布量的控制。我们还提出了时间步相关的内容和样式潜在权重调度，进一步提高了控制性。

    Denoising Diffusion models have shown remarkable capabilities in generating realistic, high-quality and diverse images. However, the extent of controllability during generation is underexplored. Inspired by techniques based on GAN latent space for image manipulation, we train a diffusion model conditioned on two latent codes, a spatial content mask and a flattened style embedding. We rely on the inductive bias of the progressive denoising process of diffusion models to encode pose/layout information in the spatial structure mask and semantic/style information in the style code. We propose two generic sampling techniques for improving controllability. We extend composable diffusion models to allow for some dependence between conditional inputs, to improve the quality of generations while also providing control over the amount of guidance from each latent code and their joint distribution. We also propose timestep dependent weight scheduling for content and style latents to further impro
    

