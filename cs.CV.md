# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ParFormer: Vision Transformer Baseline with Parallel Local Global Token Mixer and Convolution Attention Patch Embedding](https://arxiv.org/abs/2403.15004) | ParFormer提出了并行局部全局标记混合器和卷积注意力补丁嵌入，优化了特征提取能力，在图像分类和对象识别等任务中表现优于CNN和最先进的Transformer架构。 |
| [^2] | [$\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples](https://arxiv.org/abs/2402.01879) | 该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。 |
| [^3] | [OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods](https://arxiv.org/abs/2312.08255) | 该研究介绍了一个名为OCTDL的开放获取光学相干断层扫描数据集，包括超过2000张标记有疾病组和视网膜病理的OCT图像，有助于诊断眼部状况。 |
| [^4] | [Visual Acuity Prediction on Real-Life Patient Data Using a Machine Learning Based Multistage System.](http://arxiv.org/abs/2204.11970) | 本研究提供了一种使用机器学习技术开发预测模型的多阶段系统，可高精度预测三种眼疾患者的视力变化，并辅助眼科医生进行临床决策和患者咨询。 |

# 详细

[^1]: ParFormer：具有并行局部全局标记混合器和卷积注意力补丁嵌入的视觉Transformer基线

    ParFormer: Vision Transformer Baseline with Parallel Local Global Token Mixer and Convolution Attention Patch Embedding

    [https://arxiv.org/abs/2403.15004](https://arxiv.org/abs/2403.15004)

    ParFormer提出了并行局部全局标记混合器和卷积注意力补丁嵌入，优化了特征提取能力，在图像分类和对象识别等任务中表现优于CNN和最先进的Transformer架构。

    

    本文提出了ParFormer作为一种增强型Transformer架构，允许将不同的标记混合器整合到单个阶段中，从而提高特征提取能力。同时整合本地和全局数据，实现对短程和长程空间关系的精确表示，而无需像平移窗口这样需要大量计算的方法。除了并行标记混合器编码器外，我们提供了卷积注意力补丁嵌入(CAPE)，作为标准补丁嵌入的增强，通过卷积注意力模块改进标记混合器提取。我们的全面评估表明，我们的ParFormer在图像分类和物体识别等多个复杂任务中优于基于CNN和最先进的基于Transformer的架构。所提出的CAPE已被证明有益于整体MetaFormer架构，即使使用Id。

    arXiv:2403.15004v1 Announce Type: cross  Abstract: This work presents ParFormer as an enhanced transformer architecture that allows the incorporation of different token mixers into a single stage, hence improving feature extraction capabilities. Integrating both local and global data allows for precise representation of short- and long-range spatial relationships without the need for computationally intensive methods such as shifting windows. Along with the parallel token mixer encoder, We offer the Convolutional Attention Patch Embedding (CAPE) as an enhancement of standard patch embedding to improve token mixer extraction with a convolutional attention module. Our comprehensive evaluation demonstrates that our ParFormer outperforms CNN-based and state-of-the-art transformer-based architectures in image classification and several complex tasks such as object recognition. The proposed CAPE has been demonstrated to benefit the overall MetaFormer architecture, even while utilizing the Id
    
[^2]: $\sigma$-zero: 基于梯度的$\ell_0$-范数对抗样本优化

    $\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples

    [https://arxiv.org/abs/2402.01879](https://arxiv.org/abs/2402.01879)

    该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。

    

    评估深度网络对基于梯度攻击的对抗鲁棒性是具有挑战性的。虽然大多数攻击考虑$\ell_2$和$\ell_\infty$范数约束来制造输入扰动，但只有少数研究了稀疏的$\ell_1$和$\ell_0$范数攻击。特别是，由于在非凸且非可微约束上进行优化的固有复杂性，$\ell_0$范数攻击是研究最少的。然而，使用这些攻击评估对抗鲁棒性可以揭示在更传统的$\ell_2$和$\ell_\infty$范数攻击中未能测试出的弱点。在这项工作中，我们提出了一种新颖的$\ell_0$范数攻击，称为$\sigma$-zero，它利用了$\ell_0$范数的一个特殊可微近似来促进基于梯度的优化，并利用自适应投影运算符动态调整损失最小化和扰动稀疏性之间的权衡。通过在MNIST、CIFAR10和ImageNet数据集上进行广泛评估，包括...

    Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages an ad hoc differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving
    
[^3]: OCTDL：基于图像的深度学习方法的光学相干断层扫描数据集

    OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods

    [https://arxiv.org/abs/2312.08255](https://arxiv.org/abs/2312.08255)

    该研究介绍了一个名为OCTDL的开放获取光学相干断层扫描数据集，包括超过2000张标记有疾病组和视网膜病理的OCT图像，有助于诊断眼部状况。

    

    光学相干断层扫描（OCT）是一种非侵入性成像技术，在眼科学中具有广泛的临床应用。OCT可以可视化视网膜层，对早期检测和监测视网膜疾病起着重要作用。本文介绍了一个开放获取的OCT数据集（OCTDL），包括超过2000张根据疾病组和视网膜病理标记的OCT图像。该数据集包括患有老年性黄斑变性（AMD）、糖尿病黄斑水肿（DME）、玻璃体视网膜膜（ERM）、视网膜动脉闭塞（RAO）、视网膜静脉闭塞（RVO）和玻璃体黄斑界面疾病（VID）的患者的OCT记录。这些图像是使用Optovue Avanti RTVue XR采集的，采用了动态扫描长度的光栅扫描协议。

    arXiv:2312.08255v2 Announce Type: replace-cross  Abstract: Optical coherence tomography (OCT) is a non-invasive imaging technique with extensive clinical applications in ophthalmology. OCT enables the visualization of the retinal layers, playing a vital role in the early detection and monitoring of retinal diseases. OCT uses the principle of light wave interference to create detailed images of the retinal microstructures, making it a valuable tool for diagnosing ocular conditions. This work presents an open-access OCT dataset (OCTDL) comprising over 2000 OCT images labeled according to disease group and retinal pathology. The dataset consists of OCT records of patients with Age-related Macular Degeneration (AMD), Diabetic Macular Edema (DME), Epiretinal Membrane (ERM), Retinal Artery Occlusion (RAO), Retinal Vein Occlusion (RVO), and Vitreomacular Interface Disease (VID). The images were acquired with an Optovue Avanti RTVue XR using raster scanning protocols with dynamic scan length a
    
[^4]: 基于机器学习的多阶段系统对真实患者数据进行视力预测

    Visual Acuity Prediction on Real-Life Patient Data Using a Machine Learning Based Multistage System. (arXiv:2204.11970v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2204.11970](http://arxiv.org/abs/2204.11970)

    本研究提供了一种使用机器学习技术开发预测模型的多阶段系统，可高精度预测三种眼疾患者的视力变化，并辅助眼科医生进行临床决策和患者咨询。

    

    现实生活中，眼科学中的玻璃体手术药物治疗是治疗年龄相关性黄斑变性（AMD）、糖尿病性黄斑水肿（DME）和视网膜静脉阻塞（RVO）相关疾病的一种普遍治疗方法。然而，在真实世界的情况下，由于数据的异质性和不完整性，患者往往会在多年时间内失去视力，尽管接受治疗。本文采用多种IT系统，提出了一种用于研究的数据集成流程，该流程融合了德国一家最佳医疗保健医院的眼科部门的不同IT系统。经过使用机器学习技术开发预测模型，我们实现了对患者视力的预测。我们的结果表明，我们的系统可以为三种疾病的预测提供高准确性。此外，我们还展示了我们的系统可以作为工具，辅助眼科医生进行临床决策和患者咨询。

    In ophthalmology, intravitreal operative medication therapy (IVOM) is a widespread treatment for diseases related to the age-related macular degeneration (AMD), the diabetic macular edema (DME), as well as the retinal vein occlusion (RVO). However, in real-world settings, patients often suffer from loss of vision on time scales of years despite therapy, whereas the prediction of the visual acuity (VA) and the earliest possible detection of deterioration under real-life conditions is challenging due to heterogeneous and incomplete data. In this contribution, we present a workflow for the development of a research-compatible data corpus fusing different IT systems of the department of ophthalmology of a German maximum care hospital. The extensive data corpus allows predictive statements of the expected progression of a patient and his or her VA in each of the three diseases. We found out for the disease AMD a significant deterioration of the visual acuity over time. Within our proposed m
    

