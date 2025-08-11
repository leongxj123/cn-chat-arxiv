# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift](https://arxiv.org/abs/2402.10665) | 本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性 |
| [^2] | [Improved DDIM Sampling with Moment Matching Gaussian Mixtures.](http://arxiv.org/abs/2311.04938) | 在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。 |
| [^3] | [MESAHA-Net: Multi-Encoders based Self-Adaptive Hard Attention Network with Maximum Intensity Projections for Lung Nodule Segmentation in CT Scan.](http://arxiv.org/abs/2304.01576) | 本文提出了一种名为MESAHA-Net的高效端到端框架，集成了三种类型的输入，通过采用自适应硬注意力机制，逐层2D分割，实现了 CT扫描中精确的肺结节分割。 |

# 详细

[^1]: 使用事后置信度估计的选择性预测在语义分割中的性能及其在分布偏移下的表现

    Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift

    [https://arxiv.org/abs/2402.10665](https://arxiv.org/abs/2402.10665)

    本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性

    

    语义分割在各种计算机视觉应用中扮演着重要角色，然而其有效性常常受到高质量标记数据的缺乏所限。为了解决这一挑战，一个常见策略是利用在不同种群上训练的模型，如公开可用的数据集。然而，这种方法导致了分布偏移问题，在兴趣种群上表现出降低的性能。在模型错误可能带来重大后果的情况下，选择性预测方法提供了一种减轻风险、减少对专家监督依赖的手段。本文研究了在资源匮乏环境下语义分割的选择性预测，着重于应用于在分布偏移下运行的预训练模型的事后置信度估计器。我们提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性。

    arXiv:2402.10665v1 Announce Type: new  Abstract: Semantic segmentation plays a crucial role in various computer vision applications, yet its efficacy is often hindered by the lack of high-quality labeled data. To address this challenge, a common strategy is to leverage models trained on data from different populations, such as publicly available datasets. This approach, however, leads to the distribution shift problem, presenting a reduced performance on the population of interest. In scenarios where model errors can have significant consequences, selective prediction methods offer a means to mitigate risks and reduce reliance on expert supervision. This paper investigates selective prediction for semantic segmentation in low-resource settings, thus focusing on post-hoc confidence estimators applied to pre-trained models operating under distribution shift. We propose a novel image-level confidence measure tailored for semantic segmentation and demonstrate its effectiveness through expe
    
[^2]: 使用矩匹配高斯混合模型改进了DDIM采样

    Improved DDIM Sampling with Moment Matching Gaussian Mixtures. (arXiv:2311.04938v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2311.04938](http://arxiv.org/abs/2311.04938)

    在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。

    

    我们提出在Denoising Diffusion Implicit Models (DDIM)框架中使用高斯混合模型（GMM）作为反向转移算子（内核），这是一种从预训练的Denoising Diffusion Probabilistic Models (DDPM)中加速采样的广泛应用方法之一。具体而言，我们通过约束GMM的参数，匹配DDPM前向边际的一阶和二阶中心矩。我们发现，通过矩匹配，可以获得与使用高斯核的原始DDIM相同或更好质量的样本。我们在CelebAHQ和FFHQ的无条件模型以及ImageNet数据集的类条件模型上提供了实验结果。我们的结果表明，在采样步骤较少的情况下，使用GMM内核可以显著改善生成样本的质量，这是通过FID和IS指标衡量的。例如，在ImageNet 256x256上，使用10个采样步骤，我们实现了一个FID值为...

    We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ and class-conditional models trained on ImageNet datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of
    
[^3]: 基于多编码器的最大强度投影自适应硬注意力网络的CT扫描肺结节分割 MESAHA-Net（arXiv：2304.01576v1 [eess.IV]）

    MESAHA-Net: Multi-Encoders based Self-Adaptive Hard Attention Network with Maximum Intensity Projections for Lung Nodule Segmentation in CT Scan. (arXiv:2304.01576v1 [eess.IV])

    [http://arxiv.org/abs/2304.01576](http://arxiv.org/abs/2304.01576)

    本文提出了一种名为MESAHA-Net的高效端到端框架，集成了三种类型的输入，通过采用自适应硬注意力机制，逐层2D分割，实现了 CT扫描中精确的肺结节分割。

    

    准确的肺结节分割对早期肺癌诊断非常重要，因为它可以大大提高患者的生存率。计算机断层扫描（CT）图像被广泛用于肺结节分析的早期诊断。然而，肺结节的异质性，大小多样性以及周围环境的复杂性对开发鲁棒的结节分割方法提出了挑战。在本研究中，我们提出了一个高效的端到端框架，即基于多编码器的自适应硬注意力网络（MESAHA-Net），用于CT扫描中精确的肺结节分割。MESAHA-Net包括三个编码路径，一个注意力块和一个解码器块，有助于集成三种类型的输入：CT切片补丁，前向和后向的最大强度投影（MIP）图像以及包含结节的感兴趣区域（ROI）掩码。通过采用新颖的自适应硬注意力机制，MESAHA-Net逐层执行逐层2D分割。

    Accurate lung nodule segmentation is crucial for early-stage lung cancer diagnosis, as it can substantially enhance patient survival rates. Computed tomography (CT) images are widely employed for early diagnosis in lung nodule analysis. However, the heterogeneity of lung nodules, size diversity, and the complexity of the surrounding environment pose challenges for developing robust nodule segmentation methods. In this study, we propose an efficient end-to-end framework, the multi-encoder-based self-adaptive hard attention network (MESAHA-Net), for precise lung nodule segmentation in CT scans. MESAHA-Net comprises three encoding paths, an attention block, and a decoder block, facilitating the integration of three types of inputs: CT slice patches, forward and backward maximum intensity projection (MIP) images, and region of interest (ROI) masks encompassing the nodule. By employing a novel adaptive hard attention mechanism, MESAHA-Net iteratively performs slice-by-slice 2D segmentation 
    

