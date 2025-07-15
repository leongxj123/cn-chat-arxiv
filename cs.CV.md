# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation](https://arxiv.org/abs/2403.06759) | 提出一种平均L1校准误差（mL1-ACE）作为辅助损失函数，用于改善图像分割中的像素级校准，减少了校准误差并引入了数据集可靠性直方图以提高校准评估。 |
| [^2] | [Understanding Pan-Sharpening via Generalized Inverse.](http://arxiv.org/abs/2310.02718) | 通过研究广义逆理论，本文提出了一种新的全色增强算法，该算法基于简单矩阵方程描述全色增强问题，并探讨解的条件和光谱、空间分辨率的获取。通过引入降采样增强方法，我们得到了与分量替代和多分辨率分析方法相对应的广义逆矩阵表达式，并提出了一个新的模型先验来解决全色增强中的理论误差问题。 |

# 详细

[^1]: 平均校准误差：一种可微损失函数，用于改善图像分割中的可靠性

    Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation

    [https://arxiv.org/abs/2403.06759](https://arxiv.org/abs/2403.06759)

    提出一种平均L1校准误差（mL1-ACE）作为辅助损失函数，用于改善图像分割中的像素级校准，减少了校准误差并引入了数据集可靠性直方图以提高校准评估。

    

    医学图像分割的深度神经网络经常产生与经验观察不一致的过于自信的结果，这种校准错误挑战着它们的临床应用。我们提出使用平均L1校准误差（mL1-ACE）作为一种新颖的辅助损失函数，以改善像素级校准而不会损害分割质量。我们展示了，尽管使用硬分箱，这种损失是直接可微的，避免了需要近似但可微的替代或软分箱方法的必要性。我们的工作还引入了数据集可靠性直方图的概念，这一概念推广了标准的可靠性图，用于在数据集级别聚合的语义分割中细化校准的视觉评估。使用mL1-ACE，我们将平均和最大校准误差分别降低了45%和55%，同时在BraTS 2021数据集上保持了87%的Dice分数。我们在这里分享我们的代码: https://github

    arXiv:2403.06759v1 Announce Type: cross  Abstract: Deep neural networks for medical image segmentation often produce overconfident results misaligned with empirical observations. Such miscalibration, challenges their clinical translation. We propose to use marginal L1 average calibration error (mL1-ACE) as a novel auxiliary loss function to improve pixel-wise calibration without compromising segmentation quality. We show that this loss, despite using hard binning, is directly differentiable, bypassing the need for approximate but differentiable surrogate or soft binning approaches. Our work also introduces the concept of dataset reliability histograms which generalises standard reliability diagrams for refined visual assessment of calibration in semantic segmentation aggregated at the dataset level. Using mL1-ACE, we reduce average and maximum calibration error by 45% and 55% respectively, maintaining a Dice score of 87% on the BraTS 2021 dataset. We share our code here: https://github
    
[^2]: 通过广义逆理解全色增强算法

    Understanding Pan-Sharpening via Generalized Inverse. (arXiv:2310.02718v1 [cs.LG])

    [http://arxiv.org/abs/2310.02718](http://arxiv.org/abs/2310.02718)

    通过研究广义逆理论，本文提出了一种新的全色增强算法，该算法基于简单矩阵方程描述全色增强问题，并探讨解的条件和光谱、空间分辨率的获取。通过引入降采样增强方法，我们得到了与分量替代和多分辨率分析方法相对应的广义逆矩阵表达式，并提出了一个新的模型先验来解决全色增强中的理论误差问题。

    

    全色增强算法利用全色图像和多光谱图像获取具有高空间和高光谱的图像。然而，这些算法的优化是根据不同的标准设计的。我们采用简单的矩阵方程来描述全色增强问题，并讨论解的存在条件以及光谱和空间分辨率的获取。我们引入了一种降采样增强方法，以更好地获取空间和光谱降采样矩阵。通过广义逆理论，我们推导出了两种形式的广义逆矩阵表达式，可以对应于两个主要的全色增强方法：分量替代和多分辨率分析方法。具体而言，我们证明了Gram Schmidt自适应(GSA)方法遵循分量替代的广义逆矩阵表达式。我们提出了一个在光谱函数的广义逆矩阵之前的模型先验。我们对理论误差进行了分析。

    Pan-sharpening algorithm utilizes panchromatic image and multispectral image to obtain a high spatial and high spectral image. However, the optimizations of the algorithms are designed with different standards. We adopt the simple matrix equation to describe the Pan-sharpening problem. The solution existence condition and the acquirement of spectral and spatial resolution are discussed. A down-sampling enhancement method was introduced for better acquiring the spatial and spectral down-sample matrices. By the generalized inverse theory, we derived two forms of general inverse matrix formulations that can correspond to the two prominent classes of Pan-sharpening methods, that is, component substitution and multi-resolution analysis methods. Specifically, the Gram Schmidt Adaptive(GSA) was proved to follow the general inverse matrix formulation of component substitution. A model prior to the general inverse matrix of the spectral function was rendered. The theoretical errors are analyzed
    

