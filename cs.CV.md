# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Free Discontinuity Design: With an Application to the Economic Effects of Internet Shutdowns.](http://arxiv.org/abs/2309.14630) | 本文提出了一种非参数方法来估计互联网封锁对经济活动的影响，并发现印度的Internet封锁导致经济活动减少超过50％，对全球数字经济的封锁真实成本产生了新的见解。 |
| [^2] | [An automated end-to-end deep learning-based framework for lung cancer diagnosis by detecting and classifying the lung nodules.](http://arxiv.org/abs/2305.00046) | 本文提出了一种基于深度学习的智能诊断框架，针对低资源环境实现早期检测和分类肺部结节，并在公共数据集上取得了较好的表现。 |
| [^3] | [Label-Efficient Deep Learning in Medical Image Analysis: Challenges and Future Directions.](http://arxiv.org/abs/2303.12484) | 近年来深度学习在医学图像分析中取得了最先进的性能，但这种方法的标记代价大，标记不足。因此发展了高效标记深度学习方法，充分利用未标记的和弱标记的数据。该综述总结了这方面的最新进展。 |

# 详细

[^1]: 自由不连续设计：应用于互联网封锁的经济影响

    Free Discontinuity Design: With an Application to the Economic Effects of Internet Shutdowns. (arXiv:2309.14630v1 [econ.EM])

    [http://arxiv.org/abs/2309.14630](http://arxiv.org/abs/2309.14630)

    本文提出了一种非参数方法来估计互联网封锁对经济活动的影响，并发现印度的Internet封锁导致经济活动减少超过50％，对全球数字经济的封锁真实成本产生了新的见解。

    

    在治疗分配中的阈值可以产生结果的不连续性，从而揭示因果洞察力。在许多情境中，如地理环境，这些阈值是未知和多变量的。我们提出了一种非参数方法来估计由此产生的不连续性，通过将回归曲面分割成平滑和不连续部分。该估计器使用了Mumford-Shah函数的凸松弛，我们建立了其识别和收敛性。使用我们的方法，我们估计印度的Internet封锁导致经济活动减少超过50％，远远超过以前的估计，并对全球数字经济的此类封锁的真实成本产生了新的见解。

    Thresholds in treatment assignments can produce discontinuities in outcomes, revealing causal insights. In many contexts, like geographic settings, these thresholds are unknown and multivariate. We propose a non-parametric method to estimate the resulting discontinuities by segmenting the regression surface into smooth and discontinuous parts. This estimator uses a convex relaxation of the Mumford-Shah functional, for which we establish identification and convergence. Using our method, we estimate that an internet shutdown in India resulted in a reduction of economic activity by over 50%, greatly surpassing previous estimates and shedding new light on the true cost of such shutdowns for digital economies globally.
    
[^2]: 一种基于深度学习技术的肺癌诊断自动化端到端框架，用于检测和分类肺部结节

    An automated end-to-end deep learning-based framework for lung cancer diagnosis by detecting and classifying the lung nodules. (arXiv:2305.00046v1 [eess.IV])

    [http://arxiv.org/abs/2305.00046](http://arxiv.org/abs/2305.00046)

    本文提出了一种基于深度学习的智能诊断框架，针对低资源环境实现早期检测和分类肺部结节，并在公共数据集上取得了较好的表现。

    

    肺癌是全球癌症相关死亡的主要原因，在低资源环境中早期诊断对于改善患者疗效至关重要。本研究的目的是提出一种基于深度学习技术的自动化端到端框架，用于早期检测和分类肺部结节，特别是针对低资源环境。该框架由三个阶段组成：使用改进的3D Res-U-Net进行肺分割、使用YOLO-v5进行结节检测、使用基于Vision Transformer的架构进行分类。我们在开放的数据集LUNA16上对该框架进行了评估。所提出的框架的性能是使用各领域的评估指标进行衡量的。该框架在肺部分割dice系数上达到了98.82％，同时检测肺结节的平均准确度为0.76 mAP。

    Lung cancer is a leading cause of cancer-related deaths worldwide, and early detection is crucial for improving patient outcomes. Nevertheless, early diagnosis of cancer is a major challenge, particularly in low-resource settings where access to medical resources and trained radiologists is limited. The objective of this study is to propose an automated end-to-end deep learning-based framework for the early detection and classification of lung nodules, specifically for low-resource settings. The proposed framework consists of three stages: lung segmentation using a modified 3D U-Net named 3D Res-U-Net, nodule detection using YOLO-v5, and classification with a Vision Transformer-based architecture. We evaluated the proposed framework on a publicly available dataset, LUNA16. The proposed framework's performance was measured using the respective domain's evaluation matrices. The proposed framework achieved a 98.82% lung segmentation dice score while detecting the lung nodule with 0.76 mAP
    
[^3]: 医学图像分析中高效标记深度学习的挑战与未来方向

    Label-Efficient Deep Learning in Medical Image Analysis: Challenges and Future Directions. (arXiv:2303.12484v1 [cs.CV])

    [http://arxiv.org/abs/2303.12484](http://arxiv.org/abs/2303.12484)

    近年来深度学习在医学图像分析中取得了最先进的性能，但这种方法的标记代价大，标记不足。因此发展了高效标记深度学习方法，充分利用未标记的和弱标记的数据。该综述总结了这方面的最新进展。

    

    深度学习近年来得到了迅速发展，并在广泛应用中取得了最先进的性能。但是，训练模型通常需要收集大量标记数据，这需要昂贵耗时。特别是在医学图像分析（MIA）领域，数据有限，标签很难获得。因此，人们开发了高效标记深度学习方法，充分利用标记数据以及非标记和弱标记数据的丰富性。在本调查中，我们对近300篇论文进行了广泛调查，以全面概述最新进展的高效标记学习策略在MIA中的研究现状。我们首先介绍高效标记学习的背景，并将不同方案的方法归类。接下来，我们通过每种方案详细研究了目前最先进的方法。具体而言，我们进行了深入调查，覆盖了不仅是标准策略，还包括使用后处理和集合方法等方法。

    Deep learning has seen rapid growth in recent years and achieved state-of-the-art performance in a wide range of applications. However, training models typically requires expensive and time-consuming collection of large quantities of labeled data. This is particularly true within the scope of medical imaging analysis (MIA), where data are limited and labels are expensive to be acquired. Thus, label-efficient deep learning methods are developed to make comprehensive use of the labeled data as well as the abundance of unlabeled and weak-labeled data. In this survey, we extensively investigated over 300 recent papers to provide a comprehensive overview of recent progress on label-efficient learning strategies in MIA. We first present the background of label-efficient learning and categorize the approaches into different schemes. Next, we examine the current state-of-the-art methods in detail through each scheme. Specifically, we provide an in-depth investigation, covering not only canonic
    

