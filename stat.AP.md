# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Postprocessing of point predictions for probabilistic forecasting of electricity prices: Diversity matters](https://arxiv.org/abs/2404.02270) | 将点预测转换为概率预测的电力价格后处理方法中，结合Isotonic Distributional Regression与其他两种方法的预测分布可以实现显著的性能提升。 |
| [^2] | [An engine to simulate insurance fraud network data.](http://arxiv.org/abs/2308.11659) | 本论文介绍了一种模拟保险欺诈网络数据的引擎，利用索赔涉及方的社交网络特征进行学习方法，旨在开发高效准确的欺诈检测模型。但面临类别不平衡、大量未标记数据和缺乏公开数据等挑战。 |
| [^3] | [CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer.](http://arxiv.org/abs/2306.14590) | 本论文提出了一种CST-YOLO算法，基于改进的YOLOv7和CNN-Swin Transformer，引入了几个有用的模型，有效提高了血细胞检测精度，实验结果显示其在三个血细胞数据集上均优于现有最先进算法。 |

# 详细

[^1]: 电力价格概率预测的点预测后处理：多样性至关重要

    Postprocessing of point predictions for probabilistic forecasting of electricity prices: Diversity matters

    [https://arxiv.org/abs/2404.02270](https://arxiv.org/abs/2404.02270)

    将点预测转换为概率预测的电力价格后处理方法中，结合Isotonic Distributional Regression与其他两种方法的预测分布可以实现显著的性能提升。

    

    依赖于电力价格的预测分布进行操作决策相较于仅基于点预测的决策可以带来显著更高的利润。然而，在学术和工业环境中开发的大多数模型仅提供点预测。为了解决这一问题，我们研究了三种将点预测转换为概率预测的后处理方法：分位数回归平均、一致性预测和最近引入的等温分布回归。我们发现，虽然等温分布回归表现最为多样化，但将其预测分布与另外两种方法结合使用，相较于具有正态分布误差的基准模型，在德国电力市场的4.5年测试期间（涵盖COVID大流行和乌克兰战争），实现了约7.5%的改进。值得注意的是，这种组合的性能与最先进的Dis

    arXiv:2404.02270v1 Announce Type: new  Abstract: Operational decisions relying on predictive distributions of electricity prices can result in significantly higher profits compared to those based solely on point forecasts. However, the majority of models developed in both academic and industrial settings provide only point predictions. To address this, we examine three postprocessing methods for converting point forecasts into probabilistic ones: Quantile Regression Averaging, Conformal Prediction, and the recently introduced Isotonic Distributional Regression. We find that while IDR demonstrates the most varied performance, combining its predictive distributions with those of the other two methods results in an improvement of ca. 7.5% compared to a benchmark model with normally distributed errors, over a 4.5-year test period in the German power market spanning the COVID pandemic and the war in Ukraine. Remarkably, the performance of this combination is at par with state-of-the-art Dis
    
[^2]: 一种模拟保险欺诈网络数据的引擎

    An engine to simulate insurance fraud network data. (arXiv:2308.11659v1 [cs.LG])

    [http://arxiv.org/abs/2308.11659](http://arxiv.org/abs/2308.11659)

    本论文介绍了一种模拟保险欺诈网络数据的引擎，利用索赔涉及方的社交网络特征进行学习方法，旨在开发高效准确的欺诈检测模型。但面临类别不平衡、大量未标记数据和缺乏公开数据等挑战。

    

    传统上，检测保险欺诈索赔依赖于业务规则和专家判断，这使得这一过程耗时且昂贵。因此，研究人员一直在探索开发高效准确的分析策略来标记可疑索赔。从索赔涉及方的社交网络中提取特征并将其馈送给学习方法是一种特别有潜力的策略。然而，在开发欺诈检测模型时，我们面临着几个挑战。例如，欺诈的非常规性质导致了高度的类别不平衡，这增加了开发性能良好的分析分类模型的难度。此外，只有少数索赔得到调查和标签，从而产生了大量未标记的数据。另一个挑战是缺乏公开可用的数据，这妨碍了研究和模型验证。

    Traditionally, the detection of fraudulent insurance claims relies on business rules and expert judgement which makes it a time-consuming and expensive process (\'Oskarsd\'ottir et al., 2022). Consequently, researchers have been examining ways to develop efficient and accurate analytic strategies to flag suspicious claims. Feeding learning methods with features engineered from the social network of parties involved in a claim is a particularly promising strategy (see for example Van Vlasselaer et al. (2016); Tumminello et al. (2023)). When developing a fraud detection model, however, we are confronted with several challenges. The uncommon nature of fraud, for example, creates a high class imbalance which complicates the development of well performing analytic classification models. In addition, only a small number of claims are investigated and get a label, which results in a large corpus of unlabeled data. Yet another challenge is the lack of publicly available data. This hinders not 
    
[^3]: CST-YOLO: 一种基于改进的YOLOv7和CNN-Swin Transformer的血细胞检测新方法

    CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer. (arXiv:2306.14590v1 [cs.CV])

    [http://arxiv.org/abs/2306.14590](http://arxiv.org/abs/2306.14590)

    本论文提出了一种CST-YOLO算法，基于改进的YOLOv7和CNN-Swin Transformer，引入了几个有用的模型，有效提高了血细胞检测精度，实验结果显示其在三个血细胞数据集上均优于现有最先进算法。

    

    血细胞检测是计算机视觉中典型的小物体检测问题。本文提出了一种CST-YOLO模型，基于YOLOv7结构并使用CNN-Swin Transformer（CST）进行增强，这是一种CNN-Transformer融合的新尝试。同时，我们还引入了三个有用的模型：加权高效层聚合网络（W-ELAN）、多尺度通道分割（MCS）和级联卷积层（CatConv），以提高小物体检测精度。实验结果表明，我们提出的CST-YOLO在三个血细胞数据集上分别达到了92.7、95.6和91.1 mAP@0.5，优于最先进的物体检测器，如YOLOv5和YOLOv7。我们的代码可在https://github.com/mkang315/CST-YOLO上找到。

    Blood cell detection is a typical small-scale object detection problem in computer vision. In this paper, we propose a CST-YOLO model for blood cell detection based on YOLOv7 architecture and enhance it with the CNN-Swin Transformer (CST), which is a new attempt at CNN-Transformer fusion. We also introduce three other useful modules: Weighted Efficient Layer Aggregation Networks (W-ELAN), Multiscale Channel Split (MCS), and Concatenate Convolutional Layers (CatConv) in our CST-YOLO to improve small-scale object detection precision. Experimental results show that the proposed CST-YOLO achieves 92.7, 95.6, and 91.1 mAP@0.5 respectively on three blood cell datasets, outperforming state-of-the-art object detectors, e.g., YOLOv5 and YOLOv7. Our code is available at https://github.com/mkang315/CST-YOLO.
    

