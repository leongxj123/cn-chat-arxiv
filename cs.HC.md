# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Model's Interpretability and Reliability using Biomarkers](https://arxiv.org/abs/2402.12394) | 利用决策树解释基于生物标志物的诊断模型，帮助临床医生提高识别不准确预测的能力，从而增强医学诊断模型的可靠性。 |
| [^2] | [ClusterNet: A Perception-Based Clustering Model for Scattered Data.](http://arxiv.org/abs/2304.14185) | 这项工作介绍了ClusterNet，一种基于感知的分布式数据聚类模型，利用大规模数据集和基于点的深度学习模型，反映人类感知的聚类可分性。 |

# 详细

[^1]: 利用生物标志物提高模型的解释性和可靠性

    Improving Model's Interpretability and Reliability using Biomarkers

    [https://arxiv.org/abs/2402.12394](https://arxiv.org/abs/2402.12394)

    利用决策树解释基于生物标志物的诊断模型，帮助临床医生提高识别不准确预测的能力，从而增强医学诊断模型的可靠性。

    

    准确且具有解释性的诊断模型在医学这个安全关键领域至关重要。我们研究了我们提出的基于生物标志物的肺部超声诊断流程的可解释性，以增强临床医生的诊断能力。本研究的目标是评估决策树分类器利用生物标志物提供的解释是否能够改善用户识别模型不准确预测能力，与传统的显著性图相比。我们的研究发现表明，基于临床建立的生物标志物的决策树解释能够帮助临床医生检测到假阳性，从而提高医学诊断模型的可靠性。

    arXiv:2402.12394v1 Announce Type: cross  Abstract: Accurate and interpretable diagnostic models are crucial in the safety-critical field of medicine. We investigate the interpretability of our proposed biomarker-based lung ultrasound diagnostic pipeline to enhance clinicians' diagnostic capabilities. The objective of this study is to assess whether explanations from a decision tree classifier, utilizing biomarkers, can improve users' ability to identify inaccurate model predictions compared to conventional saliency maps. Our findings demonstrate that decision tree explanations, based on clinically established biomarkers, can assist clinicians in detecting false positives, thus improving the reliability of diagnostic models in medicine.
    
[^2]: ClusterNet：一种基于感知的分布式数据聚类模型

    ClusterNet: A Perception-Based Clustering Model for Scattered Data. (arXiv:2304.14185v1 [cs.LG])

    [http://arxiv.org/abs/2304.14185](http://arxiv.org/abs/2304.14185)

    这项工作介绍了ClusterNet，一种基于感知的分布式数据聚类模型，利用大规模数据集和基于点的深度学习模型，反映人类感知的聚类可分性。

    

    散点图中的聚类分离是一个通常由广泛使用的聚类技术（例如k-means或DBSCAN）来解决的任务。然而，由于这些算法基于非感知度量，它们的输出经常不能反映出人类聚类感知。为了弥合人类聚类感知和机器计算聚类之间的差距，我们提出了一种直接处理分布式数据的学习策略。为了在这些数据上学习感知聚类分离，我们进行了一项众包大规模数据集的工作，其中包括384个人群工作者对双变量数据的7,320个点聚类从属进行了标记。基于这些数据，我们能够训练ClusterNet，这是一个基于点的深度学习模型，被训练成反映人类感知的聚类可分性。为了在人类注释的数据上训练ClusterNet，我们省略了在2D画布上渲染散点图，而是使用了一个PointNet++架构，使其能够直接推理点云。在这项工作中，我们建立了一种基于感知的分布式数据聚类模型，ClusterNet。

    Cluster separation in scatterplots is a task that is typically tackled by widely used clustering techniques, such as for instance k-means or DBSCAN. However, as these algorithms are based on non-perceptual metrics, their output often does not reflect human cluster perception. To bridge the gap between human cluster perception and machine-computed clusters, we propose a learning strategy which directly operates on scattered data. To learn perceptual cluster separation on this data, we crowdsourced a large scale dataset, consisting of 7,320 point-wise cluster affiliations for bivariate data, which has been labeled by 384 human crowd workers. Based on this data, we were able to train ClusterNet, a point-based deep learning model, trained to reflect human perception of cluster separability. In order to train ClusterNet on human annotated data, we omit rendering scatterplots on a 2D canvas, but rather use a PointNet++ architecture enabling inference on point clouds directly. In this work, w
    

