# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand.](http://arxiv.org/abs/2310.20350) | 本文提出了一种结合形状完成和抓取预测的方法，实现了快速灵活的多指抓取。通过使用基于深度图像的形状完成模块和基于预测的抓取预测器，实现了在具有有限或无先验知识的情况下，对物体进行抓取的任务。 |
| [^2] | [Shape Completion with Prediction of Uncertain Regions.](http://arxiv.org/abs/2308.00377) | 该论文提出了两种方法来处理在给定模糊物体视图时可能存在的物体部分的不确定区域预测问题。研究表明这些方法可以作为任何预测空间占用的方法的直接扩展，通过后处理占用评分或直接预测不确定性指标来实现。这些方法与已知的概率形状完成方法进行了比较，并使用自动生成的深度图像数据集进行了验证。 |

# 详细

[^1]: 将形状完成和抓取预测结合，实现快速灵活的多指抓取

    Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand. (arXiv:2310.20350v1 [cs.RO])

    [http://arxiv.org/abs/2310.20350](http://arxiv.org/abs/2310.20350)

    本文提出了一种结合形状完成和抓取预测的方法，实现了快速灵活的多指抓取。通过使用基于深度图像的形状完成模块和基于预测的抓取预测器，实现了在具有有限或无先验知识的情况下，对物体进行抓取的任务。

    

    在辅助机器人中，对于具有有限或无先验知识的物体进行抓取是一项非常重要的技能。然而，在这种普适情况下，尤其是在观测能力有限和利用多指手进行灵活抓取时，仍然存在一个开放的问题。我们提出了一种新颖、快速和高保真度的深度学习流程，由基于单个深度图像的形状完成模块和基于预测的物体形状的抓取预测器组成。形状完成网络基于VQDIF，在任意查询点上预测空间占用值。作为抓取预测器，我们使用了两阶段架构，首先使用自回归模型生成手姿势，然后回归每个姿势的手指关节配置。关键因素是足够的数据真实性和增强，以及在训练过程中对困难情况的特殊关注。在物理机器人平台上进行的实验表明，成功地实现了抓取。

    Grasping objects with limited or no prior knowledge about them is a highly relevant skill in assistive robotics. Still, in this general setting, it has remained an open problem, especially when it comes to only partial observability and versatile grasping with multi-fingered hands. We present a novel, fast, and high fidelity deep learning pipeline consisting of a shape completion module that is based on a single depth image, and followed by a grasp predictor that is based on the predicted object shape. The shape completion network is based on VQDIF and predicts spatial occupancy values at arbitrary query points. As grasp predictor, we use our two-stage architecture that first generates hand poses using an autoregressive model and then regresses finger joint configurations per pose. Critical factors turn out to be sufficient data realism and augmentation, as well as special attention to difficult cases during training. Experiments on a physical robot platform demonstrate successful gras
    
[^2]: 带有不确定区域预测的形状完成

    Shape Completion with Prediction of Uncertain Regions. (arXiv:2308.00377v1 [cs.CV])

    [http://arxiv.org/abs/2308.00377](http://arxiv.org/abs/2308.00377)

    该论文提出了两种方法来处理在给定模糊物体视图时可能存在的物体部分的不确定区域预测问题。研究表明这些方法可以作为任何预测空间占用的方法的直接扩展，通过后处理占用评分或直接预测不确定性指标来实现。这些方法与已知的概率形状完成方法进行了比较，并使用自动生成的深度图像数据集进行了验证。

    

    形状完成，即从部分观测预测物体的完整几何形状，对于几个下游任务非常重要，尤其是机器人操作。当基于物体形状重建进行规划或实际抓取的预测时，指示严重几何不确定性是必不可少的。特别是在给定模糊的物体视图时，在整个物体部分存在 irreducible uncertainty 的扩展区域。为了处理这种重要情况，我们提出了两种新方法来预测这些不确定区域，这两种方法都可以作为预测局部空间占用的任何方法的直接扩展，一种是通过后处理占用评分，另一种是通过直接预测不确定性指标。我们将这些方法与两种已知的概率形状完成方法进行了比较。此外，我们还生成了一个基于ShapeNet的数据集，其中包含了真实渲染的物体视图深度图像及其带有地面真值标注。

    Shape completion, i.e., predicting the complete geometry of an object from a partial observation, is highly relevant for several downstream tasks, most notably robotic manipulation. When basing planning or prediction of real grasps on object shape reconstruction, an indication of severe geometric uncertainty is indispensable. In particular, there can be an irreducible uncertainty in extended regions about the presence of entire object parts when given ambiguous object views. To treat this important case, we propose two novel methods for predicting such uncertain regions as straightforward extensions of any method for predicting local spatial occupancy, one through postprocessing occupancy scores, the other through direct prediction of an uncertainty indicator. We compare these methods together with two known approaches to probabilistic shape completion. Moreover, we generate a dataset, derived from ShapeNet, of realistically rendered depth images of object views with ground-truth annot
    

