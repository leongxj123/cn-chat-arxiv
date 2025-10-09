# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Train-Free Segmentation in MRI with Cubical Persistent Homology.](http://arxiv.org/abs/2401.01160) | 这是一种使用拓扑数据分析进行MRI图像分割的新方法，相比传统机器学习方法具有优势，无需大量注释数据集，提供更可解释和稳定的分割框架。 |

# 详细

[^1]: 无需训练的MRI立方持续同调分割方法

    Train-Free Segmentation in MRI with Cubical Persistent Homology. (arXiv:2401.01160v1 [eess.IV])

    [http://arxiv.org/abs/2401.01160](http://arxiv.org/abs/2401.01160)

    这是一种使用拓扑数据分析进行MRI图像分割的新方法，相比传统机器学习方法具有优势，无需大量注释数据集，提供更可解释和稳定的分割框架。

    

    我们描述了一种新的MRI扫描分割方法，使用拓扑数据分析（TDA），相比传统的机器学习方法具有几个优点。它分为三个步骤，首先通过自动阈值确定要分割的整个对象，然后检测一个已知拓扑结构的独特子集，最后推导出分割的各个组成部分。虽然调用了TDA的经典思想，但这样的算法从未与深度学习方法分离提出。为了实现这一点，我们的方法除了考虑图像的同调性外，还考虑了代表性周期的定位，这是在这种情况下似乎从未被利用过的信息。特别是，它提供了无需大量注释数据集进行分割的能力。TDA还通过将拓扑特征明确映射到分割组件来提供更可解释和稳定的分割框架。

    We describe a new general method for segmentation in MRI scans using Topological Data Analysis (TDA), offering several advantages over traditional machine learning approaches. It works in three steps, first identifying the whole object to segment via automatic thresholding, then detecting a distinctive subset whose topology is known in advance, and finally deducing the various components of the segmentation. Although convoking classical ideas of TDA, such an algorithm has never been proposed separately from deep learning methods. To achieve this, our approach takes into account, in addition to the homology of the image, the localization of representative cycles, a piece of information that seems never to have been exploited in this context. In particular, it offers the ability to perform segmentation without the need for large annotated data sets. TDA also provides a more interpretable and stable framework for segmentation by explicitly mapping topological features to segmentation comp
    

