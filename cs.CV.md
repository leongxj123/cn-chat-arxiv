# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Train-Free Segmentation in MRI with Cubical Persistent Homology.](http://arxiv.org/abs/2401.01160) | 这是一种使用拓扑数据分析进行MRI图像分割的新方法，相比传统机器学习方法具有优势，无需大量注释数据集，提供更可解释和稳定的分割框架。 |
| [^2] | [You Can Ground Earlier than See: An Effective and Efficient Pipeline for Temporal Sentence Grounding in Compressed Videos.](http://arxiv.org/abs/2303.07863) | 本文提出了一种新的压缩域TSG设置，通过直接编码压缩位流来增强视觉特征表示能力和提高时间句子对齐的效率，并且在两个基准数据集的实验中表现优于目前最先进的方法。 |

# 详细

[^1]: 无需训练的MRI立方持续同调分割方法

    Train-Free Segmentation in MRI with Cubical Persistent Homology. (arXiv:2401.01160v1 [eess.IV])

    [http://arxiv.org/abs/2401.01160](http://arxiv.org/abs/2401.01160)

    这是一种使用拓扑数据分析进行MRI图像分割的新方法，相比传统机器学习方法具有优势，无需大量注释数据集，提供更可解释和稳定的分割框架。

    

    我们描述了一种新的MRI扫描分割方法，使用拓扑数据分析（TDA），相比传统的机器学习方法具有几个优点。它分为三个步骤，首先通过自动阈值确定要分割的整个对象，然后检测一个已知拓扑结构的独特子集，最后推导出分割的各个组成部分。虽然调用了TDA的经典思想，但这样的算法从未与深度学习方法分离提出。为了实现这一点，我们的方法除了考虑图像的同调性外，还考虑了代表性周期的定位，这是在这种情况下似乎从未被利用过的信息。特别是，它提供了无需大量注释数据集进行分割的能力。TDA还通过将拓扑特征明确映射到分割组件来提供更可解释和稳定的分割框架。

    We describe a new general method for segmentation in MRI scans using Topological Data Analysis (TDA), offering several advantages over traditional machine learning approaches. It works in three steps, first identifying the whole object to segment via automatic thresholding, then detecting a distinctive subset whose topology is known in advance, and finally deducing the various components of the segmentation. Although convoking classical ideas of TDA, such an algorithm has never been proposed separately from deep learning methods. To achieve this, our approach takes into account, in addition to the homology of the image, the localization of representative cycles, a piece of information that seems never to have been exploited in this context. In particular, it offers the ability to perform segmentation without the need for large annotated data sets. TDA also provides a more interpretable and stable framework for segmentation by explicitly mapping topological features to segmentation comp
    
[^2]: 一种针对压缩视频的时间句子对齐的有效和高效管道

    You Can Ground Earlier than See: An Effective and Efficient Pipeline for Temporal Sentence Grounding in Compressed Videos. (arXiv:2303.07863v1 [cs.CV])

    [http://arxiv.org/abs/2303.07863](http://arxiv.org/abs/2303.07863)

    本文提出了一种新的压缩域TSG设置，通过直接编码压缩位流来增强视觉特征表示能力和提高时间句子对齐的效率，并且在两个基准数据集的实验中表现优于目前最先进的方法。

    

    时间句子对齐旨在根据句子查询通过语义定位目标瞬间。在本文中，我们提出了一种新的压缩域TSG（Temporal Sentence Grounding）设置，直接使用压缩视频作为视觉输入。针对原始视频比特流输入，我们提出了一种新型三支路压缩空间时间融合框架（TCSF），用于有效且高效地定位。我们通过利用压缩伪影来增强视觉特征的表示能力，提出了一种直接编码压缩位流的方法，而不是先解码整个帧的方法。在两个基准数据集上的实验结果表明，我们的方法在效果和效率方面优于目前最先进的方法。

    Given an untrimmed video, temporal sentence grounding (TSG) aims to locate a target moment semantically according to a sentence query. Although previous respectable works have made decent success, they only focus on high-level visual features extracted from the consecutive decoded frames and fail to handle the compressed videos for query modelling, suffering from insufficient representation capability and significant computational complexity during training and testing. In this paper, we pose a new setting, compressed-domain TSG, which directly utilizes compressed videos rather than fully-decompressed frames as the visual input. To handle the raw video bit-stream input, we propose a novel Three-branch Compressed-domain Spatial-temporal Fusion (TCSF) framework, which extracts and aggregates three kinds of low-level visual features (I-frame, motion vector and residual features) for effective and efficient grounding. Particularly, instead of encoding the whole decoded frames like previous
    

