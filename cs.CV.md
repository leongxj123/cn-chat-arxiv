# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention](https://arxiv.org/abs/2403.10173) | 提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。 |
| [^2] | [On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions](https://arxiv.org/abs/2402.16442) | 本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。 |
| [^3] | [Uncovering Hidden Connections: Iterative Tracking and Reasoning for Video-grounded Dialog.](http://arxiv.org/abs/2310.07259) | 本文提出了一种迭代跟踪和推理策略，结合文本编码器和视觉编码器以生成准确的响应，解决了视频对话中逐步理解对话历史和吸收视频信息的挑战。 |
| [^4] | [RelationMatch: Matching In-batch Relationships for Semi-supervised Learning.](http://arxiv.org/abs/2305.10397) | RelationMatch是一种利用矩阵交叉熵（MCE）损失函数的方法，可以匹配批内关系，有效提高半监督学习和监督学习的性能。 |

# 详细

[^1]: 一种用于基于事件的对象检测的混合SNN-ANN网络，具有空间和时间注意力机制

    A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention

    [https://arxiv.org/abs/2403.10173](https://arxiv.org/abs/2403.10173)

    提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。

    

    事件相机提供高时间分辨率和动态范围，几乎没有运动模糊，非常适合对象检测任务。尖峰神经网络（SNN）与事件驱动感知数据天生匹配，在神经形态硬件上能够实现超低功耗和低延迟推断，而人工神经网络（ANN）则展示出更稳定的训练动态和更快的收敛速度，从而具有更好的任务性能。混合SNN-ANN方法是一种有前途的替代方案，能够利用SNN和ANN体系结构的优势。在这项工作中，我们引入了第一个基于混合注意力的SNN-ANN骨干网络，用于使用事件相机进行对象检测。我们提出了一种新颖的基于注意力的SNN-ANN桥接模块，从SNN层中捕捉稀疏的空间和时间关系，并将其转换为密集特征图，供骨干网络的ANN部分使用。实验结果表明，我们提出的m

    arXiv:2403.10173v1 Announce Type: cross  Abstract: Event cameras offer high temporal resolution and dynamic range with minimal motion blur, making them promising for object detection tasks. While Spiking Neural Networks (SNNs) are a natural match for event-based sensory data and enable ultra-energy efficient and low latency inference on neuromorphic hardware, Artificial Neural Networks (ANNs) tend to display more stable training dynamics and faster convergence resulting in greater task performance. Hybrid SNN-ANN approaches are a promising alternative, enabling to leverage the strengths of both SNN and ANN architectures. In this work, we introduce the first Hybrid Attention-based SNN-ANN backbone for object detection using event cameras. We propose a novel Attention-based SNN-ANN bridge module to capture sparse spatial and temporal relations from the SNN layer and convert them into dense feature maps for the ANN part of the backbone. Experimental results demonstrate that our proposed m
    
[^2]: 在具有配对次模模函数的分布式大于内存的子集选择问题研究

    On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions

    [https://arxiv.org/abs/2402.16442](https://arxiv.org/abs/2402.16442)

    本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    

    许多学习问题取决于子集选择的基本问题，即确定一组重要和代表性的点。本文提出了一种具有可证估计近似保证的新颖分布式约束算法，它通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    arXiv:2402.16442v1 Announce Type: cross  Abstract: Many learning problems hinge on the fundamental problem of subset selection, i.e., identifying a subset of important and representative points. For example, selecting the most significant samples in ML training cannot only reduce training costs but also enhance model quality. Submodularity, a discrete analogue of convexity, is commonly used for solving subset selection problems. However, existing algorithms for optimizing submodular functions are sequential, and the prior distributed methods require at least one central machine to fit the target subset. In this paper, we relax the requirement of having a central machine for the target subset by proposing a novel distributed bounding algorithm with provable approximation guarantees. The algorithm iteratively bounds the minimum and maximum utility values to select high quality points and discard the unimportant ones. When bounding does not find the complete subset, we use a multi-round, 
    
[^3]: 揭示隐藏的联系：用于视频对话的迭代跟踪和推理

    Uncovering Hidden Connections: Iterative Tracking and Reasoning for Video-grounded Dialog. (arXiv:2310.07259v1 [cs.CV])

    [http://arxiv.org/abs/2310.07259](http://arxiv.org/abs/2310.07259)

    本文提出了一种迭代跟踪和推理策略，结合文本编码器和视觉编码器以生成准确的响应，解决了视频对话中逐步理解对话历史和吸收视频信息的挑战。

    

    与传统的视觉问答相比，视频对话需要对对话历史和视频内容进行深入理解，以生成准确的响应。尽管现有的方法取得了令人称赞的进展，但它们常常面临逐步理解复杂的对话历史和吸收视频信息的挑战。为了弥补这一差距，我们提出了一种迭代跟踪和推理策略，将文本编码器、视觉编码器和生成器相结合。我们的文本编码器以路径跟踪和聚合机制为核心，能够从对话历史中获取重要的细微差别，以解释所提出的问题。同时，我们的视觉编码器利用迭代推理网络，精心设计以从视频中提取和强调关键视觉标记，增强对视觉理解的深度。最后，我们使用预训练的GPT-模型将这些丰富的信息综合起来。

    In contrast to conventional visual question answering, video-grounded dialog necessitates a profound understanding of both dialog history and video content for accurate response generation. Despite commendable strides made by existing methodologies, they often grapple with the challenges of incrementally understanding intricate dialog histories and assimilating video information. In response to this gap, we present an iterative tracking and reasoning strategy that amalgamates a textual encoder, a visual encoder, and a generator. At its core, our textual encoder is fortified with a path tracking and aggregation mechanism, adept at gleaning nuances from dialog history that are pivotal to deciphering the posed questions. Concurrently, our visual encoder harnesses an iterative reasoning network, meticulously crafted to distill and emphasize critical visual markers from videos, enhancing the depth of visual comprehension. Culminating this enriched information, we employ the pre-trained GPT-
    
[^4]: RelationMatch：用于半监督学习的批内关系匹配技术

    RelationMatch: Matching In-batch Relationships for Semi-supervised Learning. (arXiv:2305.10397v1 [cs.LG])

    [http://arxiv.org/abs/2305.10397](http://arxiv.org/abs/2305.10397)

    RelationMatch是一种利用矩阵交叉熵（MCE）损失函数的方法，可以匹配批内关系，有效提高半监督学习和监督学习的性能。

    

    半监督学习通过利用少量标记数据和未标记数据中的信息，已经在许多领域取得了显着的成功。然而，现有算法通常集中在来自相同来源的成对数据点的预测对准上，并忽略了每个批次内的点间关系。本文介绍了一种新方法RelationMatch，它利用一种矩阵交叉熵（MCE）损失函数来发掘批内关系。通过应用MCE，我们的方法在各种视觉数据集中始终优于现有最先进的方法，如FixMatch和FlexMatch。值得注意的是，在仅使用40个标签的STL-10数据集上，我们观察到相对于FlexMatch有15.21％的显著提高。此外，我们将MCE应用于监督学习场景，并观察到了一致的改进。

    Semi-supervised learning has achieved notable success by leveraging very few labeled data and exploiting the wealth of information derived from unlabeled data. However, existing algorithms usually focus on aligning predictions on paired data points augmented from an identical source, and overlook the inter-point relationships within each batch. This paper introduces a novel method, RelationMatch, which exploits in-batch relationships with a matrix cross-entropy (MCE) loss function. Through the application of MCE, our proposed method consistently surpasses the performance of established state-of-the-art methods, such as FixMatch and FlexMatch, across a variety of vision datasets. Notably, we observed a substantial enhancement of 15.21% in accuracy over FlexMatch on the STL-10 dataset using only 40 labels. Moreover, we apply MCE to supervised learning scenarios, and observe consistent improvements as well.
    

