# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InVA: Integrative Variational Autoencoder for Harmonization of Multi-modal Neuroimaging Data](https://arxiv.org/abs/2402.02734) | InVA是一种综合变分自编码器方法，利用多模态神经影像数据中不同来源的多个图像来进行预测推理，相较于传统的VAE方法具有更好的效果。 |
| [^2] | [TSGCNeXt: Dynamic-Static Multi-Graph Convolution for Efficient Skeleton-Based Action Recognition with Long-term Learning Potential.](http://arxiv.org/abs/2304.11631) | TSGCNeXt是一个用于基于骨架的动作识别的模型，具有长期学习潜力，它采用动静态多图卷积来汇集多个独立拓扑图的特征，以及构建了一个图形卷积训练加速机制。 |

# 详细

[^1]: InVA: 综合变分自编码器用于多模态神经影像数据的协调

    InVA: Integrative Variational Autoencoder for Harmonization of Multi-modal Neuroimaging Data

    [https://arxiv.org/abs/2402.02734](https://arxiv.org/abs/2402.02734)

    InVA是一种综合变分自编码器方法，利用多模态神经影像数据中不同来源的多个图像来进行预测推理，相较于传统的VAE方法具有更好的效果。

    

    在探索多个来自不同成像模式的图像之间的非线性关联方面具有重要意义。尽管有越来越多的文献研究基于多个图像来推断图像的预测推理，但现有方法在有效借用多个成像模式之间的信息来预测图像方面存在局限。本文建立在变分自编码器（VAEs）的文献基础上，提出了一种新颖的方法，称为综合变分自编码器（InVA）方法，它从不同来源获得的多个图像中借用信息来绘制图像的预测推理。所提出的方法捕捉了结果图像与输入图像之间的复杂非线性关联，并允许快速计算。数值结果表明，InVA相对于通常不允许借用输入图像之间信息的VAE具有明显的优势。

    There is a significant interest in exploring non-linear associations among multiple images derived from diverse imaging modalities. While there is a growing literature on image-on-image regression to delineate predictive inference of an image based on multiple images, existing approaches have limitations in efficiently borrowing information between multiple imaging modalities in the prediction of an image. Building on the literature of Variational Auto Encoders (VAEs), this article proposes a novel approach, referred to as Integrative Variational Autoencoder (\texttt{InVA}) method, which borrows information from multiple images obtained from different sources to draw predictive inference of an image. The proposed approach captures complex non-linear association between the outcome image and input images, while allowing rapid computation. Numerical results demonstrate substantial advantages of \texttt{InVA} over VAEs, which typically do not allow borrowing information between input imag
    
[^2]: TSGCNeXt：具备长期学习潜力的高效基于骨架的动作识别的动静态多图卷积

    TSGCNeXt: Dynamic-Static Multi-Graph Convolution for Efficient Skeleton-Based Action Recognition with Long-term Learning Potential. (arXiv:2304.11631v1 [cs.CV])

    [http://arxiv.org/abs/2304.11631](http://arxiv.org/abs/2304.11631)

    TSGCNeXt是一个用于基于骨架的动作识别的模型，具有长期学习潜力，它采用动静态多图卷积来汇集多个独立拓扑图的特征，以及构建了一个图形卷积训练加速机制。

    

    随着图卷积网络（GCN）的发展，基于骨架的动作识别在人类动作识别方面取得了显著的成果。然而，最近的研究趋向于构建具有冗余训练的复杂学习机制，并存在长时间序列的瓶颈。为了解决这些问题，我们提出了Temporal-Spatio Graph ConvNeXt（TSGCNeXt）来探索长时间骨骼序列的高效学习机制。首先，我们提出了一个新的图形学习机制，动静分离多图卷积（DS-SMG），以汇集多个独立拓扑图的特征并避免节点信息在动态卷积期间被忽略。接下来，我们构建了一个图形卷积训练加速机制，以55.08％的速度提高动态图形学习的反向传播计算速度。最后，TSGCNeXt通过三个时空学习模块重新构建了GCN的整体结构，实现了更加高效的基于骨架的动作识别。

    Skeleton-based action recognition has achieved remarkable results in human action recognition with the development of graph convolutional networks (GCNs). However, the recent works tend to construct complex learning mechanisms with redundant training and exist a bottleneck for long time-series. To solve these problems, we propose the Temporal-Spatio Graph ConvNeXt (TSGCNeXt) to explore efficient learning mechanism of long temporal skeleton sequences. Firstly, a new graph learning mechanism with simple structure, Dynamic-Static Separate Multi-graph Convolution (DS-SMG) is proposed to aggregate features of multiple independent topological graphs and avoid the node information being ignored during dynamic convolution. Next, we construct a graph convolution training acceleration mechanism to optimize the back-propagation computing of dynamic graph learning with 55.08\% speed-up. Finally, the TSGCNeXt restructure the overall structure of GCN with three Spatio-temporal learning modules,effic
    

