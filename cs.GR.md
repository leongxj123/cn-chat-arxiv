# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-Level-Set Encoder for Neural Distance Fields](https://arxiv.org/abs/2310.06644) | 本文提出了一种用于嵌入3D形状的神经网络架构，通过多尺度混合系统和连续可微的解码器，不仅能够生成有效的有符号距离场，还能够在训练和推断中仅使用零水平集的知识。同时，还提出了针对曲面法线不存在情况的损失函数修改。 |

# 详细

[^1]: 神经距离场的零水平集编码器

    Zero-Level-Set Encoder for Neural Distance Fields

    [https://arxiv.org/abs/2310.06644](https://arxiv.org/abs/2310.06644)

    本文提出了一种用于嵌入3D形状的神经网络架构，通过多尺度混合系统和连续可微的解码器，不仅能够生成有效的有符号距离场，还能够在训练和推断中仅使用零水平集的知识。同时，还提出了针对曲面法线不存在情况的损失函数修改。

    

    神经形状表示通常指使用神经网络来表示3D几何，例如，在特定空间位置计算有符号距离或占据值。本文提出了一种新颖的编码器-解码器神经网络，用于在单次前向传递中嵌入3D形状。我们的架构基于多尺度混合系统，包括基于图形和基于体素的组件，以及连续可微的解码器。此外，该网络经过训练以解决Eikonal方程，仅需要零水平集的知识进行训练和推断。这意味着，与大多数之前的工作相比，我们的网络能够输出有效的有符号距离场，而无需明确的非零距离值或形状占据的先验知识。我们还提出了一种损失函数的修改，以解决曲面法线不存在的情况，例如，非封闭曲面和非流形几何的上下文。总体上，这可以帮助减少必要的先验知识。

    Neural shape representation generally refers to representing 3D geometry using neural networks, e.g., to compute a signed distance or occupancy value at a specific spatial position. In this paper, we present a novel encoder-decoder neural network for embedding 3D shapes in a single forward pass. Our architecture is based on a multi-scale hybrid system incorporating graph-based and voxel-based components, as well as a continuously differentiable decoder. Furthermore, the network is trained to solve the Eikonal equation and only requires knowledge of the zero-level set for training and inference. This means that in contrast to most previous work, our network is able to output valid signed distance fields without explicit prior knowledge of non-zero distance values or shape occupancy. We further propose a modification of the loss function in case that surface normals are not well defined, e.g., in the context of non-watertight surfaces and non-manifold geometry. Overall, this can help red
    

