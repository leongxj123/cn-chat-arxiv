# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [N2F2: Hierarchical Scene Understanding with Nested Neural Feature Fields](https://arxiv.org/abs/2403.10997) | 利用Nested Neural Feature Fields (N2F2) 实现了层次化监督学习，提供了对物理维度或语义维度等不同粒度的场景属性全面和细致的理解。 |

# 详细

[^1]: 嵌套神经特征场的层次场景理解

    N2F2: Hierarchical Scene Understanding with Nested Neural Feature Fields

    [https://arxiv.org/abs/2403.10997](https://arxiv.org/abs/2403.10997)

    利用Nested Neural Feature Fields (N2F2) 实现了层次化监督学习，提供了对物理维度或语义维度等不同粒度的场景属性全面和细致的理解。

    

    在计算机视觉中，理解多层抽象的复杂场景仍然是一个巨大挑战。为了解决这个问题，我们引入了嵌套神经特征场 (N2F2)，这是一种新颖的方法，利用分层监督来学习单个特征场，在同一高维特征中的不同维度编码不同粒度的场景属性。我们的方法允许灵活定义层次，可以根据物理维度、语义维度或两者均匹配，从而实现对场景的全面和细致理解。我们利用2D类别无关分割模型在图像空间的任意尺度提供语义有意义的像素分组，并查询CLIP视觉编码器，为这些段落中的每个部分获得与语言对齐的嵌入。我们提出的分层监督方法将不同的嵌套特征场维度分配给提取C

    arXiv:2403.10997v1 Announce Type: cross  Abstract: Understanding complex scenes at multiple levels of abstraction remains a formidable challenge in computer vision. To address this, we introduce Nested Neural Feature Fields (N2F2), a novel approach that employs hierarchical supervision to learn a single feature field, wherein different dimensions within the same high-dimensional feature encode scene properties at varying granularities. Our method allows for a flexible definition of hierarchies, tailored to either the physical dimensions or semantics or both, thereby enabling a comprehensive and nuanced understanding of scenes. We leverage a 2D class-agnostic segmentation model to provide semantically meaningful pixel groupings at arbitrary scales in the image space, and query the CLIP vision-encoder to obtain language-aligned embeddings for each of these segments. Our proposed hierarchical supervision method then assigns different nested dimensions of the feature field to distill the C
    

