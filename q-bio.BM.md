# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Lightweight Contrastive Protein Structure-Sequence Transformation.](http://arxiv.org/abs/2303.11783) | 该论文提出了一种新的无监督学习的蛋白质结构表示预训练方法，使用强大的蛋白质语言模型和自监督结构约束，避免了破坏真实的空间结构表示和标记数据的限制。 |

# 详细

[^1]: 轻量级对比蛋白质结构-序列变换

    Lightweight Contrastive Protein Structure-Sequence Transformation. (arXiv:2303.11783v1 [q-bio.BM])

    [http://arxiv.org/abs/2303.11783](http://arxiv.org/abs/2303.11783)

    该论文提出了一种新的无监督学习的蛋白质结构表示预训练方法，使用强大的蛋白质语言模型和自监督结构约束，避免了破坏真实的空间结构表示和标记数据的限制。

    

    在大多数蛋白质下游应用中，无标签的预训练蛋白质结构模型是关键基础。传统的结构预训练方法遵循成熟的自然语言预训练方法，例如去噪重构和掩码语言建模，但通常会破坏真实的空间结构表示。其他常见的预训练方法可能会预测一组固定的预定对象类别，其中受限的监督方式限制了它们的通用性和可用性，因为需要额外的标记数据来指定任何其他的蛋白质概念。在这项工作中，我们引入了一种新的无监督蛋白质结构表示预训练方法，其中使用强大的蛋白质语言模型。特别地，我们首先建议利用现有的预训练语言模型通过无监督的对比对齐来指导结构模型的学习。此外，我们提出了一种自监督结构约束，以进一步学习内在的蛋白质结构表示形式。

    Pretrained protein structure models without labels are crucial foundations for the majority of protein downstream applications. The conventional structure pretraining methods follow the mature natural language pretraining methods such as denoised reconstruction and masked language modeling but usually destroy the real representation of spatial structures. The other common pretraining methods might predict a fixed set of predetermined object categories, where a restricted supervised manner limits their generality and usability as additional labeled data is required to specify any other protein concepts. In this work, we introduce a novel unsupervised protein structure representation pretraining with a robust protein language model. In particular, we first propose to leverage an existing pretrained language model to guide structure model learning through an unsupervised contrastive alignment. In addition, a self-supervised structure constraint is proposed to further learn the intrinsic i
    

