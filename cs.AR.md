# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices.](http://arxiv.org/abs/2311.01759) | TinyFormer是一个具有SuperNAS、SparseNAS和SparseEngine组成的框架，专门用于在MCUs上开发和部署资源高效的transformer模型。其创新之处在于提出了SparseEngine，这是第一个可以在MCUs上执行稀疏模型的transformer推理的部署框架。 |

# 详细

[^1]: TinyFormer: 高效的Transformer设计和在小型设备上的部署

    TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices. (arXiv:2311.01759v1 [cs.LG])

    [http://arxiv.org/abs/2311.01759](http://arxiv.org/abs/2311.01759)

    TinyFormer是一个具有SuperNAS、SparseNAS和SparseEngine组成的框架，专门用于在MCUs上开发和部署资源高效的transformer模型。其创新之处在于提出了SparseEngine，这是第一个可以在MCUs上执行稀疏模型的transformer推理的部署框架。

    

    在各种嵌入式物联网应用中，以微控制器单元（MCUs）为代表的小型设备上开发深度学习模型引起了广泛关注。然而，由于严重的硬件资源限制，如何高效地设计和部署最新的先进模型（如transformer）在小型设备上是一项挑战。在这项工作中，我们提出了TinyFormer，这是一个特别设计用于在MCUs上开发和部署资源高效的transformer的框架。TinyFormer主要由SuperNAS、SparseNAS和SparseEngine组成。其中，SuperNAS旨在从广大的搜索空间中寻找适当的超网络。SparseNAS评估最佳的稀疏单路径模型，包括从已识别的超网络中提取的transformer架构。最后，SparseEngine将搜索到的稀疏模型高效地部署到MCUs上。据我们所知，SparseEngine是第一个能够在MCUs上执行稀疏模型的transformer推理的部署框架。在CIFAR-10数据集上的评估结果表明，TinyFormer在保持推理精度的同时，相比于传统的transformer模型，减少了大约78％的推理计算量和53％的模型大小。

    Developing deep learning models on tiny devices (e.g. Microcontroller units, MCUs) has attracted much attention in various embedded IoT applications. However, it is challenging to efficiently design and deploy recent advanced models (e.g. transformers) on tiny devices due to their severe hardware resource constraints. In this work, we propose TinyFormer, a framework specifically designed to develop and deploy resource-efficient transformers on MCUs. TinyFormer mainly consists of SuperNAS, SparseNAS and SparseEngine. Separately, SuperNAS aims to search for an appropriate supernet from a vast search space. SparseNAS evaluates the best sparse single-path model including transformer architecture from the identified supernet. Finally, SparseEngine efficiently deploys the searched sparse models onto MCUs. To the best of our knowledge, SparseEngine is the first deployment framework capable of performing inference of sparse models with transformer on MCUs. Evaluation results on the CIFAR-10 da
    

