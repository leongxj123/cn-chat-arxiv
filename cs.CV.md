# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variational Autoencoding of Dental Point Clouds.](http://arxiv.org/abs/2307.10895) | 本论文介绍了一种新颖的点云变分自编码器（VF-Net）用于牙科点云数据的处理，该模型在各种任务中具有显著的性能，包括网格生成、形状完整和表示学习。 |

# 详细

[^1]: 牙科点云的变分自编码

    Variational Autoencoding of Dental Point Clouds. (arXiv:2307.10895v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.10895](http://arxiv.org/abs/2307.10895)

    本论文介绍了一种新颖的点云变分自编码器（VF-Net）用于牙科点云数据的处理，该模型在各种任务中具有显著的性能，包括网格生成、形状完整和表示学习。

    

    数字牙科学取得了重大进展，但仍面临许多挑战。本文介绍了FDI 16数据集，这是一个包含了大量牙齿网格和点云的数据集。此外，我们还提出了一种新颖的方法：变分FoldingNet（VF-Net），这是一种专为点云设计的完全概率变分自编码器。值得注意的是，先前的点云潜变量模型缺乏输入和输出点之间的一一对应关系。相反，它们依赖于优化Chamfer距离，这是一种缺乏归一化分布对应的度量，因此不适合概率建模。我们用合适的编码器取代了明确的最小化Chamfer距离，提高了计算效率，同时简化了概率扩展。这使得在各种任务中都可以直接应用，包括网格生成、形状完整和表示学习。在实证方面，我们提供了牙齿重建中较低的重建误差的证据。

    Digital dentistry has made significant advancements, yet numerous challenges remain. This paper introduces the FDI 16 dataset, an extensive collection of tooth meshes and point clouds. Additionally, we present a novel approach: Variational FoldingNet (VF-Net), a fully probabilistic variational autoencoder designed for point clouds. Notably, prior latent variable models for point clouds lack a one-to-one correspondence between input and output points. Instead, they rely on optimizing Chamfer distances, a metric that lacks a normalized distributional counterpart, rendering it unsuitable for probabilistic modeling. We replace the explicit minimization of Chamfer distances with a suitable encoder, increasing computational efficiency while simplifying the probabilistic extension. This allows for straightforward application in various tasks, including mesh generation, shape completion, and representation learning. Empirically, we provide evidence of lower reconstruction error in dental recon
    

