# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Coded Federated Learning: Privacy Preservation and Straggler Mitigation](https://arxiv.org/abs/2403.14905) | ACFL提出了一种新的自适应编码联邦学习方法，通过在训练之前采用个性化的数据上传到中央服务器来生成全局编码数据集，以解决原有固定权重生成全局编码数据集时可能导致学习性能下降的问题。 |

# 详细

[^1]: 自适应编码联邦学习：隐私保护与慢节点缓解

    Adaptive Coded Federated Learning: Privacy Preservation and Straggler Mitigation

    [https://arxiv.org/abs/2403.14905](https://arxiv.org/abs/2403.14905)

    ACFL提出了一种新的自适应编码联邦学习方法，通过在训练之前采用个性化的数据上传到中央服务器来生成全局编码数据集，以解决原有固定权重生成全局编码数据集时可能导致学习性能下降的问题。

    

    在本文中，我们讨论了在存在慢节点情况下的联邦学习问题。针对这一问题，我们提出了一种编码联邦学习框架，其中中央服务器聚合来自非慢节点的梯度和来自隐私保护全局编码数据集的梯度，以减轻慢节点的负面影响。然而，在聚合这些梯度时，固定权重在迭代中一直被应用，忽略了全局编码数据集的生成过程以及训练模型随着迭代的动态性。这一疏漏可能导致学习性能下降。为克服这一缺陷，我们提出了一种名为自适应编码联邦学习（ACFL）的新方法。在ACFL中，在训练之前，每个设备向中央服务器上传一个带有附加噪声的编码本地数据集，以生成符合隐私保护要求的全局编码数据集。在...

    arXiv:2403.14905v1 Announce Type: cross  Abstract: In this article, we address the problem of federated learning in the presence of stragglers. For this problem, a coded federated learning framework has been proposed, where the central server aggregates gradients received from the non-stragglers and gradient computed from a privacy-preservation global coded dataset to mitigate the negative impact of the stragglers. However, when aggregating these gradients, fixed weights are consistently applied across iterations, neglecting the generation process of the global coded dataset and the dynamic nature of the trained model over iterations. This oversight may result in diminished learning performance. To overcome this drawback, we propose a new method named adaptive coded federated learning (ACFL). In ACFL, before the training, each device uploads a coded local dataset with additive noise to the central server to generate a global coded dataset under privacy preservation requirements. During
    

