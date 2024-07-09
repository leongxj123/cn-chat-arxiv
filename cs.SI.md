# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum](https://arxiv.org/abs/2312.06441) | 本文提出了一种基于半监督GNN的欺诈检测器SEC-GFD，通过混合过滤模块和局部环境约束模块解决了异质性和标签利用问题。 |

# 详细

[^1]: 在异质性和谱问题下重新审视基于图的欺诈检测

    Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum

    [https://arxiv.org/abs/2312.06441](https://arxiv.org/abs/2312.06441)

    本文提出了一种基于半监督GNN的欺诈检测器SEC-GFD，通过混合过滤模块和局部环境约束模块解决了异质性和标签利用问题。

    

    基于图的欺诈检测（GFD）可视为一项具有挑战性的半监督节点二分类任务。近年来，图神经网络（GNN）已广泛应用于GFD，通过聚合邻居信息来刻画节点的异常可能性。然而，欺诈图在本质上是异质的，因此大多数GNN由于假设同质性而表现不佳。此外，由于存在异质性和类别不平衡问题，现有模型未充分利用宝贵的节点标签信息。为了解决上述问题，本文提出了一种基于半监督GNN的欺诈检测器SEC-GFD。该检测器包括混合过滤模块和局部环境约束模块，这两个模块分别用于解决异质性和标签利用问题。第一个模块从谱域的角度出发，在一定程度上解决了异质性问题。具体而言，它将图分割称不同的谱成分，

    Graph-based fraud detection (GFD) can be regarded as a challenging semi-supervised node binary classification task. In recent years, Graph Neural Networks (GNN) have been widely applied to GFD, characterizing the anomalous possibility of a node by aggregating neighbor information. However, fraud graphs are inherently heterophilic, thus most of GNNs perform poorly due to their assumption of homophily. In addition, due to the existence of heterophily and class imbalance problem, the existing models do not fully utilize the precious node label information. To address the above issues, this paper proposes a semi-supervised GNN-based fraud detector SEC-GFD. This detector includes a hybrid filtering module and a local environmental constraint module, the two modules are utilized to solve heterophily and label utilization problem respectively. The first module starts from the perspective of the spectral domain, and solves the heterophily problem to a certain extent. Specifically, it divides t
    

