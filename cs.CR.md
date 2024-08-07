# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Local Differential Privacy in Graph Neural Networks: a Reconstruction Approach.](http://arxiv.org/abs/2309.08569) | 本研究提出了一种学习框架，旨在为用户提供节点隐私保护，并通过在节点级别对特征和标签数据进行随机化扰动来实现。通过频率估计和重构方法，实现了对扰动数据中特征和标签的恢复。 |

# 详细

[^1]: 图神经网络中的局部差分隐私：一种重构方法

    Local Differential Privacy in Graph Neural Networks: a Reconstruction Approach. (arXiv:2309.08569v1 [cs.LG])

    [http://arxiv.org/abs/2309.08569](http://arxiv.org/abs/2309.08569)

    本研究提出了一种学习框架，旨在为用户提供节点隐私保护，并通过在节点级别对特征和标签数据进行随机化扰动来实现。通过频率估计和重构方法，实现了对扰动数据中特征和标签的恢复。

    

    图神经网络在各种应用中对建模复杂图数据取得了巨大成功。然而，有关GNN的隐私保护的研究还很有限。在本文中，我们提出了一个学习框架，可以在不丧失太多效用的情况下为用户提供节点隐私保护。我们关注一种去中心化的差分隐私概念，即局部差分隐私，并在数据被集中服务器进行模型训练之前，对节点级别的特征和标签数据应用随机化机制进行扰动。具体而言，我们研究了在高维特征设置中应用随机化机制的方法，并提出了具有严格隐私保证的LDP协议。基于随机化数据的统计分析中的频率估计，我们开发了重构方法来近似从扰动数据中恢复特征和标签。我们还制定了这个学习框架，利用了图聚类中的频率估计。

    Graph Neural Networks have achieved tremendous success in modeling complex graph data in a variety of applications. However, there are limited studies investigating privacy protection in GNNs. In this work, we propose a learning framework that can provide node privacy at the user level, while incurring low utility loss. We focus on a decentralized notion of Differential Privacy, namely Local Differential Privacy, and apply randomization mechanisms to perturb both feature and label data at the node level before the data is collected by a central server for model training. Specifically, we investigate the application of randomization mechanisms in high-dimensional feature settings and propose an LDP protocol with strict privacy guarantees. Based on frequency estimation in statistical analysis of randomized data, we develop reconstruction methods to approximate features and labels from perturbed data. We also formulate this learning framework to utilize frequency estimates of graph cluste
    

