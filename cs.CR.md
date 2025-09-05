# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Universally Robust Graph Neural Networks by Preserving Neighbor Similarity.](http://arxiv.org/abs/2401.09754) | 本文通过保持邻居相似性实现了普适鲁棒的图神经网络，并在异类图上探索了图神经网络的脆弱性。理论上证明了负分类损失的更新与基于邻居特征的成对相似性呈负相关，解释了图攻击者连接不相似节点对的行为。通过这种方法，我们新颖地提出了一种解决方案。 |

# 详细

[^1]: 通过保持邻居相似性实现普适鲁棒的图神经网络

    Universally Robust Graph Neural Networks by Preserving Neighbor Similarity. (arXiv:2401.09754v1 [cs.LG])

    [http://arxiv.org/abs/2401.09754](http://arxiv.org/abs/2401.09754)

    本文通过保持邻居相似性实现了普适鲁棒的图神经网络，并在异类图上探索了图神经网络的脆弱性。理论上证明了负分类损失的更新与基于邻居特征的成对相似性呈负相关，解释了图攻击者连接不相似节点对的行为。通过这种方法，我们新颖地提出了一种解决方案。

    

    尽管图神经网络在学习关系数据方面取得了巨大成功，但已经广泛研究发现，图神经网络在同类图上容易受到结构攻击的影响。受此启发，我们提出了一系列鲁棒模型，以增强图神经网络在同类图上的对抗鲁棒性。然而，关于异类图上的脆弱性仍然存在许多未解之谜。为了弥合这一差距，本文开始探索图神经网络在异类图上的脆弱性，并在理论上证明了负分类损失的更新与基于邻居特征的幂和聚合的成对相似性呈负相关。这一理论证明解释了实证观察，即图攻击者倾向于基于邻居特征而不是个体特征连接不相似节点对，无论是在同类图还是异类图上。通过这种方式，我们新颖地引入了一种方法

    Despite the tremendous success of graph neural networks in learning relational data, it has been widely investigated that graph neural networks are vulnerable to structural attacks on homophilic graphs. Motivated by this, a surge of robust models is crafted to enhance the adversarial robustness of graph neural networks on homophilic graphs. However, the vulnerability based on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph neural networks on heterophilic graphs and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. This theoretical proof explains the empirical observations that the graph attacker tends to connect dissimilar node pairs based on the similarities of neighbor features instead of ego features both on homophilic and heterophilic graphs. In this way, we novelly introduce a
    

