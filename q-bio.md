# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Phylo2Vec: a vector representation for binary trees.](http://arxiv.org/abs/2304.12693) | Phylo2Vec是一种新的二叉树简明表示方法，它能够轻松采样二叉树，并以系统性的方法遍历树空间。这种方法用于构建深度神经网络，能够显著提高蛋白质类别预测的性能。 |

# 详细

[^1]: Phylo2Vec: 一种二叉树的向量表示方法

    Phylo2Vec: a vector representation for binary trees. (arXiv:2304.12693v1 [q-bio.PE])

    [http://arxiv.org/abs/2304.12693](http://arxiv.org/abs/2304.12693)

    Phylo2Vec是一种新的二叉树简明表示方法，它能够轻松采样二叉树，并以系统性的方法遍历树空间。这种方法用于构建深度神经网络，能够显著提高蛋白质类别预测的性能。

    

    从生物数据推断得到的二叉进化树对于理解生物之间共享的进化历史至关重要。根据最大似然等某个最优性准则推断出树中潜在节点的位置是NP-hard问题，这推动了大量启发式方法的发展。然而，这些启发式方法通常缺乏一种系统性的方法来均匀采样随机树或有效地探索指数级增长的树空间，这对于机器学习等优化问题至关重要。因此，我们提出了Phylo2Vec，这是一种新的简明表示方法来表示进化树。Phylo2Vec将任何具有n个叶子的二叉树映射到长度为n的整数向量。我们证明了Phylo2Vec在空间中既是良定的又是双射的。Phylo2Vec的优点是：i）轻松均匀采样二叉树；ii）以非常大或小的步长系统地遍历树空间。作为概念验证，我们使用Phylo2Vec构建了一个深度神经网络，以从氨基酸序列预测蛋白质类别。我们证明了Phylo2Vec显著提高了网络的性能，超过了之前的最优结果。

    Binary phylogenetic trees inferred from biological data are central to understanding the shared evolutionary history of organisms. Inferring the placement of latent nodes in a tree by any optimality criterion (e.g., maximum likelihood) is an NP-hard problem, propelling the development of myriad heuristic approaches. Yet, these heuristics often lack a systematic means of uniformly sampling random trees or effectively exploring a tree space that grows factorially, which are crucial to optimisation problems such as machine learning. Accordingly, we present Phylo2Vec, a new parsimonious representation of a phylogenetic tree. Phylo2Vec maps any binary tree with $n$ leaves to an integer vector of length $n$. We prove that Phylo2Vec is both well-defined and bijective to the space of phylogenetic trees. The advantages of Phylo2Vec are twofold: i) easy uniform sampling of binary trees and ii) systematic ability to traverse tree space in very large or small jumps. As a proof of concept, we use P
    

