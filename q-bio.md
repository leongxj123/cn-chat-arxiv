# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DDIPrompt: Drug-Drug Interaction Event Prediction based on Graph Prompt Learning](https://arxiv.org/abs/2402.11472) | 基于图提示学习的DDIPrompt框架旨在解决药物相互作用事件预测中的高度不平衡事件分布和罕见事件标记数据稀缺性问题。 |
| [^2] | [Bidirectional Graph GAN: Representing Brain Structure-Function Connections for Alzheimer's Disease.](http://arxiv.org/abs/2309.08916) | 本研究提出了一种双向图生成对抗网络（BGGAN），用于表示阿尔茨海默病（AD）的脑结构-功能连接。通过特殊设计的内部图卷积网络模块和平衡器模块，该方法能够准确地学习结构域和功能域之间的映射函数，并解决模式坍塌问题，同时学习结构和功能特征的互补性。 |
| [^3] | [Phylo2Vec: a vector representation for binary trees.](http://arxiv.org/abs/2304.12693) | Phylo2Vec是一种新的二叉树简明表示方法，它能够轻松采样二叉树，并以系统性的方法遍历树空间。这种方法用于构建深度神经网络，能够显著提高蛋白质类别预测的性能。 |

# 详细

[^1]: 基于图提示学习的药物相互作用事件预测：DDIPrompt

    DDIPrompt: Drug-Drug Interaction Event Prediction based on Graph Prompt Learning

    [https://arxiv.org/abs/2402.11472](https://arxiv.org/abs/2402.11472)

    基于图提示学习的DDIPrompt框架旨在解决药物相互作用事件预测中的高度不平衡事件分布和罕见事件标记数据稀缺性问题。

    

    最近，由于其在建模药物分子内部和之间原子和功能团之间复杂关联方面的熟练表现，图神经网络在预测药物相互作用事件（DDI）方面变得日益普遍。然而，它们仍然受到两个重大挑战的制约：（1）高度不平衡事件分布的问题，在医学数据集中这是一个常见但关键的问题，某些相互作用被广泛地低估。这种不平衡对实现准确可靠的DDI预测构成了重大障碍。（2）罕见事件标记数据的稀缺性，在医学领域是一个普遍问题，由于数据有限，往往忽视或研究不足的罕见但潜在关键的相互作用。为此，我们提出了DDIPrompt，这是一种受最近图提示学进展启发的创新良方。我们的框架旨在解决这些问题。

    arXiv:2402.11472v1 Announce Type: cross  Abstract: Recently, Graph Neural Networks have become increasingly prevalent in predicting adverse drug-drug interactions (DDI) due to their proficiency in modeling the intricate associations between atoms and functional groups within and across drug molecules. However, they are still hindered by two significant challenges: (1) the issue of highly imbalanced event distribution, which is a common but critical problem in medical datasets where certain interactions are vastly underrepresented. This imbalance poses a substantial barrier to achieving accurate and reliable DDI predictions. (2) the scarcity of labeled data for rare events, which is a pervasive issue in the medical field where rare yet potentially critical interactions are often overlooked or under-studied due to limited available data. In response, we offer DDIPrompt, an innovative panacea inspired by the recent advancements in graph prompting. Our framework aims to address these issue
    
[^2]: 双向图生成对抗网络：用于阿尔茨海默病的脑结构-功能连接的表示

    Bidirectional Graph GAN: Representing Brain Structure-Function Connections for Alzheimer's Disease. (arXiv:2309.08916v1 [cs.AI])

    [http://arxiv.org/abs/2309.08916](http://arxiv.org/abs/2309.08916)

    本研究提出了一种双向图生成对抗网络（BGGAN），用于表示阿尔茨海默病（AD）的脑结构-功能连接。通过特殊设计的内部图卷积网络模块和平衡器模块，该方法能够准确地学习结构域和功能域之间的映射函数，并解决模式坍塌问题，同时学习结构和功能特征的互补性。

    

    揭示脑疾病的发病机制，包括阿尔茨海默病（AD），脑结构与功能之间的关系至关重要。然而，由于各种原因，将脑结构-功能连接映射是一个巨大的挑战。本文提出了一种双向图生成对抗网络（BGGAN）来表示脑结构-功能连接。具体来说，通过设计一个内部图卷积网络（InnerGCN）模块，BGGAN的生成器可以利用直接和间接脑区域的特征来学习结构域和功能域之间的映射函数。此外，还设计了一个名为Balancer的新模块来平衡生成器和判别器之间的优化。通过将Balancer引入到BGGAN中，结构生成器和功能生成器不仅可以缓解模式坍塌问题，还可以学习结构和功能特征的互补性。实验结果表明该方法能够在AD中准确地表示脑结构-功能连接。

    The relationship between brain structure and function is critical for revealing the pathogenesis of brain disease, including Alzheimer's disease (AD). However, it is a great challenge to map brain structure-function connections due to various reasons. In this work, a bidirectional graph generative adversarial networks (BGGAN) is proposed to represent brain structure-function connections. Specifically, by designing a module incorporating inner graph convolution network (InnerGCN), the generators of BGGAN can employ features of direct and indirect brain regions to learn the mapping function between structural domain and functional domain. Besides, a new module named Balancer is designed to counterpoise the optimization between generators and discriminators. By introducing the Balancer into BGGAN, both the structural generator and functional generator can not only alleviate the issue of mode collapse but also learn complementarity of structural and functional features. Experimental result
    
[^3]: Phylo2Vec: 一种二叉树的向量表示方法

    Phylo2Vec: a vector representation for binary trees. (arXiv:2304.12693v1 [q-bio.PE])

    [http://arxiv.org/abs/2304.12693](http://arxiv.org/abs/2304.12693)

    Phylo2Vec是一种新的二叉树简明表示方法，它能够轻松采样二叉树，并以系统性的方法遍历树空间。这种方法用于构建深度神经网络，能够显著提高蛋白质类别预测的性能。

    

    从生物数据推断得到的二叉进化树对于理解生物之间共享的进化历史至关重要。根据最大似然等某个最优性准则推断出树中潜在节点的位置是NP-hard问题，这推动了大量启发式方法的发展。然而，这些启发式方法通常缺乏一种系统性的方法来均匀采样随机树或有效地探索指数级增长的树空间，这对于机器学习等优化问题至关重要。因此，我们提出了Phylo2Vec，这是一种新的简明表示方法来表示进化树。Phylo2Vec将任何具有n个叶子的二叉树映射到长度为n的整数向量。我们证明了Phylo2Vec在空间中既是良定的又是双射的。Phylo2Vec的优点是：i）轻松均匀采样二叉树；ii）以非常大或小的步长系统地遍历树空间。作为概念验证，我们使用Phylo2Vec构建了一个深度神经网络，以从氨基酸序列预测蛋白质类别。我们证明了Phylo2Vec显著提高了网络的性能，超过了之前的最优结果。

    Binary phylogenetic trees inferred from biological data are central to understanding the shared evolutionary history of organisms. Inferring the placement of latent nodes in a tree by any optimality criterion (e.g., maximum likelihood) is an NP-hard problem, propelling the development of myriad heuristic approaches. Yet, these heuristics often lack a systematic means of uniformly sampling random trees or effectively exploring a tree space that grows factorially, which are crucial to optimisation problems such as machine learning. Accordingly, we present Phylo2Vec, a new parsimonious representation of a phylogenetic tree. Phylo2Vec maps any binary tree with $n$ leaves to an integer vector of length $n$. We prove that Phylo2Vec is both well-defined and bijective to the space of phylogenetic trees. The advantages of Phylo2Vec are twofold: i) easy uniform sampling of binary trees and ii) systematic ability to traverse tree space in very large or small jumps. As a proof of concept, we use P
    

