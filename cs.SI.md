# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Community detection by spectral methods in multi-layer networks](https://arxiv.org/abs/2403.12540) | 改进的谱聚类算法在多层网络中实现了更好的社区检测性能，并证明多层网络对社区检测有优势。 |
| [^2] | [Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs](https://arxiv.org/abs/2403.10231) | 提出了一种在大规模知识图谱上进行高效和自适应预测的一次性子图链接预测方法，通过将预测过程分解为从查询中提取一个子图并在该单个、查询相关子图上进行预测的两个步骤，利用非参数化和计算高效的启发式方法来提高效率。 |

# 详细

[^1]: 多层网络中基于谱方法的社区检测

    Community detection by spectral methods in multi-layer networks

    [https://arxiv.org/abs/2403.12540](https://arxiv.org/abs/2403.12540)

    改进的谱聚类算法在多层网络中实现了更好的社区检测性能，并证明多层网络对社区检测有优势。

    

    多层网络中的社区检测是网络分析中一个关键问题。本文分析了两种谱聚类算法在多层度校正随机块模型（MLDCSBM）框架下进行社区检测的性能。一种算法基于邻接矩阵的和，另一种利用了去偏和的平方邻接矩阵的和。我们在网络规模和/或层数增加时建立了这些方法在MLDCSBM下进行社区检测的一致性结果。我们的定理展示了利用多层进行社区检测的优势。此外，我们的分析表明，利用去偏和的平方邻接矩阵的谱聚类通常优于利用邻接矩阵的谱聚类。数值模拟证实了我们的算法，采用了去偏和的平方邻接矩阵。

    arXiv:2403.12540v1 Announce Type: cross  Abstract: Community detection in multi-layer networks is a crucial problem in network analysis. In this paper, we analyze the performance of two spectral clustering algorithms for community detection within the multi-layer degree-corrected stochastic block model (MLDCSBM) framework. One algorithm is based on the sum of adjacency matrices, while the other utilizes the debiased sum of squared adjacency matrices. We establish consistency results for community detection using these methods under MLDCSBM as the size of the network and/or the number of layers increases. Our theorems demonstrate the advantages of utilizing multiple layers for community detection. Moreover, our analysis indicates that spectral clustering with the debiased sum of squared adjacency matrices is generally superior to spectral clustering with the sum of adjacency matrices. Numerical simulations confirm that our algorithm, employing the debiased sum of squared adjacency matri
    
[^2]: 少即是多：大规模知识图谱上的一次性子图推理

    Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs

    [https://arxiv.org/abs/2403.10231](https://arxiv.org/abs/2403.10231)

    提出了一种在大规模知识图谱上进行高效和自适应预测的一次性子图链接预测方法，通过将预测过程分解为从查询中提取一个子图并在该单个、查询相关子图上进行预测的两个步骤，利用非参数化和计算高效的启发式方法来提高效率。

    

    要在知识图谱（KG）上推导新的事实，链接预测器从图结构中学习，并收集局部证据以找到对给定查询的答案。然而，现有方法由于利用整个KG进行预测而存在严重的可扩展性问题，这阻碍了它们在大规模KG上的应用，并且无法直接通过常规抽样方法解决。 在这项工作中，我们提出了一次性子图链接预测以实现高效且自适应的预测。 设计原则是，预测过程不直接作用于整个KG，而是分为两个步骤，即（i）根据查询仅提取一个子图和（ii）在这个单一的、查询相关的子图上进行预测。 我们发现，非参数化和计算高效的启发式方法个性化PageRank（PPR）可以有效地识别潜在答案和支持证据。

    arXiv:2403.10231v1 Announce Type: cross  Abstract: To deduce new facts on a knowledge graph (KG), a link predictor learns from the graph structure and collects local evidence to find the answer to a given query. However, existing methods suffer from a severe scalability problem due to the utilization of the whole KG for prediction, which hinders their promise on large scale KGs and cannot be directly addressed by vanilla sampling methods. In this work, we propose the one-shot-subgraph link prediction to achieve efficient and adaptive prediction. The design principle is that, instead of directly acting on the whole KG, the prediction procedure is decoupled into two steps, i.e., (i) extracting only one subgraph according to the query and (ii) predicting on this single, query dependent subgraph. We reveal that the non-parametric and computation-efficient heuristics Personalized PageRank (PPR) can effectively identify the potential answers and supporting evidence. With efficient subgraph-b
    

