# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Connected Hidden Neurons (CHNNet): An Artificial Neural Network for Rapid Convergence.](http://arxiv.org/abs/2305.10468) | 该论文提出了一个更为强大的人工神经网络模型，该模型中同一隐藏层中的隐藏神经元相互连接，可以学习复杂模式并加速收敛速度。 |
| [^2] | [NervePool: A Simplicial Pooling Layer.](http://arxiv.org/abs/2305.06315) | 单纯复形池化层NervePool在池化图结构数据时，基于顶点分区生成单纯复形的分层表示，可以灵活地建模更高阶的关系，同时缩小高维单纯形，实现降采样，减少计算成本和减少过拟合。 |

# 详细

[^1]: 连接隐藏神经元（CHNNet）：一种快速收敛的人工神经网络

    Connected Hidden Neurons (CHNNet): An Artificial Neural Network for Rapid Convergence. (arXiv:2305.10468v1 [cs.NE])

    [http://arxiv.org/abs/2305.10468](http://arxiv.org/abs/2305.10468)

    该论文提出了一个更为强大的人工神经网络模型，该模型中同一隐藏层中的隐藏神经元相互连接，可以学习复杂模式并加速收敛速度。

    

    人工神经网络的核心目的是模仿生物神经网络的功能。然而，与生物神经网络不同，传统的人工神经网络通常是按层次结构化的，这可能会妨碍神经元之间的信息流动，因为同一层中的神经元之间没有连接。因此，我们提出了一种更为强大的人工神经网络模型，其中同一隐藏层中的隐藏神经元是互相连接的，使得神经元能够学习复杂的模式并加速收敛速度。通过在浅层和深层网络中将我们提出的模型作为完全连接的层进行实验研究，我们证明这个模型可以显著提高收敛速率。

    The core purpose of developing artificial neural networks was to mimic the functionalities of biological neural networks. However, unlike biological neural networks, traditional artificial neural networks are often structured hierarchically, which can impede the flow of information between neurons as the neurons in the same layer have no connections between them. Hence, we propose a more robust model of artificial neural networks where the hidden neurons, residing in the same hidden layer, are interconnected, enabling the neurons to learn complex patterns and speeding up the convergence rate. With the experimental study of our proposed model as fully connected layers in shallow and deep networks, we demonstrate that the model results in a significant increase in convergence rate.
    
[^2]: NervePool: 一个单纯复形池化层

    NervePool: A Simplicial Pooling Layer. (arXiv:2305.06315v1 [cs.CG])

    [http://arxiv.org/abs/2305.06315](http://arxiv.org/abs/2305.06315)

    单纯复形池化层NervePool在池化图结构数据时，基于顶点分区生成单纯复形的分层表示，可以灵活地建模更高阶的关系，同时缩小高维单纯形，实现降采样，减少计算成本和减少过拟合。

    

    对于图结构数据的深度学习问题，池化层对于降采样、减少计算成本和减少过拟合都很重要。我们提出了一个池化层，NervePool，适用于单纯复形结构的数据，这种结构是图的推广，包括比顶点和边更高维度的单纯形；这种结构可以更灵活地建模更高阶的关系。所提出的单纯复合缩小方案基于顶点的分区构建，这使得我们可以生成单纯复形的分层表示，以一种学习的方式折叠信息。NervePool建立在学习的顶点群集分配的基础上，并以一种确定性的方式扩展到高维单纯形的缩小。虽然在实践中，池化操作是通过一系列矩阵运算来计算的，但是其拓扑动机是一个基于单纯形星星的并集和神经复合体的集合构造。

    For deep learning problems on graph-structured data, pooling layers are important for down sampling, reducing computational cost, and to minimize overfitting. We define a pooling layer, NervePool, for data structured as simplicial complexes, which are generalizations of graphs that include higher-dimensional simplices beyond vertices and edges; this structure allows for greater flexibility in modeling higher-order relationships. The proposed simplicial coarsening scheme is built upon partitions of vertices, which allow us to generate hierarchical representations of simplicial complexes, collapsing information in a learned fashion. NervePool builds on the learned vertex cluster assignments and extends to coarsening of higher dimensional simplices in a deterministic fashion. While in practice, the pooling operations are computed via a series of matrix operations, the topological motivation is a set-theoretic construction based on unions of stars of simplices and the nerve complex
    

