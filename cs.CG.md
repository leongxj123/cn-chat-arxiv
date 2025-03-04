# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Topological Point Cloud Clustering.](http://arxiv.org/abs/2303.16716) | 本文提出一种新的基于拓扑的点聚类方法，该方法可以利用拓扑特征描述点云内的数据点，相较于传统图模型方法更具有健壮性和效率。 |

# 详细

[^1]: 基于拓扑的点云聚类方法

    Topological Point Cloud Clustering. (arXiv:2303.16716v1 [math.AT])

    [http://arxiv.org/abs/2303.16716](http://arxiv.org/abs/2303.16716)

    本文提出一种新的基于拓扑的点聚类方法，该方法可以利用拓扑特征描述点云内的数据点，相较于传统图模型方法更具有健壮性和效率。

    

    本文提出了一种叫做拓扑点云聚类（TPCC）的新方法，它基于点云对于全局拓扑特征的贡献来聚类点。TPCC从谱聚类和拓扑数据分析中综合了有利的特征，基于考虑与所考虑的点云相关联的一个单形复合体的谱特性。由于它基于考虑稀疏特征向量计算，TPCC同样容易解释和实现，就像谱聚类一样。然而，通过不仅关注与从点云数据创建的图相关联的单个矩阵，而是关注与恰当构造的单形复合体相关联的整个Hodge-Laplacian的一整套矩阵，我们可以利用更丰富的拓扑特征来描述点云内的数据点，并受益于拓扑技术相对于噪声的相对健壮性。我们在合成和真实世界数据上测试了TPCC的性能。

    We present Topological Point Cloud Clustering (TPCC), a new method to cluster points in an arbitrary point cloud based on their contribution to global topological features. TPCC synthesizes desirable features from spectral clustering and topological data analysis and is based on considering the spectral properties of a simplicial complex associated to the considered point cloud. As it is based on considering sparse eigenvector computations, TPCC is similarly easy to interpret and implement as spectral clustering. However, by focusing not just on a single matrix associated to a graph created from the point cloud data, but on a whole set of Hodge-Laplacians associated to an appropriately constructed simplicial complex, we can leverage a far richer set of topological features to characterize the data points within the point cloud and benefit from the relative robustness of topological techniques against noise. We test the performance of TPCC on both synthetic and real-world data and compa
    

