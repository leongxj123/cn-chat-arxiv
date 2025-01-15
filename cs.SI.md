# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inhomogeneous graph trend filtering via a l2,0 cardinality penalty.](http://arxiv.org/abs/2304.05223) | 本文提出了一种基于L2，0基数惩罚的图趋势过滤（GTF）模型，可同时进行k-means聚类和基于图的最小割，以估计在节点之间具有不均匀平滑水平的分段平滑图信号，并在降噪、支持恢复和半监督分类任务上表现更好，比现有方法更高效地处理大型数据集。 |

# 详细

[^1]: 基于L2，0基数惩罚的不均匀图趋势过滤。

    Inhomogeneous graph trend filtering via a l2,0 cardinality penalty. (arXiv:2304.05223v1 [cs.LG])

    [http://arxiv.org/abs/2304.05223](http://arxiv.org/abs/2304.05223)

    本文提出了一种基于L2，0基数惩罚的图趋势过滤（GTF）模型，可同时进行k-means聚类和基于图的最小割，以估计在节点之间具有不均匀平滑水平的分段平滑图信号，并在降噪、支持恢复和半监督分类任务上表现更好，比现有方法更高效地处理大型数据集。

    

    我们研究了在图上估计分段平滑信号的方法，并提出了一种$\ell_{2,0}$-范数惩罚图趋势过滤（GTF）模型，以估计在节点之间具有不均匀平滑水平的分段平滑图信号。我们证明了所提出的GTF模型同时是基于节点上的信号的k-means聚类和基于图的最小割，其中聚类和割共享相同的分配矩阵。我们提出了两种方法来解决所提出的GTF模型：一种是基于谱分解的方法，另一种是基于模拟退火的方法。在合成和现实数据集的实验中，我们展示了所提出的GTF模型在降噪、支持恢复和半监督分类任务上表现更好，且比现有方法更高效地解决了大型数据集的问题。

    We study estimation of piecewise smooth signals over a graph. We propose a $\ell_{2,0}$-norm penalized Graph Trend Filtering (GTF) model to estimate piecewise smooth graph signals that exhibits inhomogeneous levels of smoothness across the nodes. We prove that the proposed GTF model is simultaneously a k-means clustering on the signal over the nodes and a minimum graph cut on the edges of the graph, where the clustering and the cut share the same assignment matrix. We propose two methods to solve the proposed GTF model: a spectral decomposition method and a method based on simulated annealing. In the experiment on synthetic and real-world datasets, we show that the proposed GTF model has a better performances compared with existing approaches on the tasks of denoising, support recovery and semi-supervised classification. We also show that the proposed GTF model can be solved more efficiently than existing models for the dataset with a large edge set.
    

