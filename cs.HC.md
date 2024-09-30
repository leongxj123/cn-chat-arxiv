# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cluster Exploration using Informative Manifold Projections.](http://arxiv.org/abs/2309.14857) | 该论文提出了一种新颖的方法来生成信息丰富的嵌入，以揭示高维数据中的聚类结构。通过线性组合对比PCA和峰度投影追踪两个目标，该方法能够排除先验信息相关的结构并实现有意义的数据分离。 |

# 详细

[^1]: 使用信息流形投影进行聚类探索

    Cluster Exploration using Informative Manifold Projections. (arXiv:2309.14857v1 [cs.LG])

    [http://arxiv.org/abs/2309.14857](http://arxiv.org/abs/2309.14857)

    该论文提出了一种新颖的方法来生成信息丰富的嵌入，以揭示高维数据中的聚类结构。通过线性组合对比PCA和峰度投影追踪两个目标，该方法能够排除先验信息相关的结构并实现有意义的数据分离。

    

    降维是可视化探索高维数据和发现其在二维或三维空间中的聚类结构的关键工具之一。现有文献中的大部分降维方法并未考虑实践者可能对所考虑数据集的任何先验知识。我们提出了一种新颖的方法来生成信息丰富的嵌入，不仅排除与先验知识相关的结构，而且旨在揭示任何剩余的潜在结构。为了实现这一目标，我们采用了两个目标的线性组合：首先是对比PCA，能够消除与先验信息相关的结构，其次是峰度投影追踪，可以确保在得到的嵌入中有意义的数据分离。我们将这个任务定义为流形优化问题，并在考虑三种不同类型的先验知识的各种数据集上进行了实证验证。

    Dimensionality reduction (DR) is one of the key tools for the visual exploration of high-dimensional data and uncovering its cluster structure in two- or three-dimensional spaces. The vast majority of DR methods in the literature do not take into account any prior knowledge a practitioner may have regarding the dataset under consideration. We propose a novel method to generate informative embeddings which not only factor out the structure associated with different kinds of prior knowledge but also aim to reveal any remaining underlying structure. To achieve this, we employ a linear combination of two objectives: firstly, contrastive PCA that discounts the structure associated with the prior information, and secondly, kurtosis projection pursuit which ensures meaningful data separation in the obtained embeddings. We formulate this task as a manifold optimization problem and validate it empirically across a variety of datasets considering three distinct types of prior knowledge. Lastly, 
    

