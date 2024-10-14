# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Kernel Metric Learning for Clustering Mixed-type Data.](http://arxiv.org/abs/2306.01890) | 提出了一种使用混合核测量不相似性的度量方法，并通过交叉验证找到最佳核带宽。该方法可为现有的基于距离的聚类算法提高聚类准确度，适用于包含混合类型数据的模拟和实际数据集。 |

# 详细

[^1]: 混合类型数据的核度量学习

    Kernel Metric Learning for Clustering Mixed-type Data. (arXiv:2306.01890v1 [cs.LG])

    [http://arxiv.org/abs/2306.01890](http://arxiv.org/abs/2306.01890)

    提出了一种使用混合核测量不相似性的度量方法，并通过交叉验证找到最佳核带宽。该方法可为现有的基于距离的聚类算法提高聚类准确度，适用于包含混合类型数据的模拟和实际数据集。

    

    基于距离的聚类和分类广泛应用于各个领域，以将混合数值和分类数据分组。预定义的距离测量用于根据它们的不相似性来聚类数据点。虽然存在许多适用于具有纯数字属性和几个有序和无序分类指标的数据的基于距离的度量方法，但混合型数据的最佳距离是一个尚未解决的问题。许多度量将数字属性转换为分类属性或反之亦然。他们将数据点处理为单个属性类型，或者分别计算每个属性之间的距离并将它们相加。我们提出了一种度量方法，使用混合核测量不相似性，并进行交叉验证来寻找最佳核带宽。我们的方法对包含纯连续，分类和混合类型数据的模拟和实际数据集应用于现有的基于距离的聚类算法时，提高了聚类准确度。

    Distance-based clustering and classification are widely used in various fields to group mixed numeric and categorical data. A predefined distance measurement is used to cluster data points based on their dissimilarity. While there exist numerous distance-based measures for data with pure numerical attributes and several ordered and unordered categorical metrics, an optimal distance for mixed-type data is an open problem. Many metrics convert numerical attributes to categorical ones or vice versa. They handle the data points as a single attribute type or calculate a distance between each attribute separately and add them up. We propose a metric that uses mixed kernels to measure dissimilarity, with cross-validated optimal kernel bandwidths. Our approach improves clustering accuracy when utilized for existing distance-based clustering algorithms on simulated and real-world datasets containing pure continuous, categorical, and mixed-type data.
    

