# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Clustering in Data Streams.](http://arxiv.org/abs/2307.07449) | 本研究提出了首个针对$k$-means和$k$-median聚类的差分隐私流算法，在流模型中实现对数据隐私的保护，并使用尽可能少的空间。 |

# 详细

[^1]: 数据流中的差分隐私聚类

    Differentially Private Clustering in Data Streams. (arXiv:2307.07449v1 [cs.DS])

    [http://arxiv.org/abs/2307.07449](http://arxiv.org/abs/2307.07449)

    本研究提出了首个针对$k$-means和$k$-median聚类的差分隐私流算法，在流模型中实现对数据隐私的保护，并使用尽可能少的空间。

    

    流模型是处理大规模现代数据分析的一种常见方法。在流模型中，数据点依次流入，算法只能对数据流进行一次遍历，目标是在使用尽可能少的空间的同时，在流中进行一些分析。聚类问题是基本的无监督机器学习原语，过去已经对流聚类算法进行了广泛的研究。然而，在许多实际应用中，数据隐私已成为一个核心关注点，非私有聚类算法在许多场景下不适用。在这项工作中，我们提供了第一个针对$k$-means和$k$-median聚类的差分私有流算法，该算法在长度最多为$T$的流上使用$poly(k,d,\log(T))$的空间来实现一个“常数”。

    The streaming model is an abstraction of computing over massive data streams, which is a popular way of dealing with large-scale modern data analysis. In this model, there is a stream of data points, one after the other. A streaming algorithm is only allowed one pass over the data stream, and the goal is to perform some analysis during the stream while using as small space as possible.  Clustering problems (such as $k$-means and $k$-median) are fundamental unsupervised machine learning primitives, and streaming clustering algorithms have been extensively studied in the past. However, since data privacy becomes a central concern in many real-world applications, non-private clustering algorithms are not applicable in many scenarios.  In this work, we provide the first differentially private streaming algorithms for $k$-means and $k$-median clustering of $d$-dimensional Euclidean data points over a stream with length at most $T$ using $poly(k,d,\log(T))$ space to achieve a {\it constant} 
    

