# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simultaneous Dimensionality Reduction: A Data Efficient Approach for Multimodal Representations Learning.](http://arxiv.org/abs/2310.04458) | 该论文介绍了一种数据高效的多模态表示学习方法，探索了独立降维和同时降维两种方法，并通过生成线性模型评估了其相对准确性和数据集大小要求。 |

# 详细

[^1]: 同时降维：一种数据高效的多模态表示学习方法

    Simultaneous Dimensionality Reduction: A Data Efficient Approach for Multimodal Representations Learning. (arXiv:2310.04458v1 [stat.ML])

    [http://arxiv.org/abs/2310.04458](http://arxiv.org/abs/2310.04458)

    该论文介绍了一种数据高效的多模态表示学习方法，探索了独立降维和同时降维两种方法，并通过生成线性模型评估了其相对准确性和数据集大小要求。

    

    本文探索了两种主要的降维方法：独立降维(IDR)和同时降维(SDR)。在IDR方法中，每个模态都被独立压缩，力图保留每个模态内的尽可能多的变化。相反，在SDR中，同时压缩模态以最大化减少描述之间的协变性，同时对保留单个变化的程度不太关注。典型的例子包括偏最小二乘法和典型相关分析。虽然这些降维方法是统计学的主要方法，但它们的相对精度和数据集大小要求尚不清楚。我们引入了一个生成线性模型来合成具有已知方差和协方差结构的多模态数据，以研究这些问题。我们评估了协方差的重构准确性。

    We explore two primary classes of approaches to dimensionality reduction (DR): Independent Dimensionality Reduction (IDR) and Simultaneous Dimensionality Reduction (SDR). In IDR methods, of which Principal Components Analysis is a paradigmatic example, each modality is compressed independently, striving to retain as much variation within each modality as possible. In contrast, in SDR, one simultaneously compresses the modalities to maximize the covariation between the reduced descriptions while paying less attention to how much individual variation is preserved. Paradigmatic examples include Partial Least Squares and Canonical Correlations Analysis. Even though these DR methods are a staple of statistics, their relative accuracy and data set size requirements are poorly understood. We introduce a generative linear model to synthesize multimodal data with known variance and covariance structures to examine these questions. We assess the accuracy of the reconstruction of the covariance s
    

