# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving LSH via Tensorized Random Projection](https://arxiv.org/abs/2402.07189) | 本文提出了CP-E2LSH和TT-E2LSH两种方法，用于改进局部敏感哈希算法LSH，在处理张量数据的欧几里得距离和余弦相似度时能够提供更快和更空间有效的结果。 |

# 详细

[^1]: 通过张量化随机投影改进局部敏感哈希LSH

    Improving LSH via Tensorized Random Projection

    [https://arxiv.org/abs/2402.07189](https://arxiv.org/abs/2402.07189)

    本文提出了CP-E2LSH和TT-E2LSH两种方法，用于改进局部敏感哈希算法LSH，在处理张量数据的欧几里得距离和余弦相似度时能够提供更快和更空间有效的结果。

    

    局部敏感哈希(LSH)是数据科学家用于近似最近邻搜索问题的基本算法工具，已在许多大规模数据处理应用中广泛使用，如近似重复检测、最近邻搜索、聚类等。在本文中，我们旨在提出更快和空间更有效的局部敏感哈希函数，用于张量数据的欧几里得距离和余弦相似度。通常，对于张量数据获得LSH的朴素方法涉及将张量重塑为向量，然后应用现有的向量数据LSH方法(E2LSH和SRP)。然而，对于高阶张量，这种方法变得不切实际，因为重塑向量的大小在张量的阶数中呈指数增长。因此，LSH参数的大小呈指数增加。为解决这个问题，我们提出了两种欧几里得距离和余弦相似度的LSH方法，分别是CP-E2LSH和TT-E2LSH。

    Locality sensitive hashing (LSH) is a fundamental algorithmic toolkit used by data scientists for approximate nearest neighbour search problems that have been used extensively in many large scale data processing applications such as near duplicate detection, nearest neighbour search, clustering, etc. In this work, we aim to propose faster and space efficient locality sensitive hash functions for Euclidean distance and cosine similarity for tensor data. Typically, the naive approach for obtaining LSH for tensor data involves first reshaping the tensor into vectors, followed by applying existing LSH methods for vector data $E2LSH$ and $SRP$. However, this approach becomes impractical for higher order tensors because the size of the reshaped vector becomes exponential in the order of the tensor. Consequently, the size of LSH parameters increases exponentially. To address this problem, we suggest two methods for LSH for Euclidean distance and cosine similarity, namely $CP-E2LSH$, $TT-E2LSH
    

