# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient approximation of Earth Mover's Distance Based on Nearest Neighbor Search.](http://arxiv.org/abs/2401.07378) | 本文提出了一种基于最近邻搜索的新方法NNS-EMD来逼近地球移动距离（EMD），以实现高精度、低时间复杂度和高内存效率。该方法通过减少数据点的比较数量和并行处理提供了高效的近似计算，并通过在GPU上进行向量化加速，特别适用于大型数据集。 |

# 详细

[^1]: 基于最近邻搜索的地球移动距离的高效逼近方法

    Efficient approximation of Earth Mover's Distance Based on Nearest Neighbor Search. (arXiv:2401.07378v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2401.07378](http://arxiv.org/abs/2401.07378)

    本文提出了一种基于最近邻搜索的新方法NNS-EMD来逼近地球移动距离（EMD），以实现高精度、低时间复杂度和高内存效率。该方法通过减少数据点的比较数量和并行处理提供了高效的近似计算，并通过在GPU上进行向量化加速，特别适用于大型数据集。

    

    地球移动距离（EMD）是计算机视觉和其他应用领域中的两个分布之间的重要相似度度量。然而，其精确计算的计算和内存消耗较大，限制了其在大规模问题上的可扩展性和适用性。为了降低计算成本，提出了各种近似EMD算法，但它们精度较低，可能需要额外的内存使用或手动参数调整。在本文中，我们提出了一种新颖的方法，称为NNS-EMD，使用最近邻搜索（NNS）来逼近EMD，以实现高精度、低时间复杂度和高内存效率。NNS操作减少了每次NNS迭代中所比较的数据点的数量，并提供了并行处理的机会。我们还通过在GPU上进行向量化来加速NNS-EMD，这对于大型数据集尤为有益。我们将NNS-EMD与精确EMD和最先进的近似EMD算法进行了比较。

    Earth Mover's Distance (EMD) is an important similarity measure between two distributions, used in computer vision and many other application domains. However, its exact calculation is computationally and memory intensive, which hinders its scalability and applicability for large-scale problems. Various approximate EMD algorithms have been proposed to reduce computational costs, but they suffer lower accuracy and may require additional memory usage or manual parameter tuning. In this paper, we present a novel approach, NNS-EMD, to approximate EMD using Nearest Neighbor Search (NNS), in order to achieve high accuracy, low time complexity, and high memory efficiency. The NNS operation reduces the number of data points compared in each NNS iteration and offers opportunities for parallel processing. We further accelerate NNS-EMD via vectorization on GPU, which is especially beneficial for large datasets. We compare NNS-EMD with both the exact EMD and state-of-the-art approximate EMD algori
    

