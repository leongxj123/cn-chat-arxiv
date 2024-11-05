# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep decomposition method for the limited aperture inverse obstacle scattering problem](https://arxiv.org/abs/2403.19470) | 提出了一种用于有限孔径逆障碍散射问题的深度分解方法，通过向神经网络架构提供与散射模型相关联的物理运算符，实现深度学习在逆问题上工作，并避免扭曲解决方案。 |
| [^2] | [Spectral Clustering via Orthogonalization-Free Methods.](http://arxiv.org/abs/2305.10356) | 本文提出了四种无正交化方法作为谱聚类降维，不需要昂贵的特征值估计，在聚类质量和计算成本方面均优于已有方法，适合于并行计算。 |

# 详细

[^1]: 有限孔径逆障碍散射问题的深度分解方法

    Deep decomposition method for the limited aperture inverse obstacle scattering problem

    [https://arxiv.org/abs/2403.19470](https://arxiv.org/abs/2403.19470)

    提出了一种用于有限孔径逆障碍散射问题的深度分解方法，通过向神经网络架构提供与散射模型相关联的物理运算符，实现深度学习在逆问题上工作，并避免扭曲解决方案。

    

    在本文中，我们考虑了一种针对有限孔径逆障碍散射问题的深度学习方法。传统深度学习仅依赖数据是众所周知的，当只有间接观测数据和一个物理模型可用时，这可能限制其在逆问题上的性能。在面对这些局限性时，一个基本问题出现了：是否可能使深度学习能够在没有标记数据的情况下处理逆问题，并且了解它正在学习的内容？本文提出了一个用于这些目的的深度分解方法（DDM），它不需要地面真实标签。它通过向神经网络架构提供与散射模型相关联的物理运算符来实现这一点。此外，DDM中还实现了一种基于深度学习的数据完整性方案，以防止扭曲有限孔径数据的逆问题的解决方案。此外，除了解决i

    arXiv:2403.19470v1 Announce Type: cross  Abstract: In this paper, we consider a deep learning approach to the limited aperture inverse obstacle scattering problem. It is well known that traditional deep learning relies solely on data, which may limit its performance for the inverse problem when only indirect observation data and a physical model are available. A fundamental question arises in light of these limitations: is it possible to enable deep learning to work on inverse problems without labeled data and to be aware of what it is learning? This work proposes a deep decomposition method (DDM) for such purposes, which does not require ground truth labels. It accomplishes this by providing physical operators associated with the scattering model to the neural network architecture. Additionally, a deep learning based data completion scheme is implemented in DDM to prevent distorting the solution of the inverse problem for limited aperture data. Furthermore, apart from addressing the i
    
[^2]: 通过无正交化方法的谱聚类

    Spectral Clustering via Orthogonalization-Free Methods. (arXiv:2305.10356v1 [eess.SP])

    [http://arxiv.org/abs/2305.10356](http://arxiv.org/abs/2305.10356)

    本文提出了四种无正交化方法作为谱聚类降维，不需要昂贵的特征值估计，在聚类质量和计算成本方面均优于已有方法，适合于并行计算。

    

    在谱聚类的降维中，通常使用图信号滤波器需要昂贵的特征值估计。我们在最优化设置中分析了滤波器并提出使用四种无正交化方法作为谱聚类中的降维。所提出的方法不利用任何正交化方法，在并行计算环境中不可伸缩。我们的方法在理论上构造了足够的特征空间，最多是规范化拉普拉斯矩阵特征空间的加权改变。我们在数值上假设所提出的方法与利用精确特征值但需要昂贵特征值估计的理想图信号滤波器在聚类质量上等效。数值结果表明，所提出的方法在聚类质量和计算成本方面优于基于幂迭代的方法和图信号滤波器。与基于幂迭代的方法不同，我们的方法可以轻松并行化。

    Graph Signal Filter used as dimensionality reduction in spectral clustering usually requires expensive eigenvalue estimation. We analyze the filter in an optimization setting and propose to use four orthogonalization-free methods by optimizing objective functions as dimensionality reduction in spectral clustering. The proposed methods do not utilize any orthogonalization, which is known as not well scalable in a parallel computing environment. Our methods theoretically construct adequate feature space, which is, at most, a weighted alteration to the eigenspace of a normalized Laplacian matrix. We numerically hypothesize that the proposed methods are equivalent in clustering quality to the ideal Graph Signal Filter, which exploits the exact eigenvalue needed without expensive eigenvalue estimation. Numerical results show that the proposed methods outperform Power Iteration-based methods and Graph Signal Filter in clustering quality and computation cost. Unlike Power Iteration-based meth
    

