# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Factor Fitting, Rank Allocation, and Partitioning in Multilevel Low Rank Matrices.](http://arxiv.org/abs/2310.19214) | 本文研究了多级低秩矩阵中的因子拟合、秩分配和分割问题，提出了相应的解决方法，并开发了一个开源软件包。 |

# 详细

[^1]: 在多级低秩矩阵中进行因子拟合、秩分配和分割

    Factor Fitting, Rank Allocation, and Partitioning in Multilevel Low Rank Matrices. (arXiv:2310.19214v1 [stat.ML])

    [http://arxiv.org/abs/2310.19214](http://arxiv.org/abs/2310.19214)

    本文研究了多级低秩矩阵中的因子拟合、秩分配和分割问题，提出了相应的解决方法，并开发了一个开源软件包。

    

    我们考虑多级低秩（MLR）矩阵，定义为一系列矩阵的行和列的排列，每个矩阵都是前一个矩阵的块对角修正，所有块以因子形式给出低秩矩阵。MLR矩阵扩展了低秩矩阵的概念，但它们共享许多特性，例如所需总存储空间和矩阵向量乘法的复杂度。我们解决了用Frobenius范数拟合给定矩阵到MLR矩阵的三个问题。第一个问题是因子拟合，通过调整MLR矩阵的因子来解决。第二个问题是秩分配，在每个级别中选择块的秩，满足总秩的给定值，以保持MLR矩阵所需的总存储空间。最后一个问题是选择行和列的层次分割，以及秩和因子。本文附带了一个开源软件包，实现了所提出的方法。

    We consider multilevel low rank (MLR) matrices, defined as a row and column permutation of a sum of matrices, each one a block diagonal refinement of the previous one, with all blocks low rank given in factored form. MLR matrices extend low rank matrices but share many of their properties, such as the total storage required and complexity of matrix-vector multiplication. We address three problems that arise in fitting a given matrix by an MLR matrix in the Frobenius norm. The first problem is factor fitting, where we adjust the factors of the MLR matrix. The second is rank allocation, where we choose the ranks of the blocks in each level, subject to the total rank having a given value, which preserves the total storage needed for the MLR matrix. The final problem is to choose the hierarchical partition of rows and columns, along with the ranks and factors. This paper is accompanied by an open source package that implements the proposed methods.
    

