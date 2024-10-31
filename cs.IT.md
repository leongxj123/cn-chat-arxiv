# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Solving Quadratic Systems with Full-Rank Matrices Using Sparse or Generative Priors.](http://arxiv.org/abs/2309.09032) | 本论文提出了一个方法，通过使用稀疏或生成的先验知识，解决了从全秩矩阵的二次系统中恢复信号的问题。其中，通过引入阈值Wirtinger流算法（TWF）来处理稀疏信号，并使用谱初始化和阈值梯度下降方法，在高维情况下实现了较小的测量数量。 |

# 详细

[^1]: 使用稀疏或生成的先验解决全秩矩阵的二次系统

    Solving Quadratic Systems with Full-Rank Matrices Using Sparse or Generative Priors. (arXiv:2309.09032v1 [cs.IT])

    [http://arxiv.org/abs/2309.09032](http://arxiv.org/abs/2309.09032)

    本论文提出了一个方法，通过使用稀疏或生成的先验知识，解决了从全秩矩阵的二次系统中恢复信号的问题。其中，通过引入阈值Wirtinger流算法（TWF）来处理稀疏信号，并使用谱初始化和阈值梯度下降方法，在高维情况下实现了较小的测量数量。

    

    从具有全秩矩阵的二次系统中恢复信号x在应用中经常出现，比如未分配的距离几何和亚波长成像。本文通过引入对x的先验知识，针对高维情况（m << n），使用独立同分布的标准高斯矩阵解决了该问题。首先，考虑k-稀疏的x，引入了TWF算法，该算法不需要稀疏水平k。TWF包括两个步骤：谱初始化，当m = O(k^2log n)时，确定了一个距离x足够近的点（可能会有符号翻转），以及具有很好初始化的阈值梯度下降，该下降产生了一个线性收敛到x的序列，用m = O(klog n)个测量。

    The problem of recovering a signal $\boldsymbol{x} \in \mathbb{R}^n$ from a quadratic system $\{y_i=\boldsymbol{x}^\top\boldsymbol{A}_i\boldsymbol{x},\ i=1,\ldots,m\}$ with full-rank matrices $\boldsymbol{A}_i$ frequently arises in applications such as unassigned distance geometry and sub-wavelength imaging. With i.i.d. standard Gaussian matrices $\boldsymbol{A}_i$, this paper addresses the high-dimensional case where $m\ll n$ by incorporating prior knowledge of $\boldsymbol{x}$. First, we consider a $k$-sparse $\boldsymbol{x}$ and introduce the thresholded Wirtinger flow (TWF) algorithm that does not require the sparsity level $k$. TWF comprises two steps: the spectral initialization that identifies a point sufficiently close to $\boldsymbol{x}$ (up to a sign flip) when $m=O(k^2\log n)$, and the thresholded gradient descent (with a good initialization) that produces a sequence linearly converging to $\boldsymbol{x}$ with $m=O(k\log n)$ measurements. Second, we explore the generative p
    

