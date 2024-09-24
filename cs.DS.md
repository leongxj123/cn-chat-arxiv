# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Connectivity Oracles for Predictable Vertex Failures](https://arxiv.org/abs/2312.08489) | 论文研究了在预测算法范式下设计支持顶点失败的连通性预测器的问题，并提出了一种数据结构，能够以预处理时间和查询时间的多项式关系来处理失败顶点集合。 |
| [^2] | [Noise Stability Optimization for Flat Minima with Optimal Convergence Rates.](http://arxiv.org/abs/2306.08553) | 本文提出了一个SGD-like算法，注入随机噪声并利用分布对称性来减少方差，以寻找具有低海森矩阵迹的平坦极小值，同时提供了收敛速率分析。 |

# 详细

[^1]: 预测顶点失败的连通性预测器

    Connectivity Oracles for Predictable Vertex Failures

    [https://arxiv.org/abs/2312.08489](https://arxiv.org/abs/2312.08489)

    论文研究了在预测算法范式下设计支持顶点失败的连通性预测器的问题，并提出了一种数据结构，能够以预处理时间和查询时间的多项式关系来处理失败顶点集合。

    

    设计支持顶点失败的连通性预测器是针对无向图的基本数据结构问题之一。已有的研究在查询时间方面已经有了很好的理解：以前的作品[Duan-Pettie STOC'10; Long-Saranurak FOCS'22]实现了与失败顶点数量成线性关系的查询时间，并且在需要多项式时间的预处理和多项式时间的更新的条件下是有条件最优的。我们在预测算法的范式下重新审视了这个问题：我们问，如果可以预测到失败顶点集合，查询时间是否可以提高。更具体地说，我们设计了一个数据结构，给定一个图G=(V,E)和一个预测会失败的顶点集合\widehat{D} \subseteq V（其中d=|\widehat{D}|），将其预处理时间为$\tilde{O}(d|E|)$，然后可以接收一个更新，该更新以对称差分形式给出。

    arXiv:2312.08489v2 Announce Type: replace-cross  Abstract: The problem of designing connectivity oracles supporting vertex failures is one of the basic data structures problems for undirected graphs. It is already well understood: previous works [Duan--Pettie STOC'10; Long--Saranurak FOCS'22] achieve query time linear in the number of failed vertices, and it is conditionally optimal as long as we require preprocessing time polynomial in the size of the graph and update time polynomial in the number of failed vertices.   We revisit this problem in the paradigm of algorithms with predictions: we ask if the query time can be improved if the set of failed vertices can be predicted beforehand up to a small number of errors. More specifically, we design a data structure that, given a graph $G=(V,E)$ and a set of vertices predicted to fail $\widehat{D} \subseteq V$ of size $d=|\widehat{D}|$, preprocesses it in time $\tilde{O}(d|E|)$ and then can receive an update given as the symmetric differ
    
[^2]: 噪声稳定优化对于具有最优收敛率的平坦极小值的影响

    Noise Stability Optimization for Flat Minima with Optimal Convergence Rates. (arXiv:2306.08553v1 [cs.LG])

    [http://arxiv.org/abs/2306.08553](http://arxiv.org/abs/2306.08553)

    本文提出了一个SGD-like算法，注入随机噪声并利用分布对称性来减少方差，以寻找具有低海森矩阵迹的平坦极小值，同时提供了收敛速率分析。

    

    本文研究通过加入加权扰动来找到平坦的极小值。给定一个非凸函数$f:\mathbb{R}^d\rightarrow \mathbb{R}$和一个$d$维分布$\mathcal{P}$，我们扰动$f$的权重，并定义$F(W)=\mathbb{E}[f({W+U})]$，其中$U$是一个从$\mathcal{P}$中随机抽取的样本。这个过程通过$f$的海森矩阵的迹来诱导正则化，以适应于小的、各向同性的高斯扰动。因此，加权扰动的函数偏向于带有低海森矩阵迹的极小值。本文提出了一种类似于SGD的算法，在计算梯度之前注入随机噪声，同时利用$\mathcal{P}$的对称性来减少方差。我们还提供了严格的分析，证明了...

    We consider finding flat, local minimizers by adding average weight perturbations. Given a nonconvex function $f: \mathbb{R}^d \rightarrow \mathbb{R}$ and a $d$-dimensional distribution $\mathcal{P}$ which is symmetric at zero, we perturb the weight of $f$ and define $F(W) = \mathbb{E}[f({W + U})]$, where $U$ is a random sample from $\mathcal{P}$. This injection induces regularization through the Hessian trace of $f$ for small, isotropic Gaussian perturbations. Thus, the weight-perturbed function biases to minimizers with low Hessian trace. Several prior works have studied settings related to this weight-perturbed function by designing algorithms to improve generalization. Still, convergence rates are not known for finding minima under the average perturbations of the function $F$. This paper considers an SGD-like algorithm that injects random noise before computing gradients while leveraging the symmetry of $\mathcal{P}$ to reduce variance. We then provide a rigorous analysis, showing
    

