# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Convex optimization over a probability simplex.](http://arxiv.org/abs/2305.09046) | 这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。 |

# 详细

[^1]: 概率单纯形上的凸优化

    Convex optimization over a probability simplex. (arXiv:2305.09046v1 [math.OC])

    [http://arxiv.org/abs/2305.09046](http://arxiv.org/abs/2305.09046)

    这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。

    

    我们提出了一种新的迭代方案——柯西单纯形来优化凸问题，使其满足概率单纯形上的限制条件，即$w\in\mathbb{R}^n$中$\sum_i w_i=1$，$w_i\geq0$。我们将单纯形映射到单位球的正四面体，通过梯度下降获得隐变量的解，并将结果映射回原始变量。该方法适用于高维问题，每次迭代由简单的操作组成，且针对凸函数证明了收敛速度为${O}(1/T)$。同时本文关注了信息理论（如交叉熵和KL散度）的应用。

    We propose a new iteration scheme, the Cauchy-Simplex, to optimize convex problems over the probability simplex $\{w\in\mathbb{R}^n\ |\ \sum_i w_i=1\ \textrm{and}\ w_i\geq0\}$. Other works have taken steps to enforce positivity or unit normalization automatically but never simultaneously within a unified setting. This paper presents a natural framework for manifestly requiring the probability condition. Specifically, we map the simplex to the positive quadrant of a unit sphere, envisage gradient descent in latent variables, and map the result back in a way that only depends on the simplex variable. Moreover, proving rigorous convergence results in this formulation leads inherently to tools from information theory (e.g. cross entropy and KL divergence). Each iteration of the Cauchy-Simplex consists of simple operations, making it well-suited for high-dimensional problems. We prove that it has a convergence rate of ${O}(1/T)$ for convex functions, and numerical experiments of projection 
    

