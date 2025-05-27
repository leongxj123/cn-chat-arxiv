# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Quantum Approximation Scheme for k-Means.](http://arxiv.org/abs/2308.08167) | 这个论文提出了一个量子逼近方案，用于解决经典的k-Means聚类问题，该方案的运行时间与数据点的数量具有多对数依赖关系，并且能够在高概率下输出一个近似最优解，这是第一个具有多对数运行时间的量子算法，并且能够提供一个可证明的逼近保证。 |

# 详细

[^1]: 一个用于k-Means的量子逼近方案

    A Quantum Approximation Scheme for k-Means. (arXiv:2308.08167v1 [quant-ph])

    [http://arxiv.org/abs/2308.08167](http://arxiv.org/abs/2308.08167)

    这个论文提出了一个量子逼近方案，用于解决经典的k-Means聚类问题，该方案的运行时间与数据点的数量具有多对数依赖关系，并且能够在高概率下输出一个近似最优解，这是第一个具有多对数运行时间的量子算法，并且能够提供一个可证明的逼近保证。

    

    我们在QRAM模型中提供了一个量子逼近方案（即对于任意ε > 0, 都是 (1 + ε)-逼近），用于经典的k-Means聚类问题，其运行时间仅与数据点的数量具有多对数依赖关系。具体而言，给定一个在QRAM数据结构中存储的具有N个点的数据集V，这个量子算法的运行时间为Õ(2^(Õ(k/ε))η^2d)，并且以高概率输出一个包含k个中心的集合C，满足cost(V, C) ≤ (1+ε) · cost(V, C_OPT)。这里C_OPT表示最优的k个中心，cost(.)表示标准的k-Means代价函数（即点到最近中心的平方距离之和），而η是纵横比（即最远距离与最近距离的比值）。这是第一个具有多对数运行时间的量子算法，并且能够提供一个可证明的(1+ε)逼近保证。

    We give a quantum approximation scheme (i.e., $(1 + \varepsilon)$-approximation for every $\varepsilon > 0$) for the classical $k$-means clustering problem in the QRAM model with a running time that has only polylogarithmic dependence on the number of data points. More specifically, given a dataset $V$ with $N$ points in $\mathbb{R}^d$ stored in QRAM data structure, our quantum algorithm runs in time $\tilde{O} \left( 2^{\tilde{O}(\frac{k}{\varepsilon})} \eta^2 d\right)$ and with high probability outputs a set $C$ of $k$ centers such that $cost(V, C) \leq (1+\varepsilon) \cdot cost(V, C_{OPT})$. Here $C_{OPT}$ denotes the optimal $k$-centers, $cost(.)$ denotes the standard $k$-means cost function (i.e., the sum of the squared distance of points to the closest center), and $\eta$ is the aspect ratio (i.e., the ratio of maximum distance to minimum distance). This is the first quantum algorithm with a polylogarithmic running time that gives a provable approximation guarantee of $(1+\varep
    

