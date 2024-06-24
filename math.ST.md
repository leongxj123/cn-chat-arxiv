# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm](https://arxiv.org/abs/2312.08823) | 该论文提出了一种名为Metropolis-adjusted Mirror Langevin算法的方法，用于从约束空间中进行快速采样。这种算法是对Mirror Langevin算法的改进，通过添加接受-拒绝过滤器来消除渐近偏差，并具有指数优化依赖。 |

# 详细

[^1]: 使用Metropolis-adjusted Mirror Langevin算法从约束空间中快速采样

    Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm

    [https://arxiv.org/abs/2312.08823](https://arxiv.org/abs/2312.08823)

    该论文提出了一种名为Metropolis-adjusted Mirror Langevin算法的方法，用于从约束空间中进行快速采样。这种算法是对Mirror Langevin算法的改进，通过添加接受-拒绝过滤器来消除渐近偏差，并具有指数优化依赖。

    

    我们提出了一种新的方法，称为Metropolis-adjusted Mirror Langevin算法，用于从其支持是紧凸集的分布中进行近似采样。该算法在Mirror Langevin算法（Zhang et al., 2020）的单步马尔科夫链中添加了一个接受-拒绝过滤器，Mirror Langevin算法是Mirror Langevin动力学的基本离散化。由于包含了这个过滤器，我们的方法相对于目标是无偏的，而已知的Mirror Langevin算法等Mirror Langevin动力学的离散化具有渐近偏差。对于该算法，我们还给出了混合到一个相对平滑、凸性好且与自共轭镜像函数相关的约束分布所需迭代次数的上界。由于包含Metropolis-Hastings过滤器导致的马尔科夫链是可逆的，我们得到了对误差的指数优化依赖。

    We propose a new method called the Metropolis-adjusted Mirror Langevin algorithm for approximate sampling from distributions whose support is a compact and convex set. This algorithm adds an accept-reject filter to the Markov chain induced by a single step of the Mirror Langevin algorithm (Zhang et al., 2020), which is a basic discretisation of the Mirror Langevin dynamics. Due to the inclusion of this filter, our method is unbiased relative to the target, while known discretisations of the Mirror Langevin dynamics including the Mirror Langevin algorithm have an asymptotic bias. For this algorithm, we also give upper bounds for the number of iterations taken to mix to a constrained distribution whose potential is relatively smooth, convex, and Lipschitz continuous with respect to a self-concordant mirror function. As a consequence of the reversibility of the Markov chain induced by the inclusion of the Metropolis-Hastings filter, we obtain an exponentially better dependence on the erro
    

