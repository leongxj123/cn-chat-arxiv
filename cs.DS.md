# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Testing Calibration in Subquadratic Time](https://arxiv.org/abs/2402.13187) | 该论文通过属性测试算法方面的研究，提出了一种基于近似线性规划的算法，可以在信息理论上最优地解决校准性测试问题（最多一个常数倍数）。 |
| [^2] | [Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm](https://arxiv.org/abs/2312.08823) | 该论文提出了一种名为Metropolis-adjusted Mirror Langevin算法的方法，用于从约束空间中进行快速采样。这种算法是对Mirror Langevin算法的改进，通过添加接受-拒绝过滤器来消除渐近偏差，并具有指数优化依赖。 |

# 详细

[^1]: 在次线性时间内测试校准性

    Testing Calibration in Subquadratic Time

    [https://arxiv.org/abs/2402.13187](https://arxiv.org/abs/2402.13187)

    该论文通过属性测试算法方面的研究，提出了一种基于近似线性规划的算法，可以在信息理论上最优地解决校准性测试问题（最多一个常数倍数）。

    

    在最近的机器学习和决策制定文献中，校准性已经成为二元预测模型输出的一个值得期望和广泛研究的统计性质。然而，测量模型校准性的算法方面仍然相对较少被探索。在论文 [BGHN23] 的启发下，该论文提出了一个严格的框架来衡量到校准性的距离，我们通过属性测试的视角引入了校准性研究的算法方面。我们定义了从样本中进行校准性测试的问题，其中从分布 $\mathcal{D}$（预测，二元结果）中给出 $n$ 次抽样，我们的目标是区分 $\mathcal{D}$ 完全校准和 $\mathcal{D}$ 距离校准性为 $\varepsilon$ 的情况。

    arXiv:2402.13187v1 Announce Type: new  Abstract: In the recent literature on machine learning and decision making, calibration has emerged as a desirable and widely-studied statistical property of the outputs of binary prediction models. However, the algorithmic aspects of measuring model calibration have remained relatively less well-explored. Motivated by [BGHN23], which proposed a rigorous framework for measuring distances to calibration, we initiate the algorithmic study of calibration through the lens of property testing. We define the problem of calibration testing from samples where given $n$ draws from a distribution $\mathcal{D}$ on (predictions, binary outcomes), our goal is to distinguish between the case where $\mathcal{D}$ is perfectly calibrated, and the case where $\mathcal{D}$ is $\varepsilon$-far from calibration.   We design an algorithm based on approximate linear programming, which solves calibration testing information-theoretically optimally (up to constant factor
    
[^2]: 使用Metropolis-adjusted Mirror Langevin算法从约束空间中快速采样

    Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm

    [https://arxiv.org/abs/2312.08823](https://arxiv.org/abs/2312.08823)

    该论文提出了一种名为Metropolis-adjusted Mirror Langevin算法的方法，用于从约束空间中进行快速采样。这种算法是对Mirror Langevin算法的改进，通过添加接受-拒绝过滤器来消除渐近偏差，并具有指数优化依赖。

    

    我们提出了一种新的方法，称为Metropolis-adjusted Mirror Langevin算法，用于从其支持是紧凸集的分布中进行近似采样。该算法在Mirror Langevin算法（Zhang et al., 2020）的单步马尔科夫链中添加了一个接受-拒绝过滤器，Mirror Langevin算法是Mirror Langevin动力学的基本离散化。由于包含了这个过滤器，我们的方法相对于目标是无偏的，而已知的Mirror Langevin算法等Mirror Langevin动力学的离散化具有渐近偏差。对于该算法，我们还给出了混合到一个相对平滑、凸性好且与自共轭镜像函数相关的约束分布所需迭代次数的上界。由于包含Metropolis-Hastings过滤器导致的马尔科夫链是可逆的，我们得到了对误差的指数优化依赖。

    We propose a new method called the Metropolis-adjusted Mirror Langevin algorithm for approximate sampling from distributions whose support is a compact and convex set. This algorithm adds an accept-reject filter to the Markov chain induced by a single step of the Mirror Langevin algorithm (Zhang et al., 2020), which is a basic discretisation of the Mirror Langevin dynamics. Due to the inclusion of this filter, our method is unbiased relative to the target, while known discretisations of the Mirror Langevin dynamics including the Mirror Langevin algorithm have an asymptotic bias. For this algorithm, we also give upper bounds for the number of iterations taken to mix to a constrained distribution whose potential is relatively smooth, convex, and Lipschitz continuous with respect to a self-concordant mirror function. As a consequence of the reversibility of the Markov chain induced by the inclusion of the Metropolis-Hastings filter, we obtain an exponentially better dependence on the erro
    

