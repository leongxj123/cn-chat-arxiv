# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficiently matching random inhomogeneous graphs via degree profiles.](http://arxiv.org/abs/2310.10441) | 本文提出了一种通过度特征匹配算法高效匹配随机不均匀图的方法，要求最小平均度和最小相关性达到一定阈值。 |
| [^2] | [Kernel Ridge Regression Inference.](http://arxiv.org/abs/2302.06578) | 我们提供了核岭回归方法的一致推断和置信带，为广泛应用于各种数据类型的非参数回归估计器提供了准确的统计推断方法。 |

# 详细

[^1]: 通过度特征高效匹配随机不均匀图

    Efficiently matching random inhomogeneous graphs via degree profiles. (arXiv:2310.10441v1 [cs.DS])

    [http://arxiv.org/abs/2310.10441](http://arxiv.org/abs/2310.10441)

    本文提出了一种通过度特征匹配算法高效匹配随机不均匀图的方法，要求最小平均度和最小相关性达到一定阈值。

    

    本文研究了恢复两个相关的随机图之间潜在顶点对应关系的问题，这两个图具有极不均匀且未知的不同顶点对之间的边概率。在Ding、Ma、Wu和Xu(2021)提出的度特征匹配算法的基础上，我们扩展出了一种高效的匹配算法，只要最小平均度至少为$\Omega(\log^{2} n)$，最小相关性至少为$1 - O(\log^{-2} n)$。

    In this paper, we study the problem of recovering the latent vertex correspondence between two correlated random graphs with vastly inhomogeneous and unknown edge probabilities between different pairs of vertices. Inspired by and extending the matching algorithm via degree profiles by Ding, Ma, Wu and Xu (2021), we obtain an efficient matching algorithm as long as the minimal average degree is at least $\Omega(\log^{2} n)$ and the minimal correlation is at least $1 - O(\log^{-2} n)$.
    
[^2]: 核岭回归推断

    Kernel Ridge Regression Inference. (arXiv:2302.06578v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.06578](http://arxiv.org/abs/2302.06578)

    我们提供了核岭回归方法的一致推断和置信带，为广泛应用于各种数据类型的非参数回归估计器提供了准确的统计推断方法。

    

    我们提供了核岭回归(KRR)的一致推断和置信带，这是一种广泛应用于包括排名、图像和图表在内的一般数据类型的非参数回归估计器。尽管这些数据的普遍存在，如学校分配中的排序优先级列表，但KRR的推断理论尚未完全知悉，限制了它在经济学和其他科学领域中的作用。我们构建了针对一般回归器的尖锐、一致的置信区间。为了进行推断，我们开发了一种有效的自举程序，通过对称化来消除偏差并限制计算开销。为了证明该程序，我们推导了再生核希尔伯特空间(RKHS)中部分和的有限样本、均匀高斯和自举耦合。这些推导暗示了基于RKHS单位球的经验过程的强逼近，对覆盖数具有对数依赖关系。模拟验证了置信度。

    We provide uniform inference and confidence bands for kernel ridge regression (KRR), a widely-used non-parametric regression estimator for general data types including rankings, images, and graphs. Despite the prevalence of these data -e.g., ranked preference lists in school assignment -- the inferential theory of KRR is not fully known, limiting its role in economics and other scientific domains. We construct sharp, uniform confidence sets for KRR, which shrink at nearly the minimax rate, for general regressors. To conduct inference, we develop an efficient bootstrap procedure that uses symmetrization to cancel bias and limit computational overhead. To justify the procedure, we derive finite-sample, uniform Gaussian and bootstrap couplings for partial sums in a reproducing kernel Hilbert space (RKHS). These imply strong approximation for empirical processes indexed by the RKHS unit ball with logarithmic dependence on the covering number. Simulations verify coverage. We use our proce
    

