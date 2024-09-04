# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Split Balancing for Optimal Random Forest](https://arxiv.org/abs/2402.11228) | 介绍了自适应分割平衡森林（ASBF），可在学习树表示的同时，在复杂情况下实现极小极优性，并提出了一个本地化版本，在H\"older类下达到最小极优性。 |
| [^2] | [On the Optimality of Misspecified Spectral Algorithms.](http://arxiv.org/abs/2303.14942) | 在本文中，我们研究了非准确谱算法的最优性问题。我们证明了在一些特定的RKHSs上，谱算法对于所有的$s\in (0,1)$都是极小极大最优的。 |

# 详细

[^1]: 自适应分割平衡优化随机森林

    Adaptive Split Balancing for Optimal Random Forest

    [https://arxiv.org/abs/2402.11228](https://arxiv.org/abs/2402.11228)

    介绍了自适应分割平衡森林（ASBF），可在学习树表示的同时，在复杂情况下实现极小极优性，并提出了一个本地化版本，在H\"older类下达到最小极优性。

    

    尽管随机森林通常用于回归问题，但现有方法在复杂情况下缺乏适应性，或在简单、平滑情景下失去最优性。在本研究中，我们介绍了自适应分割平衡森林（ASBF），能够从数据中学习树表示，同时在Lipschitz类下实现极小极优性。为了利用更高阶的平滑性水平，我们进一步提出了一个本地化版本，该版本在任意$q \in \mathbb{N}$和$\beta \in (0,1]$的Hölder类$\mathcal{H}^{q,\beta}$下达到最小极优性。与广泛使用的随机特征选择不同，我们考虑了对现有方法的平衡修改。我们的结果表明，过度依赖辅助随机性可能会损害树模型的逼近能力，导致次优结果。相反，一个更平衡、更少随机的方法表现出最佳性能。

    arXiv:2402.11228v1 Announce Type: cross  Abstract: While random forests are commonly used for regression problems, existing methods often lack adaptability in complex situations or lose optimality under simple, smooth scenarios. In this study, we introduce the adaptive split balancing forest (ASBF), capable of learning tree representations from data while simultaneously achieving minimax optimality under the Lipschitz class. To exploit higher-order smoothness levels, we further propose a localized version that attains the minimax rate under the H\"older class $\mathcal{H}^{q,\beta}$ for any $q\in\mathbb{N}$ and $\beta\in(0,1]$. Rather than relying on the widely-used random feature selection, we consider a balanced modification to existing approaches. Our results indicate that an over-reliance on auxiliary randomness may compromise the approximation power of tree models, leading to suboptimal results. Conversely, a less random, more balanced approach demonstrates optimality. Additionall
    
[^2]: 非准确谱算法的最优性

    On the Optimality of Misspecified Spectral Algorithms. (arXiv:2303.14942v2 [math.ST] CROSS LISTED)

    [http://arxiv.org/abs/2303.14942](http://arxiv.org/abs/2303.14942)

    在本文中，我们研究了非准确谱算法的最优性问题。我们证明了在一些特定的RKHSs上，谱算法对于所有的$s\in (0,1)$都是极小极大最优的。

    

    在非准确谱算法问题中，研究人员通常假设地下真实函数$f_{\rho}^{*} \in [\mathcal{H}]^{s}$，其中$\mathcal{H}$是一个再生核希尔伯特空间(RKHS)的较平滑插值空间，$s\in (0,1)$。现有的极小极大最优结果要求$\|f_{\rho}^{*}\|_{L^{\infty}}<\infty$，这隐含地要求$s > \alpha_{0}$，其中$\alpha_{0}\in (0,1)$是嵌入指数，一个依赖于$\mathcal{H}$的常数。关于谱算法是否对所有的$s\in (0,1)$都是最优的问题已经存在多年。在本文中，我们证明了谱算法是对于任意的$\alpha_{0}-\frac{1}{\beta} < s < 1$都是极小极大最优的，其中$\beta$是$\mathcal{H}$的特征值衰减率。我们还给出了几类满足$ \alpha_0 = \frac{1}{\beta} $的RKHSs，因此，谱算法在这些RKHSs上对于所有的$s\in (0,1)$都是极小极大最优的。

    In the misspecified spectral algorithms problem, researchers usually assume the underground true function $f_{\rho}^{*} \in [\mathcal{H}]^{s}$, a less-smooth interpolation space of a reproducing kernel Hilbert space (RKHS) $\mathcal{H}$ for some $s\in (0,1)$. The existing minimax optimal results require $\|f_{\rho}^{*}\|_{L^{\infty}}<\infty$ which implicitly requires $s > \alpha_{0}$ where $\alpha_{0}\in (0,1)$ is the embedding index, a constant depending on $\mathcal{H}$. Whether the spectral algorithms are optimal for all $s\in (0,1)$ is an outstanding problem lasting for years. In this paper, we show that spectral algorithms are minimax optimal for any $\alpha_{0}-\frac{1}{\beta} < s < 1$, where $\beta$ is the eigenvalue decay rate of $\mathcal{H}$. We also give several classes of RKHSs whose embedding index satisfies $ \alpha_0 = \frac{1}{\beta} $. Thus, the spectral algorithms are minimax optimal for all $s\in (0,1)$ on these RKHSs.
    

