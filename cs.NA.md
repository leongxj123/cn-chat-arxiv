# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Fractional Differential Equations](https://arxiv.org/abs/2403.02737) | 提出了神经FDE，一种新型深度神经网络架构，可调整FDE以适应数据动态，可能优于神经OD。 |
| [^2] | [Lattice Approximations in Wasserstein Space.](http://arxiv.org/abs/2310.09149) | 本论文研究了在Wasserstein空间中通过离散和分段常数测度进行的结构逼近方法。结果表明，对于满秩的格点按比例缩放后得到的Voronoi分割逼近的测度误差是$O(h)$，逼近的$N$项误差为$O(N^{-\frac1d})$，并且可以推广到非紧支撑测度。 |

# 详细

[^1]: 神经分数阶微分方程

    Neural Fractional Differential Equations

    [https://arxiv.org/abs/2403.02737](https://arxiv.org/abs/2403.02737)

    提出了神经FDE，一种新型深度神经网络架构，可调整FDE以适应数据动态，可能优于神经OD。

    

    分数阶微分方程（FDEs）是科学和工程中建模复杂系统的基本工具。 它们将传统的微分和积分概念扩展到非整数阶，使得能够更精确地表示具有非局部和记忆依赖行为特征的过程。在这个背景下，受神经常微分方程（Neural ODEs）的启发，我们提出了神经FDE，这是一种调整FDE以适应数据动态的新型深度神经网络架构。这项工作全面概述了神经FDE中采用的数值方法和神经FDE架构。数值结果表明，尽管计算要求更高，神经FDE可能优于神经OD。

    arXiv:2403.02737v1 Announce Type: new  Abstract: Fractional Differential Equations (FDEs) are essential tools for modelling complex systems in science and engineering. They extend the traditional concepts of differentiation and integration to non-integer orders, enabling a more precise representation of processes characterised by non-local and memory-dependent behaviours.   This property is useful in systems where variables do not respond to changes instantaneously, but instead exhibit a strong memory of past interactions.   Having this in mind, and drawing inspiration from Neural Ordinary Differential Equations (Neural ODEs), we propose the Neural FDE, a novel deep neural network architecture that adjusts a FDE to the dynamics of data.   This work provides a comprehensive overview of the numerical method employed in Neural FDEs and the Neural FDE architecture. The numerical outcomes suggest that, despite being more computationally demanding, the Neural FDE may outperform the Neural OD
    
[^2]: 微分水平空间中的格点逼近

    Lattice Approximations in Wasserstein Space. (arXiv:2310.09149v1 [stat.ML])

    [http://arxiv.org/abs/2310.09149](http://arxiv.org/abs/2310.09149)

    本论文研究了在Wasserstein空间中通过离散和分段常数测度进行的结构逼近方法。结果表明，对于满秩的格点按比例缩放后得到的Voronoi分割逼近的测度误差是$O(h)$，逼近的$N$项误差为$O(N^{-\frac1d})$，并且可以推广到非紧支撑测度。

    

    我们考虑在Wasserstein空间$W_p(\mathbb{R}^d)$中通过离散和分段常数测度来对测度进行结构逼近。我们证明，如果一个满秩的格点$\Lambda$按照$h\in(0,1]$的比例进行缩放，那么基于$h\Lambda$的Voronoi分割得到的测度逼近是$O(h)$，不论$d$或$p$的取值。之后，我们使用覆盖论证证明，对于紧支撑的测度的$N$项逼近是$O(N^{-\frac1d})$，这与最优量化器和经验测度逼近在大多数情况下已知的速率相匹配。最后，我们将这些结果推广到非紧支撑测度，要求其具有足够的衰减性质。

    We consider structured approximation of measures in Wasserstein space $W_p(\mathbb{R}^d)$ for $p\in[1,\infty)$ by discrete and piecewise constant measures based on a scaled Voronoi partition of $\mathbb{R}^d$. We show that if a full rank lattice $\Lambda$ is scaled by a factor of $h\in(0,1]$, then approximation of a measure based on the Voronoi partition of $h\Lambda$ is $O(h)$ regardless of $d$ or $p$. We then use a covering argument to show that $N$-term approximations of compactly supported measures is $O(N^{-\frac1d})$ which matches known rates for optimal quantizers and empirical measure approximation in most instances. Finally, we extend these results to noncompactly supported measures with sufficient decay.
    

