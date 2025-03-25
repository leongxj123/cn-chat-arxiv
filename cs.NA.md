# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sharp Analysis of Power Iteration for Tensor PCA.](http://arxiv.org/abs/2401.01047) | 本文中，我们对Tensor PCA模型中的功率迭代算法进行了详细分析，超越了之前的限制，并建立了关于收敛次数的尖锐界限和算法阈值。我们还提出了一种有效的停止准则来获得高度相关的解决方案。 |
| [^2] | [Optimal Approximation of Zonoids and Uniform Approximation by Shallow Neural Networks.](http://arxiv.org/abs/2307.15285) | 本论文解决了Zonoid的最优逼近和浅层神经网络的均匀逼近两个问题。对于Zonoid的逼近，我们填补了在$d=2,3$时的对数差距，实现了在所有维度上的解决方案。对于神经网络的逼近，我们的技术在$k \geq 1$时显著提高了目前的逼近率，并能够均匀逼近目标函数及其导数。 |

# 详细

[^1]: Tensor PCA的功率迭代的尖锐分析

    Sharp Analysis of Power Iteration for Tensor PCA. (arXiv:2401.01047v1 [cs.LG])

    [http://arxiv.org/abs/2401.01047](http://arxiv.org/abs/2401.01047)

    本文中，我们对Tensor PCA模型中的功率迭代算法进行了详细分析，超越了之前的限制，并建立了关于收敛次数的尖锐界限和算法阈值。我们还提出了一种有效的停止准则来获得高度相关的解决方案。

    

    我们调查了Richard和Montanari（2014）引入的Tensor PCA模型的功率迭代算法。之前研究Tensor功率迭代算法的工作要么仅限于固定次数的迭代，要么需要一个非平凡的与数据无关的初始化。在本文中，我们超越了这些限制，并对随机初始化的Tensor功率迭代的动态进行了多项式数量级的分析。我们的贡献有三个方面：首先，我们建立了对于广泛的信噪比范围下，功率迭代收敛到种植信号所需迭代次数的尖锐界限。其次，我们的分析揭示了实际的算法阈值比文献中猜测的要小一个polylog(n)的因子，其中n是环境维度。最后，我们提出了一种简单而有效的功率迭代停止准则，可以保证输出与真实信号高度相关的解决方案。

    We investigate the power iteration algorithm for the tensor PCA model introduced in Richard and Montanari (2014). Previous work studying the properties of tensor power iteration is either limited to a constant number of iterations, or requires a non-trivial data-independent initialization. In this paper, we move beyond these limitations and analyze the dynamics of randomly initialized tensor power iteration up to polynomially many steps. Our contributions are threefold: First, we establish sharp bounds on the number of iterations required for power method to converge to the planted signal, for a broad range of the signal-to-noise ratios. Second, our analysis reveals that the actual algorithmic threshold for power iteration is smaller than the one conjectured in literature by a polylog(n) factor, where n is the ambient dimension. Finally, we propose a simple and effective stopping criterion for power iteration, which provably outputs a solution that is highly correlated with the true si
    
[^2]: Zonoid的最优逼近和浅层神经网络的均匀逼近

    Optimal Approximation of Zonoids and Uniform Approximation by Shallow Neural Networks. (arXiv:2307.15285v1 [stat.ML])

    [http://arxiv.org/abs/2307.15285](http://arxiv.org/abs/2307.15285)

    本论文解决了Zonoid的最优逼近和浅层神经网络的均匀逼近两个问题。对于Zonoid的逼近，我们填补了在$d=2,3$时的对数差距，实现了在所有维度上的解决方案。对于神经网络的逼近，我们的技术在$k \geq 1$时显著提高了目前的逼近率，并能够均匀逼近目标函数及其导数。

    

    我们研究了以下两个相关问题。第一个问题是确定一个任意的在$\mathbb{R}^{d+1}$空间中的Zonoid可以通过$n$个线段的Hausdorff距离来逼近的误差。第二个问题是确定浅层ReLU$^k$神经网络在其变分空间中的均匀范数的最优逼近率。第一个问题已经在$d \neq 2, 3$时得到解决，但当$d = 2, 3$时，最优上界和最优下界之间仍存在一个对数差距。我们填补了这个差距，完成了所有维度上的解决方案。对于第二个问题，我们的技术在$k \geq 1$时显著提高了现有的逼近率，并实现了目标函数及其导数的均匀逼近。

    We study the following two related problems. The first is to determine to what error an arbitrary zonoid in $\mathbb{R}^{d+1}$ can be approximated in the Hausdorff distance by a sum of $n$ line segments. The second is to determine optimal approximation rates in the uniform norm for shallow ReLU$^k$ neural networks on their variation spaces. The first of these problems has been solved for $d\neq 2,3$, but when $d=2,3$ a logarithmic gap between the best upper and lower bounds remains. We close this gap, which completes the solution in all dimensions. For the second problem, our techniques significantly improve upon existing approximation rates when $k\geq 1$, and enable uniform approximation of both the target function and its derivatives.
    

