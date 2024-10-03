# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Time-Uniform Confidence Spheres for Means of Random Vectors](https://arxiv.org/abs/2311.08168) | 该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。 |
| [^2] | [Fitting an ellipsoid to a quadratic number of random points.](http://arxiv.org/abs/2307.01181) | 将$n$个高斯随机向量拟合到以原点为中心的椭球体边界的问题$(\mathrm{P})$，我们提出了一个基于随机向量Gram矩阵集中性的改进方法，证明了当$n \leq d^2 / C$时，问题$(\mathrm{P})$具有很高的可行性概率。 |
| [^3] | [Minimax Signal Detection in Sparse Additive Models.](http://arxiv.org/abs/2304.09398) | 本研究针对稀疏加性模型中的信号检测问题建立了极小极大分离速率，揭示了稀疏性和函数空间选择之间的非平凡交互作用，并研究了对稀疏性的自适应性和其在通用函数空间中的适用性。在Sobolev空间设置下，我们还讨论了对稀疏性和平滑性的自适应性。 |

# 详细

[^1]: 随机向量均值的时间均匀置信球

    Time-Uniform Confidence Spheres for Means of Random Vectors

    [https://arxiv.org/abs/2311.08168](https://arxiv.org/abs/2311.08168)

    该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。

    

    我们推导并研究了时间均匀置信球——包含随机向量均值并且跨越所有样本量具有很高概率的置信球序列（CSSs）。受Catoni和Giulini原始工作启发，我们统一并扩展了他们的分析，涵盖顺序设置并处理各种分布假设。我们的结果包括有界随机向量的经验伯恩斯坦CSS（导致新颖的经验伯恩斯坦置信区间，渐近宽度按照真实未知方差成比例缩放）、用于子-$\psi$随机向量的CSS（包括子伽马、子泊松和子指数分布）、和用于重尾随机向量（仅有两阶矩）的CSS。最后，我们提供了两个抵抗Huber噪声污染的CSS。第一个是我们经验伯恩斯坦CSS的鲁棒版本，第二个扩展了单变量序列最近的工作。

    arXiv:2311.08168v2 Announce Type: replace-cross  Abstract: We derive and study time-uniform confidence spheres -- confidence sphere sequences (CSSs) -- which contain the mean of random vectors with high probability simultaneously across all sample sizes. Inspired by the original work of Catoni and Giulini, we unify and extend their analysis to cover both the sequential setting and to handle a variety of distributional assumptions. Our results include an empirical-Bernstein CSS for bounded random vectors (resulting in a novel empirical-Bernstein confidence interval with asymptotic width scaling proportionally to the true unknown variance), CSSs for sub-$\psi$ random vectors (which includes sub-gamma, sub-Poisson, and sub-exponential), and CSSs for heavy-tailed random vectors (two moments only). Finally, we provide two CSSs that are robust to contamination by Huber noise. The first is a robust version of our empirical-Bernstein CSS, and the second extends recent work in the univariate se
    
[^2]: 将大量随机点拟合成椭球体的问题

    Fitting an ellipsoid to a quadratic number of random points. (arXiv:2307.01181v1 [math.PR])

    [http://arxiv.org/abs/2307.01181](http://arxiv.org/abs/2307.01181)

    将$n$个高斯随机向量拟合到以原点为中心的椭球体边界的问题$(\mathrm{P})$，我们提出了一个基于随机向量Gram矩阵集中性的改进方法，证明了当$n \leq d^2 / C$时，问题$(\mathrm{P})$具有很高的可行性概率。

    

    我们考虑当$n, d \to \infty $时，将$n$个标准高斯随机向量拟合到以原点为中心的椭球体的边界的问题$(\mathrm{P})$。这个问题被猜测具有尖锐的可行性转变：对于任意$\varepsilon > 0$，如果$n \leq (1 - \varepsilon) d^2 / 4$，那么$(\mathrm{P})$有很高的概率有解；而如果$n \geq (1 + \varepsilon) d^2 /4$，那么$(\mathrm{P})$有很高的概率无解。目前，对于负面情况，只知道$n \geq d^2 / 2$是平凡的一个上界，而对于正面情况，已知的最好结果是假设$n \leq d^2 / \mathrm{polylog}(d)$。在这项工作中，我们利用Bartl和Mendelson关于随机向量的Gram矩阵集中性的一个关键结果改进了以前的方法。这使得我们可以给出一个简单的证明，当$n \leq d^2 / C$时，问题$(\mathrm{P})$有很高的概率是可行的，其中$C> 0$是一个（可能很大的）常数。

    We consider the problem $(\mathrm{P})$ of fitting $n$ standard Gaussian random vectors in $\mathbb{R}^d$ to the boundary of a centered ellipsoid, as $n, d \to \infty$. This problem is conjectured to have a sharp feasibility transition: for any $\varepsilon > 0$, if $n \leq (1 - \varepsilon) d^2 / 4$ then $(\mathrm{P})$ has a solution with high probability, while $(\mathrm{P})$ has no solutions with high probability if $n \geq (1 + \varepsilon) d^2 /4$. So far, only a trivial bound $n \geq d^2 / 2$ is known on the negative side, while the best results on the positive side assume $n \leq d^2 / \mathrm{polylog}(d)$. In this work, we improve over previous approaches using a key result of Bartl & Mendelson on the concentration of Gram matrices of random vectors under mild assumptions on their tail behavior. This allows us to give a simple proof that $(\mathrm{P})$ is feasible with high probability when $n \leq d^2 / C$, for a (possibly large) constant $C > 0$.
    
[^3]: 稀疏加性模型中的极小极大信号检测

    Minimax Signal Detection in Sparse Additive Models. (arXiv:2304.09398v1 [math.ST])

    [http://arxiv.org/abs/2304.09398](http://arxiv.org/abs/2304.09398)

    本研究针对稀疏加性模型中的信号检测问题建立了极小极大分离速率，揭示了稀疏性和函数空间选择之间的非平凡交互作用，并研究了对稀疏性的自适应性和其在通用函数空间中的适用性。在Sobolev空间设置下，我们还讨论了对稀疏性和平滑性的自适应性。

    

    在高维度的建模需求中，稀疏加性模型是一种有吸引力的选择。我们研究了信号检测问题，并建立了一个稀疏加性信号检测的极小极大分离速率。我们的结果是非渐近的，并适用于单变量分量函数属于一般再生核希尔伯特空间的情况。与估计理论不同，极小极大分离速率揭示了稀疏性和函数空间选择之间的非平凡交互作用。我们还研究了对稀疏性的自适应性，并建立了一个通用函数空间的自适应测试速率；在某些空间中，自适应性是可能的，而在其他空间中则会产生不可避免的代价。最后，我们在Sobolev空间设置下研究了对稀疏性和平滑性的自适应性，并更正了文献中存在的一些说法。

    Sparse additive models are an attractive choice in circumstances calling for modelling flexibility in the face of high dimensionality. We study the signal detection problem and establish the minimax separation rate for the detection of a sparse additive signal. Our result is nonasymptotic and applicable to the general case where the univariate component functions belong to a generic reproducing kernel Hilbert space. Unlike the estimation theory, the minimax separation rate reveals a nontrivial interaction between sparsity and the choice of function space. We also investigate adaptation to sparsity and establish an adaptive testing rate for a generic function space; adaptation is possible in some spaces while others impose an unavoidable cost. Finally, adaptation to both sparsity and smoothness is studied in the setting of Sobolev space, and we correct some existing claims in the literature.
    

