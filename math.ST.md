# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Computational-Statistical Gaps for Improper Learning in Sparse Linear Regression](https://arxiv.org/abs/2402.14103) | 该研究探讨了稀疏线性回归中的计算统计差距问题，为了高效地找到可以在样本上实现非平凡预测误差的潜在密集估计的回归向量，需要至少 $\Omega(k \log (d/k))$ 个样本。 |
| [^2] | [Information Geometry of Wasserstein Statistics on Shapes and Affine Deformations.](http://arxiv.org/abs/2307.12508) | 在这篇论文中，我们研究了Wasserstein统计在仿射变形统计模型中的信息几何特征，比较了信息几何和Wasserstein几何的估计器的优缺点，并发现Wasserstein估计量在椭圆对称仿射变形模型中是矩估计量，在波形为高斯分布时与信息几何估计量重合。 |

# 详细

[^1]: 稀疏线性回归中不当学习的计算统计差距

    Computational-Statistical Gaps for Improper Learning in Sparse Linear Regression

    [https://arxiv.org/abs/2402.14103](https://arxiv.org/abs/2402.14103)

    该研究探讨了稀疏线性回归中的计算统计差距问题，为了高效地找到可以在样本上实现非平凡预测误差的潜在密集估计的回归向量，需要至少 $\Omega(k \log (d/k))$ 个样本。

    

    我们研究了稀疏线性回归中不当学习的计算统计差距。具体来说，给定来自维度为 $d$ 的 $k$-稀疏线性模型的 $n$ 个样本，我们询问了在时间多项式中的最小样本复杂度，以便高效地找到一个对这 $n$ 个样本达到非平凡预测误差的潜在密集估计的回归向量。信息理论上，这可以用 $\Theta(k \log (d/k))$ 个样本实现。然而，尽管在文献中很显著，但没有已知的多项式时间算法可以在不附加对模型的其他限制的情况下使用少于 $\Theta(d)$ 个样本达到相同的保证。类似地，现有的困难结果要么仅限于适当设置，在该设置中估计值也必须是稀疏的，要么仅适用于特定算法。

    arXiv:2402.14103v1 Announce Type: new  Abstract: We study computational-statistical gaps for improper learning in sparse linear regression. More specifically, given $n$ samples from a $k$-sparse linear model in dimension $d$, we ask what is the minimum sample complexity to efficiently (in time polynomial in $d$, $k$, and $n$) find a potentially dense estimate for the regression vector that achieves non-trivial prediction error on the $n$ samples. Information-theoretically this can be achieved using $\Theta(k \log (d/k))$ samples. Yet, despite its prominence in the literature, there is no polynomial-time algorithm known to achieve the same guarantees using less than $\Theta(d)$ samples without additional restrictions on the model. Similarly, existing hardness results are either restricted to the proper setting, in which the estimate must be sparse as well, or only apply to specific algorithms.   We give evidence that efficient algorithms for this task require at least (roughly) $\Omega(
    
[^2]: 形状和仿射变形的Wasserstein统计的信息几何

    Information Geometry of Wasserstein Statistics on Shapes and Affine Deformations. (arXiv:2307.12508v1 [math.ST])

    [http://arxiv.org/abs/2307.12508](http://arxiv.org/abs/2307.12508)

    在这篇论文中，我们研究了Wasserstein统计在仿射变形统计模型中的信息几何特征，比较了信息几何和Wasserstein几何的估计器的优缺点，并发现Wasserstein估计量在椭圆对称仿射变形模型中是矩估计量，在波形为高斯分布时与信息几何估计量重合。

    

    信息几何和Wasserstein几何是介绍概率分布流形中的两个主要结构，它们捕捉了不同的特征。我们在仿射变形统计模型的Li和Zhao（2023）框架中研究了Wasserstein几何的特征，它是位置-尺度模型的多维泛化。我们比较了基于信息几何和Wasserstein几何的估计器的优点和缺点。在Wasserstein几何中，概率分布的形状和仿射变形是分离的，表明在对波形扰动具有鲁棒性的同时，会损失Fisher效率。我们证明了在椭圆对称仿射变形模型的情况下Wasserstein估计量是矩估计量。它与信息几何估计量（最大似然估计量）仅在波形为高斯分布时重合。Wasserstein效率的作用是...

    Information geometry and Wasserstein geometry are two main structures introduced in a manifold of probability distributions, and they capture its different characteristics. We study characteristics of Wasserstein geometry in the framework of Li and Zhao (2023) for the affine deformation statistical model, which is a multi-dimensional generalization of the location-scale model. We compare merits and demerits of estimators based on information geometry and Wasserstein geometry. The shape of a probability distribution and its affine deformation are separated in the Wasserstein geometry, showing its robustness against the waveform perturbation in exchange for the loss in Fisher efficiency. We show that the Wasserstein estimator is the moment estimator in the case of the elliptically symmetric affine deformation model. It coincides with the information-geometrical estimator (maximum-likelihood estimator) when and only when the waveform is Gaussian. The role of the Wasserstein efficiency is 
    

