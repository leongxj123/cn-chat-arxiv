# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Block Majorization Minimization with Extrapolation and Application to $\beta$-NMF.](http://arxiv.org/abs/2401.06646) | 本文提出了一种使用外推的块主导极小化方法（BMMe）来解决多凸优化问题，并将其应用于$\beta$-NMF。通过使用独特的自适应更新规则来更新外推参数，该方法在实验中展现出显著的加速效果。 |

# 详细

[^1]: 使用外推的块主导极小化方法和应用于$\beta$-NMF

    Block Majorization Minimization with Extrapolation and Application to $\beta$-NMF. (arXiv:2401.06646v1 [cs.LG])

    [http://arxiv.org/abs/2401.06646](http://arxiv.org/abs/2401.06646)

    本文提出了一种使用外推的块主导极小化方法（BMMe）来解决多凸优化问题，并将其应用于$\beta$-NMF。通过使用独特的自适应更新规则来更新外推参数，该方法在实验中展现出显著的加速效果。

    

    我们提出了一种使用外推的块主导极小化方法（BMMe）来解决一类多凸优化问题。BMMe的外推参数使用一种新颖的自适应更新规则来更新。通过将块主导极小化重新表述为块镜像下降方法，并在每次迭代中自适应更新Bregman散度，我们建立了BMMe的子序列收敛性。我们使用这种方法设计了高效的算法来处理$\beta$-NMF中的非负矩阵分解问题，其中$\beta\in [1,2]$。这些算法是使用外推的乘法更新，并从我们的新结果中获得了收敛性保证。我们还通过大量实验实证了BMMe在$\beta$-NMF中的显著加速效果。

    We propose a Block Majorization Minimization method with Extrapolation (BMMe) for solving a class of multi-convex optimization problems. The extrapolation parameters of BMMe are updated using a novel adaptive update rule. By showing that block majorization minimization can be reformulated as a block mirror descent method, with the Bregman divergence adaptively updated at each iteration, we establish subsequential convergence for BMMe. We use this method to design efficient algorithms to tackle nonnegative matrix factorization problems with the $\beta$-divergences ($\beta$-NMF) for $\beta\in [1,2]$. These algorithms, which are multiplicative updates with extrapolation, benefit from our novel results that offer convergence guarantees. We also empirically illustrate the significant acceleration of BMMe for $\beta$-NMF through extensive experiments.
    

