# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [$\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples](https://arxiv.org/abs/2402.01879) | 该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。 |
| [^2] | [Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies.](http://arxiv.org/abs/2210.06140) | 本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。 |

# 详细

[^1]: $\sigma$-zero: 基于梯度的$\ell_0$-范数对抗样本优化

    $\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples

    [https://arxiv.org/abs/2402.01879](https://arxiv.org/abs/2402.01879)

    该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。

    

    评估深度网络对基于梯度攻击的对抗鲁棒性是具有挑战性的。虽然大多数攻击考虑$\ell_2$和$\ell_\infty$范数约束来制造输入扰动，但只有少数研究了稀疏的$\ell_1$和$\ell_0$范数攻击。特别是，由于在非凸且非可微约束上进行优化的固有复杂性，$\ell_0$范数攻击是研究最少的。然而，使用这些攻击评估对抗鲁棒性可以揭示在更传统的$\ell_2$和$\ell_\infty$范数攻击中未能测试出的弱点。在这项工作中，我们提出了一种新颖的$\ell_0$范数攻击，称为$\sigma$-zero，它利用了$\ell_0$范数的一个特殊可微近似来促进基于梯度的优化，并利用自适应投影运算符动态调整损失最小化和扰动稀疏性之间的权衡。通过在MNIST、CIFAR10和ImageNet数据集上进行广泛评估，包括...

    Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages an ad hoc differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving
    
[^2]: 差分隐私引导采样：新的隐私分析与推断策略

    Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies. (arXiv:2210.06140v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.06140](http://arxiv.org/abs/2210.06140)

    本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。

    

    差分隐私机制通过引入随机性来保护个人信息，但在应用中，统计推断仍然缺乏通用技术。本文研究了一个差分隐私引导采样方法，通过发布多个私有引导采样估计来推断样本分布并构建置信区间。我们的隐私分析提供了单个差分隐私引导采样估计的隐私成本新结果，适用于任何差分隐私机制，并指出了现有文献中引导采样的一些误用。使用Gaussian-DP（GDP）框架，我们证明从满足 $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP 的机制中释放 $B$ 个差分隐私引导采样估计，在 $B$ 趋近无限大时渐近地满足 $\mu$-GDP。此外，我们使用差分隐私引导采样估计的反卷积对样本分布进行准确推断。

    Differentially private (DP) mechanisms protect individual-level information by introducing randomness into the statistical analysis procedure. Despite the availability of numerous DP tools, there remains a lack of general techniques for conducting statistical inference under DP. We examine a DP bootstrap procedure that releases multiple private bootstrap estimates to infer the sampling distribution and construct confidence intervals (CIs). Our privacy analysis presents new results on the privacy cost of a single DP bootstrap estimate, applicable to any DP mechanisms, and identifies some misapplications of the bootstrap in the existing literature. Using the Gaussian-DP (GDP) framework (Dong et al.,2022), we show that the release of $B$ DP bootstrap estimates from mechanisms satisfying $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP asymptotically satisfies $\mu$-GDP as $B$ goes to infinity. Moreover, we use deconvolution with the DP bootstrap estimates to accurately infer the sampling distribution
    

