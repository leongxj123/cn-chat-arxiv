# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differential Privacy of Noisy (S)GD under Heavy-Tailed Perturbations](https://arxiv.org/abs/2403.02051) | 在重尾扰动下，噪声SGD实现了差分隐私保证，适用于广泛的损失函数类，特别是非凸函数。 |

# 详细

[^1]: 在重尾扰动下噪声(S)GD的差分隐私

    Differential Privacy of Noisy (S)GD under Heavy-Tailed Perturbations

    [https://arxiv.org/abs/2403.02051](https://arxiv.org/abs/2403.02051)

    在重尾扰动下，噪声SGD实现了差分隐私保证，适用于广泛的损失函数类，特别是非凸函数。

    

    将重尾噪声注入随机梯度下降(SGD)的迭代中已经引起越来越多的关注。尽管对导致的算法的各种理论性质进行了分析，主要来自学习理论和优化视角，但它们的隐私保护性质尚未建立。为了弥补这一缺口，我们为噪声SGD提供差分隐私(DP)保证，当注入的噪声遵循$\alpha$-稳定分布时，该分布包括一系列重尾分布(具有无限方差)以及高斯分布。考虑$(\epsilon,\delta)$-DP框架，我们表明带有重尾扰动的SGD实现了$(0,\tilde{\mathcal{O}}(1/n))$-DP的广泛损失函数类，这些函数可以是非凸的，这里$n$是数据点的数量。作为一项显着的副产品，与以往的工作相反，该工作要求有界se

    arXiv:2403.02051v1 Announce Type: cross  Abstract: Injecting heavy-tailed noise to the iterates of stochastic gradient descent (SGD) has received increasing attention over the past few years. While various theoretical properties of the resulting algorithm have been analyzed mainly from learning theory and optimization perspectives, their privacy preservation properties have not yet been established. Aiming to bridge this gap, we provide differential privacy (DP) guarantees for noisy SGD, when the injected noise follows an $\alpha$-stable distribution, which includes a spectrum of heavy-tailed distributions (with infinite variance) as well as the Gaussian distribution. Considering the $(\epsilon, \delta)$-DP framework, we show that SGD with heavy-tailed perturbations achieves $(0, \tilde{\mathcal{O}}(1/n))$-DP for a broad class of loss functions which can be non-convex, where $n$ is the number of data points. As a remarkable byproduct, contrary to prior work that necessitates bounded se
    

