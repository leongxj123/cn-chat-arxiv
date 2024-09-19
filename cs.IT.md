# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient reductions between some statistical models](https://arxiv.org/abs/2402.07717) | 本研究提出了一种在不知道源统计模型参数的情况下，高效地将样本从源模型转换为目标模型的方法，并构造了几个归约方法。这些归约方法能适应不同的问题，例如专家混合模型、相位恢复和信号降噪等，并且可以处理缺失数据。此外，该研究还指出了一个潜在的应用，即将一个差分隐私机制转换为另一个机制。 |
| [^2] | [Canonical foliations of neural networks: application to robustness.](http://arxiv.org/abs/2203.00922) | 本文探讨了利用黎曼几何和叶面理论创新应用于神经网络鲁棒性的新视角，提出了一种适用于数据空间的以曲率为考量因素的 two-step spectral 对抗攻击方法。 |

# 详细

[^1]: 一些统计模型之间的高效归约

    Efficient reductions between some statistical models

    [https://arxiv.org/abs/2402.07717](https://arxiv.org/abs/2402.07717)

    本研究提出了一种在不知道源统计模型参数的情况下，高效地将样本从源模型转换为目标模型的方法，并构造了几个归约方法。这些归约方法能适应不同的问题，例如专家混合模型、相位恢复和信号降噪等，并且可以处理缺失数据。此外，该研究还指出了一个潜在的应用，即将一个差分隐私机制转换为另一个机制。

    

    我们研究了在不知道源模型参数的情况下，近似地将来自源统计模型的样本转换为目标统计模型的样本的问题，并构造了几个计算上高效的这种统计实验之间的归约。具体而言，我们提供了计算上高效的程序，可以近似将均匀分布、Erlang分布和拉普拉斯分布的位置模型归约到一般的目标族。我们通过建立一些经典的高维问题之间的非渐近归约来说明我们的方法，包括专家混合模型、相位恢复和信号降噪等。值得注意的是，这些归约保持了结构，并可以适应缺失数据。我们还指出了将一个差分隐私机制转换为另一个机制的可能应用。

    We study the problem of approximately transforming a sample from a source statistical model to a sample from a target statistical model without knowing the parameters of the source model, and construct several computationally efficient such reductions between statistical experiments. In particular, we provide computationally efficient procedures that approximately reduce uniform, Erlang, and Laplace location models to general target families. We illustrate our methodology by establishing nonasymptotic reductions between some canonical high-dimensional problems, spanning mixtures of experts, phase retrieval, and signal denoising. Notably, the reductions are structure preserving and can accommodate missing data. We also point to a possible application in transforming one differentially private mechanism to another.
    
[^2]: 神经网络的规范叶面：鲁棒性应用研究

    Canonical foliations of neural networks: application to robustness. (arXiv:2203.00922v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2203.00922](http://arxiv.org/abs/2203.00922)

    本文探讨了利用黎曼几何和叶面理论创新应用于神经网络鲁棒性的新视角，提出了一种适用于数据空间的以曲率为考量因素的 two-step spectral 对抗攻击方法。

    

    深度学习模型易受到对抗攻击。而对抗学习正在变得至关重要。本文提出了一种新的神经网络鲁棒性视角，采用黎曼几何和叶面理论。通过创建考虑数据空间曲率的新对抗攻击，即 two-step spectral attack，来说明这个想法。数据空间被视为一个配备了神经网络的 Fisher 信息度量（FIM）拉回的（退化的）黎曼流形。大多数情况下，该度量仅为半正定，其内核成为研究的核心对象。从该核中导出一个规范叶面。横向叶的曲率给出了适当的修正，从而得到了两步近似的测地线和一种新的高效对抗攻击。该方法首先在一个 2D 玩具示例中进行演示。

    Deep learning models are known to be vulnerable to adversarial attacks. Adversarial learning is therefore becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory. The idea is illustrated by creating a new adversarial attack that takes into account the curvature of the data space. This new adversarial attack called the two-step spectral attack is a piece-wise linear approximation of a geodesic in the data space. The data space is treated as a (degenerate) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of transverse leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. The method is first illustrated on a 2D toy example
    

