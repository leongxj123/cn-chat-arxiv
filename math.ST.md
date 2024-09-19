# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient reductions between some statistical models](https://arxiv.org/abs/2402.07717) | 本研究提出了一种在不知道源统计模型参数的情况下，高效地将样本从源模型转换为目标模型的方法，并构造了几个归约方法。这些归约方法能适应不同的问题，例如专家混合模型、相位恢复和信号降噪等，并且可以处理缺失数据。此外，该研究还指出了一个潜在的应用，即将一个差分隐私机制转换为另一个机制。 |
| [^2] | [About the Cost of Central Privacy in Density Estimation.](http://arxiv.org/abs/2306.14535) | 本研究对于利普希茨和 Sobolev 空间中的非参数密度估计，通过考虑中心隐私的影响，发现了直方图估计器在 L2 风险下对于利普希茨分布是最优的，并且在正常差分隐私情况下也是如此；同时发现，在一些情况下，施加隐私会降低对于 Sobolev 密度的正则极小风险估计。此外，本研究还发现在纯投影估计设定下，所谓的投影估计器对于相同类密度几乎是最优的。 |

# 详细

[^1]: 一些统计模型之间的高效归约

    Efficient reductions between some statistical models

    [https://arxiv.org/abs/2402.07717](https://arxiv.org/abs/2402.07717)

    本研究提出了一种在不知道源统计模型参数的情况下，高效地将样本从源模型转换为目标模型的方法，并构造了几个归约方法。这些归约方法能适应不同的问题，例如专家混合模型、相位恢复和信号降噪等，并且可以处理缺失数据。此外，该研究还指出了一个潜在的应用，即将一个差分隐私机制转换为另一个机制。

    

    我们研究了在不知道源模型参数的情况下，近似地将来自源统计模型的样本转换为目标统计模型的样本的问题，并构造了几个计算上高效的这种统计实验之间的归约。具体而言，我们提供了计算上高效的程序，可以近似将均匀分布、Erlang分布和拉普拉斯分布的位置模型归约到一般的目标族。我们通过建立一些经典的高维问题之间的非渐近归约来说明我们的方法，包括专家混合模型、相位恢复和信号降噪等。值得注意的是，这些归约保持了结构，并可以适应缺失数据。我们还指出了将一个差分隐私机制转换为另一个机制的可能应用。

    We study the problem of approximately transforming a sample from a source statistical model to a sample from a target statistical model without knowing the parameters of the source model, and construct several computationally efficient such reductions between statistical experiments. In particular, we provide computationally efficient procedures that approximately reduce uniform, Erlang, and Laplace location models to general target families. We illustrate our methodology by establishing nonasymptotic reductions between some canonical high-dimensional problems, spanning mixtures of experts, phase retrieval, and signal denoising. Notably, the reductions are structure preserving and can accommodate missing data. We also point to a possible application in transforming one differentially private mechanism to another.
    
[^2]: 关于中心隐私在密度估计中的成本

    About the Cost of Central Privacy in Density Estimation. (arXiv:2306.14535v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2306.14535](http://arxiv.org/abs/2306.14535)

    本研究对于利普希茨和 Sobolev 空间中的非参数密度估计，通过考虑中心隐私的影响，发现了直方图估计器在 L2 风险下对于利普希茨分布是最优的，并且在正常差分隐私情况下也是如此；同时发现，在一些情况下，施加隐私会降低对于 Sobolev 密度的正则极小风险估计。此外，本研究还发现在纯投影估计设定下，所谓的投影估计器对于相同类密度几乎是最优的。

    

    我们研究利普希茨和 Sobolev 空间中的非参数密度估计，在中心隐私条件下进行。我们考虑了隐私预算不是常数的情况。我们考虑了经典的中心差分隐私定义，以及较新的中心集中差分隐私概念。我们证实了 Barber & Duchi (2014) 的结果，即直方图估计器在对于 L2 风险下对于利普希茨分布是最优的，并且在正常差分隐私情况下也是如此，我们将其扩展到其他范数和隐私概念。然后，我们研究更高程度的光滑性，得出两个结论：首先，与常数隐私预算需要的情况相反（Wasserman &amp; Zhou, 2010），在 Sobolev 密度上施加隐私会降低正则极小风险估计。其次，在这种新的纯投影估计设定下，所谓的投影估计器对于相同类密度是几乎最优的。

    We study non-parametric density estimation for densities in Lipschitz and Sobolev spaces, and under central privacy. In particular, we investigate regimes where the privacy budget is not supposed to be constant. We consider the classical definition of central differential privacy, but also the more recent notion of central concentrated differential privacy. We recover the result of Barber \& Duchi (2014) stating that histogram estimators are optimal against Lipschitz distributions for the L2 risk, and under regular differential privacy, and we extend it to other norms and notions of privacy. Then, we investigate higher degrees of smoothness, drawing two conclusions: First, and contrary to what happens with constant privacy budget (Wasserman \& Zhou, 2010), there are regimes where imposing privacy degrades the regular minimax risk of estimation on Sobolev densities. Second, so-called projection estimators are near-optimal against the same classes of densities in this new setup with pure
    

