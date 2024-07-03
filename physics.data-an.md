# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Probabilistic Stability of Stochastic Gradient Descent.](http://arxiv.org/abs/2303.13093) | 本文重新定义了随机梯度下降的稳定性，并使用概率稳定性来回答深度学习理论中的一个根本问题：SGD如何从数量庞大的可能严重过拟合的解中选择有意义的神经网络解。 |

# 详细

[^1]: 随机梯度下降的概率稳定性

    The Probabilistic Stability of Stochastic Gradient Descent. (arXiv:2303.13093v1 [cs.LG])

    [http://arxiv.org/abs/2303.13093](http://arxiv.org/abs/2303.13093)

    本文重新定义了随机梯度下降的稳定性，并使用概率稳定性来回答深度学习理论中的一个根本问题：SGD如何从数量庞大的可能严重过拟合的解中选择有意义的神经网络解。

    

    深度学习理论中的一个基本开放问题是如何定义和理解随机梯度下降(SGD)接近固定点的稳定性。传统文献依赖于参数统计矩，特别是参数方差的收敛来量化稳定性。本文重新定义了SGD的稳定性，并使用\textit{概率收敛}条件来定义SGD的\textit{概率稳定性}。提出的稳定性直接回答了深度学习理论中的一个根本问题：SGD如何从数量庞大的可能严重过拟合的解中选择有意义的神经网络解。为了实现这一点，我们表明只有在概率性稳定性的镜头下，SGD才表现出丰富而实际相关的学习阶段，如完全失去稳定性阶段、不正确学习阶段、收敛到低秩鞍点阶段和正确学习阶段。当应用于神经网络时，这些相图意味着具有实际意义的稳定和不稳定区域。

    A fundamental open problem in deep learning theory is how to define and understand the stability of stochastic gradient descent (SGD) close to a fixed point. Conventional literature relies on the convergence of statistical moments, esp., the variance, of the parameters to quantify the stability. We revisit the definition of stability for SGD and use the \textit{convergence in probability} condition to define the \textit{probabilistic stability} of SGD. The proposed stability directly answers a fundamental question in deep learning theory: how SGD selects a meaningful solution for a neural network from an enormous number of solutions that may overfit badly. To achieve this, we show that only under the lens of probabilistic stability does SGD exhibit rich and practically relevant phases of learning, such as the phases of the complete loss of stability, incorrect learning, convergence to low-rank saddles, and correct learning. When applied to a neural network, these phase diagrams imply t
    

