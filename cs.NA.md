# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Shallow ReLU neural networks and finite elements](https://arxiv.org/abs/2403.05809) | 在凸多面体网格上，提出了用两个隐藏层的ReLU神经网络来弱表示分段线性函数，并根据网格中的多面体和超平面的数量准确确定了所需的神经元数，建立了浅层ReLU神经网络和有限元函数之间的联系。 |
| [^2] | [On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit](https://arxiv.org/abs/2402.06388) | 该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。 |

# 详细

[^1]: 浅层ReLU神经网络和有限元

    Shallow ReLU neural networks and finite elements

    [https://arxiv.org/abs/2403.05809](https://arxiv.org/abs/2403.05809)

    在凸多面体网格上，提出了用两个隐藏层的ReLU神经网络来弱表示分段线性函数，并根据网格中的多面体和超平面的数量准确确定了所需的神经元数，建立了浅层ReLU神经网络和有限元函数之间的联系。

    

    我们指出在凸多面体网格上，可以用两个隐藏层的ReLU神经网络在弱意义下表示（连续或不连续的）分段线性函数。此外，基于涉及到的多面体和超平面的数量，准确给出了弱表示所需的两个隐藏层的神经元数。这些结果自然地适用于常数和线性有限元函数。这种弱表示建立了浅层ReLU神经网络和有限元函数之间的桥梁，并为通过有限元函数分析ReLU神经网络在$L^p$范数中的逼近能力提供了视角。此外，我们还讨论了最近张量神经网络对张量有限元函数的严格表示。

    arXiv:2403.05809v1 Announce Type: cross  Abstract: We point out that (continuous or discontinuous) piecewise linear functions on a convex polytope mesh can be represented by two-hidden-layer ReLU neural networks in a weak sense. In addition, the numbers of neurons of the two hidden layers required to weakly represent are accurately given based on the numbers of polytopes and hyperplanes involved in this mesh. The results naturally hold for constant and linear finite element functions. Such weak representation establishes a bridge between shallow ReLU neural networks and finite element functions, and leads to a perspective for analyzing approximation capability of ReLU neural networks in $L^p$ norm via finite element functions. Moreover, we discuss the strict representation for tensor finite element functions via the recent tensor neural networks.
    
[^2]: 关于随机梯度下降（SGD）的收敛速度及其在修改的多臂赌博机上的策略梯度应用

    On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit

    [https://arxiv.org/abs/2402.06388](https://arxiv.org/abs/2402.06388)

    该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。

    

    我们提出了一个自包含的证明，证明了当学习速率遵循逆时间衰减规则时，随机梯度下降（SGD）的收敛速度；接下来，我们将这些结果应用于带有L2正则化的修改的策略梯度多臂赌博机（MAB）的收敛性分析。

    We present a self-contained proof of the convergence rate of the Stochastic Gradient Descent (SGD) when the learning rate follows an inverse time decays schedule; we next apply the results to the convergence of a modified form of policy gradient Multi-Armed Bandit (MAB) with $L2$ regularization.
    

