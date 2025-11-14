# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spectral methods for Neural Integral Equations](https://arxiv.org/abs/2312.05654) | 本文引入了一个基于谱方法的神经积分方程框架，通过在谱域中学习算子，降低了计算成本，并保证了高插值精度。 |
| [^2] | [Unbiased estimators for the Heston model with stochastic interest rates.](http://arxiv.org/abs/2301.12072) | 本研究结合了无偏估计器和具有随机利率的Heston模型，通过开发半精确的对数欧拉方案，证明了其收敛率为O(h)，适用于多种模型。 |

# 详细

[^1]: 神经积分方程的谱方法

    Spectral methods for Neural Integral Equations

    [https://arxiv.org/abs/2312.05654](https://arxiv.org/abs/2312.05654)

    本文引入了一个基于谱方法的神经积分方程框架，通过在谱域中学习算子，降低了计算成本，并保证了高插值精度。

    

    arXiv:2312.05654v3 公告类型：替换-跨交摘要：神经积分方程是基于积分方程理论的深度学习模型，其中模型由积分算子和通过优化过程学习的相应方程（第二种）组成。这种方法允许利用机器学习中积分算子的非局部特性，但计算成本很高。在本文中，我们介绍了基于谱方法的神经积分方程框架，该方法使我们能够在谱域中学习一个算子，从而降低计算成本，同时保证高插值精度。我们研究了我们方法的性质，并展示了关于模型近似能力和收敛到数值方法解的各种理论保证。我们提供了数值实验来展示所得模型的实际有效性。

    arXiv:2312.05654v3 Announce Type: replace-cross  Abstract: Neural integral equations are deep learning models based on the theory of integral equations, where the model consists of an integral operator and the corresponding equation (of the second kind) which is learned through an optimization procedure. This approach allows to leverage the nonlocal properties of integral operators in machine learning, but it is computationally expensive. In this article, we introduce a framework for neural integral equations based on spectral methods that allows us to learn an operator in the spectral domain, resulting in a cheaper computational cost, as well as in high interpolation accuracy. We study the properties of our methods and show various theoretical guarantees regarding the approximation capabilities of the model, and convergence to solutions of the numerical methods. We provide numerical experiments to demonstrate the practical effectiveness of the resulting model.
    
[^2]: 具有随机利率的Heston模型的无偏估计器

    Unbiased estimators for the Heston model with stochastic interest rates. (arXiv:2301.12072v2 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2301.12072](http://arxiv.org/abs/2301.12072)

    本研究结合了无偏估计器和具有随机利率的Heston模型，通过开发半精确的对数欧拉方案，证明了其收敛率为O(h)，适用于多种模型。

    

    我们结合了Rhee和Glynn（Operations Research: 63(5), 1026-1043，2015）中的无偏估计器和具有随机利率的Heston模型。具体地，我们首先为具有随机利率的Heston模型开发了一个半精确的对数欧拉方案。然后，在一些温和的假设下，我们证明收敛率在L^2范数中是O(h)，其中h是步长。该结果适用于许多模型，如Heston-Hull-While模型，Heston-CIR模型和Heston-Black-Karasinski模型。数值实验支持我们的理论收敛率。

    We combine the unbiased estimators in Rhee and Glynn (Operations Research: 63(5), 1026-1043, 2015) and the Heston model with stochastic interest rates. Specifically, we first develop a semi-exact log-Euler scheme for the Heston model with stochastic interest rates. Then, under mild assumptions, we show that the convergence rate in the $L^2$ norm is $O(h)$, where $h$ is the step size. The result applies to a large class of models, such as the Heston-Hull-While model, the Heston-CIR model and the Heston-Black-Karasinski model. Numerical experiments support our theoretical convergence rate.
    

