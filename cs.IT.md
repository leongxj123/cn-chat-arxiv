# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Convex Data-Driven Inverse Optimal Control for Nonlinear, Non-stationary and Stochastic Systems.](http://arxiv.org/abs/2306.13928) | 本文提出了一个凸数据驱动逆最优控制方案，能够有效解决非线性、非平稳和随机系统下的成本估计问题。 |
| [^2] | [Pointwise convergence theorem of gradient descent in sparse deep neural network.](http://arxiv.org/abs/2304.08172) | 本文研究了稀疏深度神经网络中梯度下降的点对点收敛定理，针对非光滑指示函数构造了一种特殊形状的DNN，实现了梯度下降过程的点对点收敛。 |

# 详细

[^1]: 针对非线性、非平稳和随机系统的凸数据驱动逆最优控制研究

    On Convex Data-Driven Inverse Optimal Control for Nonlinear, Non-stationary and Stochastic Systems. (arXiv:2306.13928v1 [math.OC])

    [http://arxiv.org/abs/2306.13928](http://arxiv.org/abs/2306.13928)

    本文提出了一个凸数据驱动逆最优控制方案，能够有效解决非线性、非平稳和随机系统下的成本估计问题。

    

    本文主要论述了一个有限时域的逆控制问题，其目的是从观测值中推断出驱动智能体行动的成本，即使这个成本是非凸和非平稳的，同时受到非线性、非平稳和随机因素的影响。在这种情况下，我们提出了一个解决方案，通过解决一个优化问题来实现成本估计，即使代理成本不是凸的，本文也能够生成凸问题。为了得出这个结果，我们还研究了一个以随机策略为决策变量的有限时域前向控制问题，并给出了最优解的显式表达式。此外，我们将我们的发现转化为算法流程，并通过虚拟实验和真实硬件实验验证了我们的方法的有效性。所有的实验结果都证实了我们方法的有效性。

    This paper is concerned with a finite-horizon inverse control problem, which has the goal of inferring, from observations, the possibly non-convex and non-stationary cost driving the actions of an agent. In this context, we present a result that enables cost estimation by solving an optimization problem that is convex even when the agent cost is not and when the underlying dynamics is nonlinear, non-stationary and stochastic. To obtain this result, we also study a finite-horizon forward control problem that has randomized policies as decision variables. For this problem, we give an explicit expression for the optimal solution. Moreover, we turn our findings into algorithmic procedures and we show the effectiveness of our approach via both in-silico and experimental validations with real hardware. All the experiments confirm the effectiveness of our approach.
    
[^2]: 稀疏深度神经网络中梯度下降的点对点收敛定理

    Pointwise convergence theorem of gradient descent in sparse deep neural network. (arXiv:2304.08172v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2304.08172](http://arxiv.org/abs/2304.08172)

    本文研究了稀疏深度神经网络中梯度下降的点对点收敛定理，针对非光滑指示函数构造了一种特殊形状的DNN，实现了梯度下降过程的点对点收敛。

    

    深度神经网络（DNN）的理论结构逐渐得到了阐明。Imaizumi-Fukumizu（2019）和Suzuki（2019）指出，当目标函数为非光滑函数时，DNN的学习能力优于先前的理论。然而，据作者所知，迄今为止的众多研究尝试在没有任何统计论证的情况下进行数学研究，探究真正能够引发梯度下降的DNN架构的点对点收敛性，这一尝试似乎更贴近实际DNN。本文将目标函数限制为非光滑指示函数，并在ReLU-DNN中构造了一个稀疏且具有特殊形状的DNN，从而实现了梯度下降过程中的点对点收敛。

    The theoretical structure of deep neural network (DNN) has been clarified gradually. Imaizumi-Fukumizu (2019) and Suzuki (2019) clarified that the learning ability of DNN is superior to the previous theories when the target function is non-smooth functions. However, as far as the author is aware, none of the numerous works to date attempted to mathematically investigate what kind of DNN architectures really induce pointwise convergence of gradient descent (without any statistical argument), and this attempt seems to be closer to the practical DNNs. In this paper we restrict target functions to non-smooth indicator functions, and construct a deep neural network inducing pointwise convergence provided by gradient descent process in ReLU-DNN. The DNN has a sparse and a special shape, with certain variable transformations.
    

