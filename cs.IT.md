# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Families of costs with zero and nonnegative MTW tensor in optimal transport.](http://arxiv.org/abs/2401.00953) | 这篇论文介绍了在最优传送中使用形式为$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数时的零和非负MTW张量的计算方法，并提供了MTW张量在零向量上为零的条件以及相应的线性ODE的简化方法。此外，还给出了逆函数的解析表达式以及一些具体的应用情况。 |
| [^2] | [Scheduling and Aggregation Design for Asynchronous Federated Learning over Wireless Networks.](http://arxiv.org/abs/2212.07356) | 本文提出了一种异步联邦学习的调度策略和聚合加权设计，通过采用基于信道感知数据重要性的调度策略和“年龄感知”的聚合加权设计来解决FL系统中的“拖沓”问题，并通过仿真证实了其有效性。 |

# 详细

[^1]: 拥有零和非负MTW张量的费用族在最优传送中的应用

    Families of costs with zero and nonnegative MTW tensor in optimal transport. (arXiv:2401.00953v1 [math.AP])

    [http://arxiv.org/abs/2401.00953](http://arxiv.org/abs/2401.00953)

    这篇论文介绍了在最优传送中使用形式为$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数时的零和非负MTW张量的计算方法，并提供了MTW张量在零向量上为零的条件以及相应的线性ODE的简化方法。此外，还给出了逆函数的解析表达式以及一些具体的应用情况。

    

    我们计算了在$\mathbb{R}^n$上具有形式$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数的最优传送问题的MTW张量（或交叉曲率）。其中，$\mathsf{u}$是一个具有逆函数$\mathsf{s}$的标量函数，$x^{\ft}y$是属于$\mathbb{R}^n$开子集的向量$x，y$的非退化双线性配对。MTW张量在Kim-McCann度量下对于零向量的条件是一个四阶非线性ODE，可以被简化为具有常数系数$P$和$S$的形式为$\mathsf{s}^{(2)} - S\mathsf{s}^{(1)} + P\mathsf{s} = 0$的线性ODE。最终得到的逆函数包括Lambert和广义反双曲/三角函数。平方欧氏度量和$\log$型费用是这些解的实例。这个家族的最优映射也是显式的。

    We compute explicitly the MTW tensor (or cross curvature) for the optimal transport problem on $\mathbb{R}^n$ with a cost function of form $\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$, where $\mathsf{u}$ is a scalar function with inverse $\mathsf{s}$, $x^{\ft}y$ is a nondegenerate bilinear pairing of vectors $x, y$ belonging to an open subset of $\mathbb{R}^n$. The condition that the MTW-tensor vanishes on null vectors under the Kim-McCann metric is a fourth-order nonlinear ODE, which could be reduced to a linear ODE of the form $\mathsf{s}^{(2)} - S\mathsf{s}^{(1)} + P\mathsf{s} = 0$ with constant coefficients $P$ and $S$. The resulting inverse functions include {\it Lambert} and {\it generalized inverse hyperbolic\slash trigonometric} functions. The square Euclidean metric and $\log$-type costs are equivalent to instances of these solutions. The optimal map for the family is also explicit. For cost functions of a similar form on a hyperboloid model of the hyperbolic space and u
    
[^2]: 异步联邦学习在无线网络中的调度和聚合设计

    Scheduling and Aggregation Design for Asynchronous Federated Learning over Wireless Networks. (arXiv:2212.07356v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.07356](http://arxiv.org/abs/2212.07356)

    本文提出了一种异步联邦学习的调度策略和聚合加权设计，通过采用基于信道感知数据重要性的调度策略和“年龄感知”的聚合加权设计来解决FL系统中的“拖沓”问题，并通过仿真证实了其有效性。

    

    联邦学习（FL）是一种协作的机器学习（ML）框架，它结合了设备上的训练和基于服务器的聚合来在分布式代理间训练通用的ML模型。本文中，我们提出了一种异步FL设计，采用周期性的聚合来解决FL系统中的“拖沓”问题。考虑到有限的无线通信资源，我们研究了不同调度策略和聚合设计对收敛性能的影响。基于降低聚合模型更新的偏差和方差的重要性，我们提出了一个调度策略，它同时考虑了用户设备的信道质量和训练数据表示。通过仿真验证了我们的基于信道感知数据重要性的调度策略相对于同步联邦学习提出的现有最新方法的有效性。此外，我们还展示了一种“年龄感知”的聚合加权设计可以显著提高学习性能。

    Federated Learning (FL) is a collaborative machine learning (ML) framework that combines on-device training and server-based aggregation to train a common ML model among distributed agents. In this work, we propose an asynchronous FL design with periodic aggregation to tackle the straggler issue in FL systems. Considering limited wireless communication resources, we investigate the effect of different scheduling policies and aggregation designs on the convergence performance. Driven by the importance of reducing the bias and variance of the aggregated model updates, we propose a scheduling policy that jointly considers the channel quality and training data representation of user devices. The effectiveness of our channel-aware data-importance-based scheduling policy, compared with state-of-the-art methods proposed for synchronous FL, is validated through simulations. Moreover, we show that an ``age-aware'' aggregation weighting design can significantly improve the learning performance i
    

