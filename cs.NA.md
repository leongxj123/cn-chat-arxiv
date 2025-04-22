# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options](https://arxiv.org/abs/2403.02832) | 本研究提出使用随机化拟蒙特卡洛积分来提高傅里叶方法在高维情境下的可扩展性，以解决高效定价多资产期权的挑战。 |
| [^2] | [Enhanced Adaptive Gradient Algorithms for Nonconvex-PL Minimax Optimization.](http://arxiv.org/abs/2303.03984) | 本文提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决非凸-PL极小极大问题，其中AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。 |

# 详细

[^1]: 高效傅里叶定价多资产期权的拟蒙特卡洛方法

    Quasi-Monte Carlo for Efficient Fourier Pricing of Multi-Asset Options

    [https://arxiv.org/abs/2403.02832](https://arxiv.org/abs/2403.02832)

    本研究提出使用随机化拟蒙特卡洛积分来提高傅里叶方法在高维情境下的可扩展性，以解决高效定价多资产期权的挑战。

    

    在定量金融中，高效定价多资产期权是一个重要挑战。蒙特卡洛（MC）方法仍然是定价引擎的主要选择；然而，其收敛速度慢阻碍了其实际应用。傅里叶方法利用特征函数的知识，准确快速地估值多达两个资产的期权。然而，在高维设置中，由于常用的积分技术具有张量积（TP）结构，它们面临障碍。本文主张使用随机化拟蒙特卡洛（RQMC）积分来改善高维傅里叶方法的可扩展性。RQMC技术受益于被积函数的光滑性，缓解了维度灾难，同时提供了实用的误差估计。然而，RQMC在无界域$\mathbb{R}^d$上的适用性需要将域转换为$[0,1]^d$，这可能...

    arXiv:2403.02832v1 Announce Type: new  Abstract: Efficiently pricing multi-asset options poses a significant challenge in quantitative finance. The Monte Carlo (MC) method remains the prevalent choice for pricing engines; however, its slow convergence rate impedes its practical application. Fourier methods leverage the knowledge of the characteristic function to accurately and rapidly value options with up to two assets. Nevertheless, they face hurdles in the high-dimensional settings due to the tensor product (TP) structure of commonly employed quadrature techniques. This work advocates using the randomized quasi-MC (RQMC) quadrature to improve the scalability of Fourier methods with high dimensions. The RQMC technique benefits from the smoothness of the integrand and alleviates the curse of dimensionality while providing practical error estimates. Nonetheless, the applicability of RQMC on the unbounded domain, $\mathbb{R}^d$, requires a domain transformation to $[0,1]^d$, which may r
    
[^2]: 非凸-PL极小极大优化的增强自适应梯度算法

    Enhanced Adaptive Gradient Algorithms for Nonconvex-PL Minimax Optimization. (arXiv:2303.03984v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2303.03984](http://arxiv.org/abs/2303.03984)

    本文提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决非凸-PL极小极大问题，其中AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。

    This paper proposes a class of enhanced momentum-based gradient descent ascent methods (MSGDA and AdaMSGDA) to solve nonconvex-PL minimax problems, where the AdaMSGDA algorithm can use various adaptive learning rates to update variables x and y without relying on any global and coordinate-wise adaptive learning rates. Theoretical analysis shows that MSGDA and AdaMSGDA methods have the best known sample (gradient) complexity of O(ε−3) in finding an ε-stationary solution.

    本文研究了一类非凸非凹的极小极大优化问题（即$\min_x\max_y f(x,y)$），其中$f(x,y)$在$x$上可能是非凸的，在$y$上是非凹的，并满足Polyak-Lojasiewicz（PL）条件。此外，我们提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决这些随机非凸-PL极小极大问题。特别地，我们的AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们提出了一种有效的收敛分析框架来解决我们的方法。具体而言，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解（即$\mathbb{E}\|\nabla F(x)\|\leq \epsilon$，其中$F(x)=\max_y f(x,y)$）时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。

    In the paper, we study a class of nonconvex nonconcave minimax optimization problems (i.e., $\min_x\max_y f(x,y)$), where $f(x,y)$ is possible nonconvex in $x$, and it is nonconcave and satisfies the Polyak-Lojasiewicz (PL) condition in $y$. Moreover, we propose a class of enhanced momentum-based gradient descent ascent methods (i.e., MSGDA and AdaMSGDA) to solve these stochastic Nonconvex-PL minimax problems. In particular, our AdaMSGDA algorithm can use various adaptive learning rates in updating the variables $x$ and $y$ without relying on any global and coordinate-wise adaptive learning rates. Theoretically, we present an effective convergence analysis framework for our methods. Specifically, we prove that our MSGDA and AdaMSGDA methods have the best known sample (gradient) complexity of $O(\epsilon^{-3})$ only requiring one sample at each loop in finding an $\epsilon$-stationary solution (i.e., $\mathbb{E}\|\nabla F(x)\|\leq \epsilon$, where $F(x)=\max_y f(x,y)$). This manuscript 
    

