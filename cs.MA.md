# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Optimal Tax Design in Nonatomic Congestion Games](https://arxiv.org/abs/2402.07437) | 本研究致力于学习如何设计最优税收，以在非原子拥堵博弈中提高效率。为了解决指数级的税收函数空间、梯度不存在和目标函数的非凸性等挑战，该算法利用了分段线性税收、额外的线性项和有效的子例程的新颖组成部分。 |

# 详细

[^1]: 非原子拥堵博弈中学习最优税收设计

    Learning Optimal Tax Design in Nonatomic Congestion Games

    [https://arxiv.org/abs/2402.07437](https://arxiv.org/abs/2402.07437)

    本研究致力于学习如何设计最优税收，以在非原子拥堵博弈中提高效率。为了解决指数级的税收函数空间、梯度不存在和目标函数的非凸性等挑战，该算法利用了分段线性税收、额外的线性项和有效的子例程的新颖组成部分。

    

    本研究探讨了如何学习最优税收设计，以在非原子拥堵博弈中最大化效率。众所周知，玩家之间的自利行为可能会破坏系统的效率。税务机制是缓解此问题并引导社会最优行为的常见方法。在这项工作中，我们首次采取了学习最优税收的初始步骤，该最优税收可以通过平衡反馈来最小化社会成本，即税务设计者只能观察到强制税收下的均衡状态。由于指数级的税收函数空间，梯度不存在和目标函数的非凸性，现有算法不适用。为了解决这些挑战，我们的算法利用了几个新颖的组成部分：（1）分段线性税收来近似最优税收；（2）额外的线性项来保证强凸潜力函数；（3）有效的子例程来找到“边界”税收。该算法可以找到一个$\epsilon$-最优税收，时间复杂度为$O(\bet

    We study how to learn the optimal tax design to maximize the efficiency in nonatomic congestion games. It is known that self-interested behavior among the players can damage the system's efficiency. Tax mechanisms is a common method to alleviate this issue and induce socially optimal behavior. In this work, we take the initial step for learning the optimal tax that can minimize the social cost with \emph{equilibrium feedback}, i.e., the tax designer can only observe the equilibrium state under the enforced tax. Existing algorithms are not applicable due to the exponentially large tax function space, nonexistence of the gradient, and nonconvexity of the objective. To tackle these challenges, our algorithm leverages several novel components: (1) piece-wise linear tax to approximate the optimal tax; (2) an extra linear term to guarantee a strongly convex potential function; (3) efficient subroutine to find the ``boundary'' tax. The algorithm can find an $\epsilon$-optimal tax with $O(\bet
    

