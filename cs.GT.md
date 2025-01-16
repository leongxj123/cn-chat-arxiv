# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Optimal Tax Design in Nonatomic Congestion Games](https://arxiv.org/abs/2402.07437) | 本研究致力于学习如何设计最优税收，以在非原子拥堵博弈中提高效率。为了解决指数级的税收函数空间、梯度不存在和目标函数的非凸性等挑战，该算法利用了分段线性税收、额外的线性项和有效的子例程的新颖组成部分。 |
| [^2] | [Probabilistic Verification in Mechanism Design.](http://arxiv.org/abs/1908.05556) | 该论文介绍了机制设计中的概率验证模型，通过选择统计测试来验证代理人的声明，并判断是否有最佳测试筛选所有其他类型。这个方法能够利润最大化并不断插值解决验证的问题。 |

# 详细

[^1]: 非原子拥堵博弈中学习最优税收设计

    Learning Optimal Tax Design in Nonatomic Congestion Games

    [https://arxiv.org/abs/2402.07437](https://arxiv.org/abs/2402.07437)

    本研究致力于学习如何设计最优税收，以在非原子拥堵博弈中提高效率。为了解决指数级的税收函数空间、梯度不存在和目标函数的非凸性等挑战，该算法利用了分段线性税收、额外的线性项和有效的子例程的新颖组成部分。

    

    本研究探讨了如何学习最优税收设计，以在非原子拥堵博弈中最大化效率。众所周知，玩家之间的自利行为可能会破坏系统的效率。税务机制是缓解此问题并引导社会最优行为的常见方法。在这项工作中，我们首次采取了学习最优税收的初始步骤，该最优税收可以通过平衡反馈来最小化社会成本，即税务设计者只能观察到强制税收下的均衡状态。由于指数级的税收函数空间，梯度不存在和目标函数的非凸性，现有算法不适用。为了解决这些挑战，我们的算法利用了几个新颖的组成部分：（1）分段线性税收来近似最优税收；（2）额外的线性项来保证强凸潜力函数；（3）有效的子例程来找到“边界”税收。该算法可以找到一个$\epsilon$-最优税收，时间复杂度为$O(\bet

    We study how to learn the optimal tax design to maximize the efficiency in nonatomic congestion games. It is known that self-interested behavior among the players can damage the system's efficiency. Tax mechanisms is a common method to alleviate this issue and induce socially optimal behavior. In this work, we take the initial step for learning the optimal tax that can minimize the social cost with \emph{equilibrium feedback}, i.e., the tax designer can only observe the equilibrium state under the enforced tax. Existing algorithms are not applicable due to the exponentially large tax function space, nonexistence of the gradient, and nonconvexity of the objective. To tackle these challenges, our algorithm leverages several novel components: (1) piece-wise linear tax to approximate the optimal tax; (2) an extra linear term to guarantee a strongly convex potential function; (3) efficient subroutine to find the ``boundary'' tax. The algorithm can find an $\epsilon$-optimal tax with $O(\bet
    
[^2]: 机制设计中的概率验证

    Probabilistic Verification in Mechanism Design. (arXiv:1908.05556v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/1908.05556](http://arxiv.org/abs/1908.05556)

    该论文介绍了机制设计中的概率验证模型，通过选择统计测试来验证代理人的声明，并判断是否有最佳测试筛选所有其他类型。这个方法能够利润最大化并不断插值解决验证的问题。

    

    我们在机制设计背景下引入了一个概率验证模型。委托人选择一个统计测试来验证代理人的声明。代理人的真实类型决定了他可以通过每个测试的概率。我们刻画了每个类型是否有一个相关的测试，最好地筛选出所有其他类型，无论社会选择规则如何。如果这个条件成立，那么测试技术可以用一个易于处理的简化形式表示。我们使用这个简化形式来解决验证的利润最大化机制。随着验证的改进，解决方案从无验证解决方案到完全剩余提取不断插值。

    We introduce a model of probabilistic verification in a mechanism design setting. The principal selects a statistical test to verify the agent's claim. The agent's true type determines the probability with which he can pass each test. We characterize whether each type has an associated test that best screens out all other types, no matter the social choice rule. If this condition holds, then the testing technology can be represented in a tractable reduced form. We use this reduced form to solve for profit-maximizing mechanisms with verification. As verification improves, the solution continuously interpolates from the no-verification solution to full surplus extraction.
    

