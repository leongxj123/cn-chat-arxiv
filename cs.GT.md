# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamic Pricing and Learning with Long-term Reference Effects](https://arxiv.org/abs/2402.12562) | 在考虑顾客价格期望对当前价格反应的情况下，研究了一种具有长期参考效应的动态定价问题，提出了一种新颖的参考价格机制，展示在该机制下降价政策几乎是最优的，为线性需求模型提供了近似最优降价策略。 |
| [^2] | [Scalable Virtual Valuations Combinatorial Auction Design by Combining Zeroth-Order and First-Order Optimization Method](https://arxiv.org/abs/2402.11904) | 本文提出了一种结合零阶和一阶优化方法，设计了可扩展的虚拟估值组合拍卖，以解决组合候选分配的可缩放性问题。 |
| [^3] | [Smooth Nash Equilibria: Algorithms and Complexity.](http://arxiv.org/abs/2309.12226) | 光滑纳什均衡是纳什均衡的一个松弛变种，可以通过实现与最佳光滑策略的偏离相同的效用来达到。我们定义了强和弱光滑纳什均衡的概念，并证明了它们在计算性质上优于传统的纳什均衡。 |

# 详细

[^1]: 具有长期参考效应的动态定价与学习

    Dynamic Pricing and Learning with Long-term Reference Effects

    [https://arxiv.org/abs/2402.12562](https://arxiv.org/abs/2402.12562)

    在考虑顾客价格期望对当前价格反应的情况下，研究了一种具有长期参考效应的动态定价问题，提出了一种新颖的参考价格机制，展示在该机制下降价政策几乎是最优的，为线性需求模型提供了近似最优降价策略。

    

    我们考虑了一个动态定价问题，其中顾客对当前价格的反应受到顾客价格期望，即参考价格的影响。我们研究了一种简单而新颖的参考价格机制，其中参考价格是卖家过去提供的价格的平均值。与更常见的指数平滑机制相反，在我们的参考价格机制中，卖家提供的价格对未来顾客期望有更长期的影响。我们展示，在这种机制下，降价政策几乎是最优的，不受模型参数的影响。这符合一个常见的直觉，即卖家可以通过以较高的价格出发，然后逐渐降低价格，因为顾客会觉得他们正在购买通常更昂贵的物品上的便宜货。对于线性需求模型，我们还提供了近似最优降价策略的详细特征性描述以及一个有效的方法。

    arXiv:2402.12562v1 Announce Type: new  Abstract: We consider a dynamic pricing problem where customer response to the current price is impacted by the customer price expectation, aka reference price. We study a simple and novel reference price mechanism where reference price is the average of the past prices offered by the seller. As opposed to the more commonly studied exponential smoothing mechanism, in our reference price mechanism the prices offered by seller have a longer term effect on the future customer expectations.   We show that under this mechanism, a markdown policy is near-optimal irrespective of the parameters of the model. This matches the common intuition that a seller may be better off by starting with a higher price and then decreasing it, as the customers feel like they are getting bargains on items that are ordinarily more expensive. For linear demand models, we also provide a detailed characterization of the near-optimal markdown policy along with an efficient way
    
[^2]: 通过结合零阶和一阶优化方法设计可扩展的虚拟估值组合拍卖

    Scalable Virtual Valuations Combinatorial Auction Design by Combining Zeroth-Order and First-Order Optimization Method

    [https://arxiv.org/abs/2402.11904](https://arxiv.org/abs/2402.11904)

    本文提出了一种结合零阶和一阶优化方法，设计了可扩展的虚拟估值组合拍卖，以解决组合候选分配的可缩放性问题。

    

    arXiv:2402.11904v1 公告类型: 交叉论坛 摘要: 自动化拍卖设计旨在利用机器学习发现高收入和激励兼容的机制。确保主导战略激励兼容性（DSIC）至关重要，而最有效的方法是将机制限制在仿射最大化拍卖（AMAs）范围内。然而，现有的基于AMA的方法面临挑战，如可扩展性问题（由组合候选分配导致）和收入的不可微性。在本文中，为了实现可扩展的AMA方法，我们进一步将拍卖机制限制在虚拟估值组合拍卖（VVCAs）范围内，这是具有更少参数的AMAs子集。最初，我们使用可并行化的动态规划算法计算VVCA的获胜分配。随后，我们提出了一种结合了零阶和一阶技术的新型优化方法来优化VVCA参数。

    arXiv:2402.11904v1 Announce Type: cross  Abstract: Automated auction design seeks to discover empirically high-revenue and incentive-compatible mechanisms using machine learning. Ensuring dominant strategy incentive compatibility (DSIC) is crucial, and the most effective approach is to confine the mechanism to Affine Maximizer Auctions (AMAs). Nevertheless, existing AMA-based approaches encounter challenges such as scalability issues (arising from combinatorial candidate allocations) and the non-differentiability of revenue. In this paper, to achieve a scalable AMA-based method, we further restrict the auction mechanism to Virtual Valuations Combinatorial Auctions (VVCAs), a subset of AMAs with significantly fewer parameters. Initially, we employ a parallelizable dynamic programming algorithm to compute the winning allocation of a VVCA. Subsequently, we propose a novel optimization method that combines both zeroth-order and first-order techniques to optimize the VVCA parameters. Extens
    
[^3]: 光滑纳什均衡：算法和复杂性

    Smooth Nash Equilibria: Algorithms and Complexity. (arXiv:2309.12226v1 [cs.GT])

    [http://arxiv.org/abs/2309.12226](http://arxiv.org/abs/2309.12226)

    光滑纳什均衡是纳什均衡的一个松弛变种，可以通过实现与最佳光滑策略的偏离相同的效用来达到。我们定义了强和弱光滑纳什均衡的概念，并证明了它们在计算性质上优于传统的纳什均衡。

    

    纳什均衡的一个基本缺点是其计算复杂性：在正则形式的博弈中，近似纳什均衡是PPAD难的。在本文中，受到平滑分析思想的启发，我们引入了一个被称为$\sigma$-光滑纳什均衡的松弛变种，其中$\sigma$是光滑性参数。在$\sigma$-光滑纳什均衡中，玩家们只需要实现至少与他们最佳$\sigma$-光滑策略的偏离相同的效用，而这个$\sigma$-光滑策略是不会对任何固定动作产生过多质量（根据$\sigma$参数化）。我们区分了两种$\sigma$-光滑纳什均衡的变种：强$\sigma$-光滑纳什均衡，在这种情况下，玩家们需要在均衡中采用$\sigma$-光滑策略进行游戏；弱$\sigma$-光滑纳什均衡中，没有这样的要求。我们证明了无论是弱$\sigma$-光滑纳什均衡还是强$\sigma$-光滑纳什均衡，都比纳什均衡具有更好的计算性质。

    A fundamental shortcoming of the concept of Nash equilibrium is its computational intractability: approximating Nash equilibria in normal-form games is PPAD-hard. In this paper, inspired by the ideas of smoothed analysis, we introduce a relaxed variant of Nash equilibrium called $\sigma$-smooth Nash equilibrium, for a smoothness parameter $\sigma$. In a $\sigma$-smooth Nash equilibrium, players only need to achieve utility at least as high as their best deviation to a $\sigma$-smooth strategy, which is a distribution that does not put too much mass (as parametrized by $\sigma$) on any fixed action. We distinguish two variants of $\sigma$-smooth Nash equilibria: strong $\sigma$-smooth Nash equilibria, in which players are required to play $\sigma$-smooth strategies under equilibrium play, and weak $\sigma$-smooth Nash equilibria, where there is no such requirement.  We show that both weak and strong $\sigma$-smooth Nash equilibria have superior computational properties to Nash equilibri
    

