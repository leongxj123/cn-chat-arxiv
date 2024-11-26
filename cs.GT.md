# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Assignment Game: New Mechanisms for Equitable Core Imputations](https://arxiv.org/abs/2402.11437) | 本论文提出了一种计算分配博弈的更加公平核分配的组合多项式时间机制。 |
| [^2] | [Persuading a Learning Agent](https://arxiv.org/abs/2402.09721) | 在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。 |
| [^3] | [Rethinking Scaling Laws for Learning in Strategic Environments](https://arxiv.org/abs/2402.07588) | 本文重新思考了在战略环境中学习的比例定律，发现战略互动可以打破传统的观点，即模型越大或表达能力越强并不一定会随之提高性能。通过几个战略环境的例子，我们展示了这种现象的影响。 |
| [^4] | [Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics.](http://arxiv.org/abs/2301.13306) | 本文提出了一个基于梯度的学习算法，可以在多种拍卖方式下满足预算和ROI约束，并达到个体后悔逐渐减小；结果表明，当各自竞争时，期望资金流动至少达到最优分配的期望流动的一半。 |

# 详细

[^1]: 《分配博弈：公平核分配的新机制》

    The Assignment Game: New Mechanisms for Equitable Core Imputations

    [https://arxiv.org/abs/2402.11437](https://arxiv.org/abs/2402.11437)

    本论文提出了一种计算分配博弈的更加公平核分配的组合多项式时间机制。

    

    《分配博弈》的核分配集形成一个（非有限）分配格。迄今为止，仅已知有效算法用于计算其两个极端分配；但是，其中每一个都最大程度地偏袒一个方，不利于双方的分配，导致盈利不均衡。另一个问题是，由一个玩家组成的子联盟（或者来自分配两侧的一系玩家）可以获得零利润，因此核分配不必给予他们任何利润。因此，核分配在个体代理人层面上不提供任何公平性保证。这引出一个问题，即如何计算更公平的核分配。在本文中，我们提出了一个计算分配博弈的Leximin和Leximax核分配的组合（即，该机制不涉及LP求解器）多项式时间机制。这些分配以不同方式实现了“公平性”：

    arXiv:2402.11437v1 Announce Type: cross  Abstract: The set of core imputations of the assignment game forms a (non-finite) distributive lattice. So far, efficient algorithms were known for computing only its two extreme imputations; however, each of them maximally favors one side and disfavors the other side of the bipartition, leading to inequitable profit sharing. Another issue is that a sub-coalition consisting of one player (or a set of players from the same side of the bipartition) can make zero profit, therefore a core imputation is not obliged to give them any profit. Hence core imputations make no fairness guarantee at the level of individual agents. This raises the question of computing {\em more equitable core imputations}.   In this paper, we give combinatorial (i.e., the mechanism does not invoke an LP-solver) polynomial time mechanisms for computing the leximin and leximax core imputations for the assignment game. These imputations achieve ``fairness'' in different ways: w
    
[^2]: 说服一位学习代理

    Persuading a Learning Agent

    [https://arxiv.org/abs/2402.09721](https://arxiv.org/abs/2402.09721)

    在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。

    

    我们研究了一个重复的贝叶斯说服问题（更一般地，任何具有完全信息的广义委托-代理问题），其中委托人没有承诺能力，代理人使用算法来学习如何对委托人的信号做出响应。我们将这个问题简化为一个一次性的广义委托-代理问题，代理人近似地最佳响应。通过这个简化，我们可以证明：如果代理人使用上下文无遗憾学习算法，则委托人可以保证其效用与经典无学习模型中具有承诺的委托人的最优效用之间可以无限接近；如果代理人使用上下文无交换遗憾学习算法，则委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。委托人在学习模型与非学习模型中可以获得的效用之间的差距是有界的。

    arXiv:2402.09721v1 Announce Type: cross  Abstract: We study a repeated Bayesian persuasion problem (and more generally, any generalized principal-agent problem with complete information) where the principal does not have commitment power and the agent uses algorithms to learn to respond to the principal's signals. We reduce this problem to a one-shot generalized principal-agent problem with an approximately-best-responding agent. This reduction allows us to show that: if the agent uses contextual no-regret learning algorithms, then the principal can guarantee a utility that is arbitrarily close to the principal's optimal utility in the classic non-learning model with commitment; if the agent uses contextual no-swap-regret learning algorithms, then the principal cannot obtain any utility significantly more than the optimal utility in the non-learning model with commitment. The difference between the principal's obtainable utility in the learning model and the non-learning model is bound
    
[^3]: 重新思考战略环境中学习的比例定律

    Rethinking Scaling Laws for Learning in Strategic Environments

    [https://arxiv.org/abs/2402.07588](https://arxiv.org/abs/2402.07588)

    本文重新思考了在战略环境中学习的比例定律，发现战略互动可以打破传统的观点，即模型越大或表达能力越强并不一定会随之提高性能。通过几个战略环境的例子，我们展示了这种现象的影响。

    

    越来越大的机器学习模型的部署反映出一个共识：模型越有表达能力，越拥有大量数据，就能改善性能。随着模型在各种真实场景中的部署，它们不可避免地面临着战略环境。本文考虑了模型与战略互动对比例定律的相互作用对性能的影响这个自然问题。我们发现战略互动可以打破传统的比例定律观点，即性能并不一定随着模型的扩大和/或表达能力的增强（即使有无限数据）而单调提高。我们通过战略回归、战略分类和多智能体强化学习的例子展示了这一现象的影响，这些例子展示了战略环境中的限制模型或策略类的表达能力即可。

    The deployment of ever-larger machine learning models reflects a growing consensus that the more expressive the model$\unicode{x2013}$and the more data one has access to$\unicode{x2013}$the more one can improve performance. As models get deployed in a variety of real world scenarios, they inevitably face strategic environments. In this work, we consider the natural question of how the interplay of models and strategic interactions affects scaling laws. We find that strategic interactions can break the conventional view of scaling laws$\unicode{x2013}$meaning that performance does not necessarily monotonically improve as models get larger and/ or more expressive (even with infinite data). We show the implications of this phenomenon in several contexts including strategic regression, strategic classification, and multi-agent reinforcement learning through examples of strategic environments in which$\unicode{x2013}$by simply restricting the expressivity of one's model or policy class$\uni
    
[^4]: 带有预算和ROI约束的自动出价算法：效率、后悔和节奏动态

    Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics. (arXiv:2301.13306v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2301.13306](http://arxiv.org/abs/2301.13306)

    本文提出了一个基于梯度的学习算法，可以在多种拍卖方式下满足预算和ROI约束，并达到个体后悔逐渐减小；结果表明，当各自竞争时，期望资金流动至少达到最优分配的期望流动的一半。

    

    我们研究了自动出价算法在在线广告平台上进行博弈的情况。每个自动出价算法被赋予任务，在多轮重复拍卖中，最大化其广告主的总价值，同时受到预算和/或投资回报率约束。我们提出了一种基于梯度的学习算法，它可以保证满足所有约束条件，并达到逐渐减小的个体后悔。我们的算法仅使用自助反馈，并可与第一或第二价格拍卖以及任何“中间”拍卖格式一起使用。我们的主要结果是，当这些自动出价算法相互竞争时，所有轮次的期望资金流动 welfare 都至少达到了任何分配所实现的期望最优流动 welfare 的一半。这在出价动态是否收敛到均衡以及广告主估值之间的相关结构如何不同的情况下均成立。

    We study a game between autobidding algorithms that compete in an online advertising platform. Each autobidder is tasked with maximizing its advertiser's total value over multiple rounds of a repeated auction, subject to budget and/or return-on-investment constraints. We propose a gradient-based learning algorithm that is guaranteed to satisfy all constraints and achieves vanishing individual regret. Our algorithm uses only bandit feedback and can be used with the first- or second-price auction, as well as with any "intermediate" auction format. Our main result is that when these autobidders play against each other, the resulting expected liquid welfare over all rounds is at least half of the expected optimal liquid welfare achieved by any allocation. This holds whether or not the bidding dynamics converges to an equilibrium and regardless of the correlation structure between advertiser valuations.
    

