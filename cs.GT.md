# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics.](http://arxiv.org/abs/2301.13306) | 本文提出了一个基于梯度的学习算法，可以在多种拍卖方式下满足预算和ROI约束，并达到个体后悔逐渐减小；结果表明，当各自竞争时，期望资金流动至少达到最优分配的期望流动的一半。 |

# 详细

[^1]: 带有预算和ROI约束的自动出价算法：效率、后悔和节奏动态

    Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics. (arXiv:2301.13306v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2301.13306](http://arxiv.org/abs/2301.13306)

    本文提出了一个基于梯度的学习算法，可以在多种拍卖方式下满足预算和ROI约束，并达到个体后悔逐渐减小；结果表明，当各自竞争时，期望资金流动至少达到最优分配的期望流动的一半。

    

    我们研究了自动出价算法在在线广告平台上进行博弈的情况。每个自动出价算法被赋予任务，在多轮重复拍卖中，最大化其广告主的总价值，同时受到预算和/或投资回报率约束。我们提出了一种基于梯度的学习算法，它可以保证满足所有约束条件，并达到逐渐减小的个体后悔。我们的算法仅使用自助反馈，并可与第一或第二价格拍卖以及任何“中间”拍卖格式一起使用。我们的主要结果是，当这些自动出价算法相互竞争时，所有轮次的期望资金流动 welfare 都至少达到了任何分配所实现的期望最优流动 welfare 的一半。这在出价动态是否收敛到均衡以及广告主估值之间的相关结构如何不同的情况下均成立。

    We study a game between autobidding algorithms that compete in an online advertising platform. Each autobidder is tasked with maximizing its advertiser's total value over multiple rounds of a repeated auction, subject to budget and/or return-on-investment constraints. We propose a gradient-based learning algorithm that is guaranteed to satisfy all constraints and achieves vanishing individual regret. Our algorithm uses only bandit feedback and can be used with the first- or second-price auction, as well as with any "intermediate" auction format. Our main result is that when these autobidders play against each other, the resulting expected liquid welfare over all rounds is at least half of the expected optimal liquid welfare achieved by any allocation. This holds whether or not the bidding dynamics converges to an equilibrium and regardless of the correlation structure between advertiser valuations.
    

