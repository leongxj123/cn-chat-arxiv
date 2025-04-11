# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bandit Social Learning: Exploration under Myopic Behavior.](http://arxiv.org/abs/2302.07425) | 该论文研究了自私行为下的劫匪社交学习问题，发现存在一种探索激励权衡，即武器探索和社交探索之间的权衡，受到代理的短视行为的限制会加剧这种权衡，并导致遗憾率与代理数量成线性关系。 |

# 详细

[^1]: 自私行为下的劫匪社交学习

    Bandit Social Learning: Exploration under Myopic Behavior. (arXiv:2302.07425v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.07425](http://arxiv.org/abs/2302.07425)

    该论文研究了自私行为下的劫匪社交学习问题，发现存在一种探索激励权衡，即武器探索和社交探索之间的权衡，受到代理的短视行为的限制会加剧这种权衡，并导致遗憾率与代理数量成线性关系。

    

    我们研究了一种社交学习动态，其中代理按照简单的多臂劫匪协议共同行动。代理以顺序方式到达，选择武器并接收相关奖励。每个代理观察先前代理的完整历史记录（武器和奖励），不存在私有信号。尽管代理共同面临开发和利用的探索折衷，但每个代理人都是一见钟情的，无需考虑探索。我们允许一系列与（参数化）置信区间一致的自私行为，包括“无偏”行为和各种行为偏差。虽然这些行为的极端版本对应于众所周知的劫匪算法，但我们证明了更温和的版本会导致明显的探索失败，因此遗憾率与代理数量成线性关系。我们通过分析“温和乐观”的代理提供匹配的遗憾上界。因此，我们建立了两种类型的探索激励之间的基本权衡：武器探索是固有于劫匪问题的，只受当前代理的行动影响，而社交探索是由先前代理行为驱动的，因此有利于未来代理。由于代理的短视行为限制了社交探索，这种权衡被加剧。

    We study social learning dynamics where the agents collectively follow a simple multi-armed bandit protocol. Agents arrive sequentially, choose arms and receive associated rewards. Each agent observes the full history (arms and rewards) of the previous agents, and there are no private signals. While collectively the agents face exploration-exploitation tradeoff, each agent acts myopically, without regards to exploration. Motivating scenarios concern reviews and ratings on online platforms.  We allow a wide range of myopic behaviors that are consistent with (parameterized) confidence intervals, including the "unbiased" behavior as well as various behaviorial biases. While extreme versions of these behaviors correspond to well-known bandit algorithms, we prove that more moderate versions lead to stark exploration failures, and consequently to regret rates that are linear in the number of agents. We provide matching upper bounds on regret by analyzing "moderately optimistic" agents.  As a
    

