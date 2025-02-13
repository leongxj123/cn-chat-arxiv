# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Single-token vs Two-token Blockchain Tokenomics](https://arxiv.org/abs/2403.15429) | 论文研究了由用户和验证者组成的PoS区块链系统中的令牌经济设计，探讨了系统服务提供与适当奖励方案如何共同导致具有可取特征的均衡状态。 |
| [^2] | [am-AMM: An Auction-Managed Automated Market Maker](https://arxiv.org/abs/2403.03367) | 本文提出了一个针对AMMs两个重要未解决问题的单一机制：减少对有信息的订单流的损失，最大化对无信息的订单流的收益。 |
| [^3] | [Algorithmic Persuasion Through Simulation](https://arxiv.org/abs/2311.18138) | 通过模拟接收者行为的贝叶斯劝导问题中，发送者设计了一个最优消息策略并设计了一个多项式时间查询算法，以优化其预期效用。 |

# 详细

[^1]: 单令牌 vs 双令牌区块链经济模型

    Single-token vs Two-token Blockchain Tokenomics

    [https://arxiv.org/abs/2403.15429](https://arxiv.org/abs/2403.15429)

    论文研究了由用户和验证者组成的PoS区块链系统中的令牌经济设计，探讨了系统服务提供与适当奖励方案如何共同导致具有可取特征的均衡状态。

    

    我们考虑了由用户和验证者组成的PoS区块链系统的令牌经济设计中产生的长期均衡，两者都致力于最大化自己的效用。验证者是系统维护者，他们通过执行系统正常运行所需的工作来获得令牌作为奖励，而用户则通过支付这些令牌来获取所需的系统服务水平。我们研究了系统服务提供和适当奖励方案如何共同导致具有可取特征的均衡状态：（1）可持续性：系统保持各方参与，（2）去中心化：多个验证者参与，（3）稳定性：用于与系统交易的基础令牌的价格路径随时间不会发生大幅变化，和（4）可行性：该机制易于实现为智能合约，即不需要在链上进行买回令牌或执行其他操作。

    arXiv:2403.15429v1 Announce Type: cross  Abstract: We consider long-term equilibria that arise in the tokenomics design of proof-of-stake (PoS) blockchain systems that comprise of users and validators, both striving to maximize their own utilities. Validators are system maintainers who get rewarded with tokens for performing the work necessary for the system to function properly, while users compete and pay with such tokens for getting a desired system service level.   We study how the system service provision and suitable rewards schemes together can lead to equilibria with desirable characteristics (1) viability: the system keeps parties engaged, (2) decentralization: multiple validators are participating, (3) stability: the price path of the underlying token used to transact with the system does not change widely over time, and (4) feasibility: the mechanism is easy to implement as a smart contract, i.e., it does not require fiat reserves on-chain for buy back of tokens or to perfor
    
[^2]: am-AMM: 一个拍卖管理的自动做市商

    am-AMM: An Auction-Managed Automated Market Maker

    [https://arxiv.org/abs/2403.03367](https://arxiv.org/abs/2403.03367)

    本文提出了一个针对AMMs两个重要未解决问题的单一机制：减少对有信息的订单流的损失，最大化对无信息的订单流的收益。

    

    自动做市商（AMMs）已成为区块链上去中心化交易所的主要市场机制。本文提出了一个针对AMMs两个重要未解决问题的单一机制：减少对有信息的订单流的损失，最大化对无信息的订单流的收益。这个“拍卖管理的AMM”通过在链上进行一次不受审查的拍卖，以临时行使“流动性池管理者”职能的权利。流动性池管理者设置池的交换费率，并从交换中获得的费用。流动性池管理者可以通过针对小价格波动而对冲流动性池来独占一些套利，并且可以设置交换费用，包括零售订单流的价格敏感性，并适应不断变化的市场条件，最终将两者的收益归结为流动性提供方。流动性提供方可以进入和退出池…

    arXiv:2403.03367v1 Announce Type: new  Abstract: Automated market makers (AMMs) have emerged as the dominant market mechanism for trading on decentralized exchanges implemented on blockchains. This paper presents a single mechanism that targets two important unsolved problems for AMMs: reducing losses to informed orderflow, and maximizing revenue from uninformed orderflow. The "auction-managed AMM" works by running a censorship-resistant onchain auction for the right to temporarily act as "pool manager" for a constant-product AMM. The pool manager sets the swap fee rate on the pool, and also receives the accrued fees from swaps. The pool manager can exclusively capture some arbitrage by trading against the pool in response to small price movements, and also can set swap fees incorporating price sensitivity of retail orderflow and adapting to changing market conditions, with the benefits from both ultimately accruing to liquidity providers. Liquidity providers can enter and exit the poo
    
[^3]: 通过模拟进行算法性劝导

    Algorithmic Persuasion Through Simulation

    [https://arxiv.org/abs/2311.18138](https://arxiv.org/abs/2311.18138)

    通过模拟接收者行为的贝叶斯劝导问题中，发送者设计了一个最优消息策略并设计了一个多项式时间查询算法，以优化其预期效用。

    

    我们研究了一个贝叶斯劝导问题，其中发送者希望说服接收者采取二元行为，例如购买产品。发送者了解世界的（二元）状态，比如产品质量是高还是低，但是对接收者的信念和效用只有有限的信息。受到客户调查、用户研究和生成式人工智能的最新进展的启发，我们允许发送者通过查询模拟接收者的行为来了解更多关于接收者的信息。在固定数量的查询之后，发送者承诺一个消息策略，接收者根据收到的消息来最大化她的预期效用来采取行动。我们对发送者在任何接收者类型分布下的最优消息策略进行了表征。然后，我们设计了一个多项式时间查询算法，优化了这个贝叶斯劝导游戏中发送者的预期效用。

    arXiv:2311.18138v2 Announce Type: replace-cross Abstract: We study a Bayesian persuasion problem where a sender wants to persuade a receiver to take a binary action, such as purchasing a product. The sender is informed about the (binary) state of the world, such as whether the quality of the product is high or low, but only has limited information about the receiver's beliefs and utilities. Motivated by customer surveys, user studies, and recent advances in generative AI, we allow the sender to learn more about the receiver by querying an oracle that simulates the receiver's behavior. After a fixed number of queries, the sender commits to a messaging policy and the receiver takes the action that maximizes her expected utility given the message she receives. We characterize the sender's optimal messaging policy given any distribution over receiver types. We then design a polynomial-time querying algorithm that optimizes the sender's expected utility in this Bayesian persuasion game. We 
    

