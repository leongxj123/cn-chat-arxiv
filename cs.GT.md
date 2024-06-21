# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Convergence of Federated Learning Algorithms without Data Similarity](https://arxiv.org/abs/2403.02347) | 本文提出了一种无需数据相似性条件的联邦学习算法收敛性分析框架，通过推导出三种常用步长调度的精确表达式，实现了对算法收敛性能的全面评估。 |
| [^2] | [Collusion-Resilience in Transaction Fee Mechanism Design](https://arxiv.org/abs/2402.09321) | 本论文研究了交易手续费机制设计中的防勾结性问题，讨论了多个要求和属性，并指出在存在交易竞争时，任何TFM都无法同时满足所有要求和属性。 |
| [^3] | [Multi-Sender Persuasion -- A Computational Perspective](https://arxiv.org/abs/2402.04971) | 这项研究考虑了多个具有信息优势的发信者向单个自私行为者传递信号以影响其行为的问题，并提出了一种新颖的可微神经网络方法来近似解决这一问题。通过额外梯度算法，我们发现了超越已有方法的局部均衡解。 |

# 详细

[^1]: 关于无需数据相似性条件的联邦学习算法收敛性

    On the Convergence of Federated Learning Algorithms without Data Similarity

    [https://arxiv.org/abs/2403.02347](https://arxiv.org/abs/2403.02347)

    本文提出了一种无需数据相似性条件的联邦学习算法收敛性分析框架，通过推导出三种常用步长调度的精确表达式，实现了对算法收敛性能的全面评估。

    

    数据相似性假设传统上被广泛依赖于理解联邦学习方法的收敛行为。不幸的是，这种方法通常要求根据数据相似性程度微调步长。当数据相似性较低时，这些小步长会导致联邦方法的收敛速度不可接受地慢。本文提出了一种新颖和统一的框架，用于分析联邦学习算法的收敛性，无需数据相似性条件。我们的分析集中在一个不等式上，这个不等式捕捉了步长对算法收敛性能的影响。通过将我们的定理应用于众所周知的联邦算法，我们推导出了三种广泛使用的步长调度的精确表达式：固定步长、递减步长和步衰减步长，这些表达式独立于数据相似性条件。最后，我们对性能进行了全面评估。

    arXiv:2403.02347v1 Announce Type: new  Abstract: Data similarity assumptions have traditionally been relied upon to understand the convergence behaviors of federated learning methods. Unfortunately, this approach often demands fine-tuning step sizes based on the level of data similarity. When data similarity is low, these small step sizes result in an unacceptably slow convergence speed for federated methods. In this paper, we present a novel and unified framework for analyzing the convergence of federated learning algorithms without the need for data similarity conditions. Our analysis centers on an inequality that captures the influence of step sizes on algorithmic convergence performance. By applying our theorems to well-known federated algorithms, we derive precise expressions for three widely used step size schedules: fixed, diminishing, and step-decay step sizes, which are independent of data similarity conditions. Finally, we conduct comprehensive evaluations of the performance 
    
[^2]: 交易手续费机制设计中的防勾结性

    Collusion-Resilience in Transaction Fee Mechanism Design

    [https://arxiv.org/abs/2402.09321](https://arxiv.org/abs/2402.09321)

    本论文研究了交易手续费机制设计中的防勾结性问题，讨论了多个要求和属性，并指出在存在交易竞争时，任何TFM都无法同时满足所有要求和属性。

    

    在区块链协议中，用户通过交易手续费机制（TFM）进行竞标，以便将其交易包含并获得确认。Roughgarden（EC'21）对TFM进行了正式的处理，并提出了三个要求：用户激励兼容性（UIC），矿工激励兼容性（MIC）以及一种称为OCA-proofness的防勾结性形式。当没有交易之间的竞争时，Ethereum的EIP-1559机制同时满足这三个属性，但当有过多的符合条件的交易无法放入单个区块时，失去了UIC属性。Chung和Shi（SODA'23）考虑了一种替代的防勾结性概念，称为c-side-construct-proofness(c-SCP)，并证明了当交易之间存在竞争时，任何TFM都不能满足UIC、MIC和至少为1的任何c的c-SCP。OCA-proofness断言用户和矿工不应该能够从协议中“偷取”，并且在直觉上比UIC、MIC更弱。

    arXiv:2402.09321v1 Announce Type: cross Abstract: Users bid in a transaction fee mechanism (TFM) to get their transactions included and confirmed by a blockchain protocol. Roughgarden (EC'21) initiated the formal treatment of TFMs and proposed three requirements: user incentive compatibility (UIC), miner incentive compatibility (MIC), and a form of collusion-resilience called OCA-proofness. Ethereum's EIP-1559 mechanism satisfies all three properties simultaneously when there is no contention between transactions, but loses the UIC property when there are too many eligible transactions to fit in a single block. Chung and Shi (SODA'23) considered an alternative notion of collusion-resilience, called c-side-constract-proofness (c-SCP), and showed that, when there is contention between transactions, no TFM can satisfy UIC, MIC, and c-SCP for any c at least 1. OCA-proofness asserts that the users and a miner should not be able to "steal from the protocol" and is intuitively weaker than the
    
[^3]: 多发信者说服 - 从计算的角度来看

    Multi-Sender Persuasion -- A Computational Perspective

    [https://arxiv.org/abs/2402.04971](https://arxiv.org/abs/2402.04971)

    这项研究考虑了多个具有信息优势的发信者向单个自私行为者传递信号以影响其行为的问题，并提出了一种新颖的可微神经网络方法来近似解决这一问题。通过额外梯度算法，我们发现了超越已有方法的局部均衡解。

    

    我们考虑到具有信息优势的多个发信者向单个自私行为者传递信号以使其采取某些行动。这些设置是计算经济学，多智能体学习和具有多个目标的机器学习中普遍存在的。核心解决方案概念是发信者信号策略的纳什均衡。理论上，我们证明一般情况下找到一个均衡是PPAD-Hard的;实际上，计算一个发信者的最佳响应甚至是NP-Hard的。鉴于这些固有的困难，我们转而寻找局部纳什均衡。我们提出了一种新颖的可微神经网络来近似该游戏的非线性和不连续效用。结合额外梯度算法，我们发现了超越完全展示均衡和现有神经网络发现的局部均衡。广义上，我们的理论和实证贡献对广泛的类别感兴趣。

    We consider multiple senders with informational advantage signaling to convince a single self-interested actor towards certain actions. Generalizing the seminal Bayesian Persuasion framework, such settings are ubiquitous in computational economics, multi-agent learning, and machine learning with multiple objectives. The core solution concept here is the Nash equilibrium of senders' signaling policies. Theoretically, we prove that finding an equilibrium in general is PPAD-Hard; in fact, even computing a sender's best response is NP-Hard. Given these intrinsic difficulties, we turn to finding local Nash equilibria. We propose a novel differentiable neural network to approximate this game's non-linear and discontinuous utilities. Complementing this with the extra-gradient algorithm, we discover local equilibria that Pareto dominates full-revelation equilibria and those found by existing neural networks. Broadly, our theoretical and empirical contributions are of interest to a large class 
    

