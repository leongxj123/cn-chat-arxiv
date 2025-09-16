# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collusion-Resilience in Transaction Fee Mechanism Design](https://arxiv.org/abs/2402.09321) | 本论文研究了交易手续费机制设计中的防勾结性问题，讨论了多个要求和属性，并指出在存在交易竞争时，任何TFM都无法同时满足所有要求和属性。 |

# 详细

[^1]: 交易手续费机制设计中的防勾结性

    Collusion-Resilience in Transaction Fee Mechanism Design

    [https://arxiv.org/abs/2402.09321](https://arxiv.org/abs/2402.09321)

    本论文研究了交易手续费机制设计中的防勾结性问题，讨论了多个要求和属性，并指出在存在交易竞争时，任何TFM都无法同时满足所有要求和属性。

    

    在区块链协议中，用户通过交易手续费机制（TFM）进行竞标，以便将其交易包含并获得确认。Roughgarden（EC'21）对TFM进行了正式的处理，并提出了三个要求：用户激励兼容性（UIC），矿工激励兼容性（MIC）以及一种称为OCA-proofness的防勾结性形式。当没有交易之间的竞争时，Ethereum的EIP-1559机制同时满足这三个属性，但当有过多的符合条件的交易无法放入单个区块时，失去了UIC属性。Chung和Shi（SODA'23）考虑了一种替代的防勾结性概念，称为c-side-construct-proofness(c-SCP)，并证明了当交易之间存在竞争时，任何TFM都不能满足UIC、MIC和至少为1的任何c的c-SCP。OCA-proofness断言用户和矿工不应该能够从协议中“偷取”，并且在直觉上比UIC、MIC更弱。

    arXiv:2402.09321v1 Announce Type: cross Abstract: Users bid in a transaction fee mechanism (TFM) to get their transactions included and confirmed by a blockchain protocol. Roughgarden (EC'21) initiated the formal treatment of TFMs and proposed three requirements: user incentive compatibility (UIC), miner incentive compatibility (MIC), and a form of collusion-resilience called OCA-proofness. Ethereum's EIP-1559 mechanism satisfies all three properties simultaneously when there is no contention between transactions, but loses the UIC property when there are too many eligible transactions to fit in a single block. Chung and Shi (SODA'23) considered an alternative notion of collusion-resilience, called c-side-constract-proofness (c-SCP), and showed that, when there is contention between transactions, no TFM can satisfy UIC, MIC, and c-SCP for any c at least 1. OCA-proofness asserts that the users and a miner should not be able to "steal from the protocol" and is intuitively weaker than the
    

