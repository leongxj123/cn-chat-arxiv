# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features.](http://arxiv.org/abs/2308.16391) | 这篇论文提出了一种基于交易的方法来提高以太坊上庞氏骗局的检测鲁棒性和准确性。现有的方法主要基于智能合约源代码或操作码进行检测，但缺乏鲁棒性。通过分析交易数据，可以更有效地识别庞氏骗局，因为交易更难伪装。 |

# 详细

[^1]: 提高以太坊上庞氏骗局检测的鲁棒性和准确性的方法

    Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features. (arXiv:2308.16391v1 [cs.CR])

    [http://arxiv.org/abs/2308.16391](http://arxiv.org/abs/2308.16391)

    这篇论文提出了一种基于交易的方法来提高以太坊上庞氏骗局的检测鲁棒性和准确性。现有的方法主要基于智能合约源代码或操作码进行检测，但缺乏鲁棒性。通过分析交易数据，可以更有效地识别庞氏骗局，因为交易更难伪装。

    

    区块链的快速发展导致越来越多的资金涌入加密货币市场，也吸引了近年来网络犯罪分子的兴趣。庞氏骗局作为一种老式的欺诈行为，现在也流行于区块链上，给许多加密货币投资者造成了巨大的财务损失。现有文献中已经提出了一些庞氏骗局检测方法，其中大多数是基于智能合约的源代码或操作码进行检测的。虽然基于合约代码的方法在准确性方面表现出色，但它缺乏鲁棒性：首先，大部分以太坊上的合约源代码并不公开可用；其次，庞氏骗局开发者可以通过混淆操作码或者创造新的分配逻辑来欺骗基于合约代码的检测模型（因为这些模型仅在现有的庞氏逻辑上进行训练）。基于交易的方法可以提高检测的鲁棒性，因为与智能合约不同，交易更加难以伪装。

    The rapid development of blockchain has led to more and more funding pouring into the cryptocurrency market, which also attracted cybercriminals' interest in recent years. The Ponzi scheme, an old-fashioned fraud, is now popular on the blockchain, causing considerable financial losses to many crypto-investors. A few Ponzi detection methods have been proposed in the literature, most of which detect a Ponzi scheme based on its smart contract source code or opcode. The contract-code-based approach, while achieving very high accuracy, is not robust: first, the source codes of a majority of contracts on Ethereum are not available, and second, a Ponzi developer can fool a contract-code-based detection model by obfuscating the opcode or inventing a new profit distribution logic that cannot be detected (since these models were trained on existing Ponzi logics only). A transaction-based approach could improve the robustness of detection because transactions, unlike smart contracts, are harder t
    

