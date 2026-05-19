# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning](https://arxiv.org/abs/2309.01243) | 本文提出了正态分布不可区分性谱定理 (NDIS Theorem)，旨在利用查询本身的随机性改进随机化机器学习查询的差分隐私机制。 |
| [^2] | [Smart Learning to Find Dumb Contracts.](http://arxiv.org/abs/2304.10726) | DLVA是一种用于以太坊智能合约的强大深度学习漏洞检测工具，其算法涵盖了源代码到字节码的扩展，并且速度比传统漏洞检测工具提高了10-500倍，并成功地发现了一些Slither误标记的易受攻击的合约。 |

# 详细

[^1]: 正态分布不可区分性谱及其在隐私保护机器学习中的应用

    The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning

    [https://arxiv.org/abs/2309.01243](https://arxiv.org/abs/2309.01243)

    本文提出了正态分布不可区分性谱定理 (NDIS Theorem)，旨在利用查询本身的随机性改进随机化机器学习查询的差分隐私机制。

    

    要实现差分隐私(DP)，通常需要随机化基础查询的输出。在大数据分析中，人们经常使用随机化草图/聚合算法来使处理高维数据变得可行。直观地，这样的机器学习(ML)算法应该提供一些固有的隐私性，但现有的大部分DP机制并没有利用这种固有的随机性，导致潜在的多余噪音。我们工作的动机问题是：(如何)可以通过利用查询本身的随机性来提高随机化ML查询的DP机制的效用？为了给出积极的答案，我们证明了正态分布不可区分性谱定理(简称为NDIS定理)，这是一个具有深远实际影响的理论结果。总的来说，NDIS是一个用于$(\epsilon,\delta)$-不可区分性谱(简称为$

    arXiv:2309.01243v2 Announce Type: replace-cross  Abstract: To achieve differential privacy (DP) one typically randomizes the output of the underlying query. In big data analytics, one often uses randomized sketching/aggregation algorithms to make processing high-dimensional data tractable. Intuitively, such machine learning (ML) algorithms should provide some inherent privacy, yet most if not all existing DP mechanisms do not leverage this inherent randomness, resulting in potentially redundant noising.   The motivating question of our work is:   (How) can we improve the utility of DP mechanisms for randomized ML queries, by leveraging the randomness of the query itself?   Towards a (positive) answer, we prove the Normal Distributions Indistinguishability Spectrum Theorem (in short, NDIS Theorem), a theoretical result with far-reaching practical implications. In a nutshell, NDIS is a closed-form analytic computation for the $(\epsilon,\delta)$-indistinguishability-spectrum (in short, $
    
[^2]: 智能学习发现 愚笨合约

    Smart Learning to Find Dumb Contracts. (arXiv:2304.10726v1 [cs.CR])

    [http://arxiv.org/abs/2304.10726](http://arxiv.org/abs/2304.10726)

    DLVA是一种用于以太坊智能合约的强大深度学习漏洞检测工具，其算法涵盖了源代码到字节码的扩展，并且速度比传统漏洞检测工具提高了10-500倍，并成功地发现了一些Slither误标记的易受攻击的合约。

    

    我们引入了基于强大深度学习技术的 Deep Learning Vulnerability Analyzer （DLVA），它是一种针对以字节码为基础的以太坊智能合约的漏洞检测工具。我们在没有手动特征工程、预定义模式或专家规则的情况下，将源代码分析扩展到字节码，训练DLVA判断字节码。DLVA训练算法的鲁棒性也很强：它克服了1.25%误标记合约的错误率，学生超越了老师，并发现了Slither误标记的易受攻击的合约。DLVA比基于形式方法的传统智能合约漏洞检测工具快得多：DLVA检查了29个漏洞所需的时间为0.2秒，速度提高了10-500倍。DLVA有三个关键组成部分：Smart Contract to Vector（SC2Vec）将智能合约转换为深度学习模型的向量表示。Bytecode Tokenizer（BCT）将底层字节码转换为神经网络的有意义的标记，DLVA是神经网络模型，可预测智能合约是否包含漏洞。我们对Etherscan的28,505个经过验证的智能合约数据集进行了DLVA评估，发现它取得了0.964的AUC（真阳率/假阳率曲线下的面积）得分。与基线方法相比，DLVA在F1分数上显示了30.7%的改进，它是精度和召回的调和平均值。

    We introduce Deep Learning Vulnerability Analyzer (DLVA), a vulnerability detection tool for Ethereum smart contracts based on powerful deep learning techniques for sequential data adapted for bytecode. We train DLVA to judge bytecode even though the supervising oracle, Slither, can only judge source code. DLVA's training algorithm is general: we "extend" a source code analysis to bytecode without any manual feature engineering, predefined patterns, or expert rules. DLVA's training algorithm is also robust: it overcame a 1.25% error rate mislabeled contracts, and the student surpassing the teacher; found vulnerable contracts that Slither mislabeled. In addition to extending a source code analyzer to bytecode, DLVA is much faster than conventional tools for smart contract vulnerability detection based on formal methods: DLVA checks contracts for 29 vulnerabilities in 0.2 seconds, a speedup of 10-500x+ compared to traditional tools.  DLVA has three key components. Smart Contract to Vecto
    

