# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Smart Learning to Find Dumb Contracts.](http://arxiv.org/abs/2304.10726) | DLVA是一种用于以太坊智能合约的强大深度学习漏洞检测工具，其算法涵盖了源代码到字节码的扩展，并且速度比传统漏洞检测工具提高了10-500倍，并成功地发现了一些Slither误标记的易受攻击的合约。 |

# 详细

[^1]: 智能学习发现 愚笨合约

    Smart Learning to Find Dumb Contracts. (arXiv:2304.10726v1 [cs.CR])

    [http://arxiv.org/abs/2304.10726](http://arxiv.org/abs/2304.10726)

    DLVA是一种用于以太坊智能合约的强大深度学习漏洞检测工具，其算法涵盖了源代码到字节码的扩展，并且速度比传统漏洞检测工具提高了10-500倍，并成功地发现了一些Slither误标记的易受攻击的合约。

    

    我们引入了基于强大深度学习技术的 Deep Learning Vulnerability Analyzer （DLVA），它是一种针对以字节码为基础的以太坊智能合约的漏洞检测工具。我们在没有手动特征工程、预定义模式或专家规则的情况下，将源代码分析扩展到字节码，训练DLVA判断字节码。DLVA训练算法的鲁棒性也很强：它克服了1.25%误标记合约的错误率，学生超越了老师，并发现了Slither误标记的易受攻击的合约。DLVA比基于形式方法的传统智能合约漏洞检测工具快得多：DLVA检查了29个漏洞所需的时间为0.2秒，速度提高了10-500倍。DLVA有三个关键组成部分：Smart Contract to Vector（SC2Vec）将智能合约转换为深度学习模型的向量表示。Bytecode Tokenizer（BCT）将底层字节码转换为神经网络的有意义的标记，DLVA是神经网络模型，可预测智能合约是否包含漏洞。我们对Etherscan的28,505个经过验证的智能合约数据集进行了DLVA评估，发现它取得了0.964的AUC（真阳率/假阳率曲线下的面积）得分。与基线方法相比，DLVA在F1分数上显示了30.7%的改进，它是精度和召回的调和平均值。

    We introduce Deep Learning Vulnerability Analyzer (DLVA), a vulnerability detection tool for Ethereum smart contracts based on powerful deep learning techniques for sequential data adapted for bytecode. We train DLVA to judge bytecode even though the supervising oracle, Slither, can only judge source code. DLVA's training algorithm is general: we "extend" a source code analysis to bytecode without any manual feature engineering, predefined patterns, or expert rules. DLVA's training algorithm is also robust: it overcame a 1.25% error rate mislabeled contracts, and the student surpassing the teacher; found vulnerable contracts that Slither mislabeled. In addition to extending a source code analyzer to bytecode, DLVA is much faster than conventional tools for smart contract vulnerability detection based on formal methods: DLVA checks contracts for 29 vulnerabilities in 0.2 seconds, a speedup of 10-500x+ compared to traditional tools.  DLVA has three key components. Smart Contract to Vecto
    

