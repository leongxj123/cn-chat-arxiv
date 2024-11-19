# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TSIS: A Supplementary Algorithm to t-SMILES for Fragment-based Molecular Representation](https://arxiv.org/abs/2402.02164) | 本研究引入了TSIS算法作为t-SMILES的补充，用于改进基于字符串的分子表示方法。实验证明，TSIS模型在处理语法中的长期依赖性方面表现优于其他模型。 |

# 详细

[^1]: TSIS: t-SMILES的补充算法用于基于片段的分子表示

    TSIS: A Supplementary Algorithm to t-SMILES for Fragment-based Molecular Representation

    [https://arxiv.org/abs/2402.02164](https://arxiv.org/abs/2402.02164)

    本研究引入了TSIS算法作为t-SMILES的补充，用于改进基于字符串的分子表示方法。实验证明，TSIS模型在处理语法中的长期依赖性方面表现优于其他模型。

    

    字符串基本的分子表示方法，如SMILES，在线性表示分子信息方面是事实上的标准。然而，必须使用配对符号和解析算法导致了长的语法依赖关系，使得即使是最先进的深度学习模型也难以准确理解语法和语义。尽管DeepSMILES和SELFIES已经解决了某些限制，但它们仍然在处理高级语法方面存在困难，使得一些字符串难以阅读。本研究引入了一个补充算法TSIS（TSID简化），用于t-SMILES家族。TSIS与另一个基于片段的线性解决方案SAFE进行了比较实验，结果表明SAFE在处理语法中的长期依赖性时存在挑战。TSIS继续使用t-SMILES中定义的树作为其基础数据结构，这使其与SAFE模型有所不同。TSIS模型的性能超过了SAFE模型，表明t-SMILES的树结构起到了重要作用。

    String-based molecular representations, such as SMILES, are a de facto standard for linearly representing molecular information. However, the must be paired symbols and the parsing algorithm result in long grammatical dependencies, making it difficult for even state-of-the-art deep learning models to accurately comprehend the syntax and semantics. Although DeepSMILES and SELFIES have addressed certain limitations, they still struggle with advanced grammar, which makes some strings difficult to read. This study introduces a supplementary algorithm, TSIS (TSID Simplified), to t-SMILES family. Comparative experiments between TSIS and another fragment-based linear solution, SAFE, indicate that SAFE presents challenges in managing long-term dependencies in grammar. TSIS continues to use the tree defined in t-SMILES as its foundational data structure, which sets it apart from the SAFE model. The performance of TSIS models surpasses that of SAFE models, indicating that the tree structure of t
    

