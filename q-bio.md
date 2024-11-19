# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Supervised machine learning for microbiomics: bridging the gap between current and best practices](https://arxiv.org/abs/2402.17621) | 该研究通过分析大量期刊文章，总结了监督机器学习在微生物组学中的现有实践，探讨了实验设计方法的优缺点，并提出了如何避免常见实验设计缺陷的指导。 |
| [^2] | [TSIS: A Supplementary Algorithm to t-SMILES for Fragment-based Molecular Representation](https://arxiv.org/abs/2402.02164) | 本研究引入了TSIS算法作为t-SMILES的补充，用于改进基于字符串的分子表示方法。实验证明，TSIS模型在处理语法中的长期依赖性方面表现优于其他模型。 |

# 详细

[^1]: 用于微生物组学的监督机器学习：弥合当前和最佳实践之间的差距

    Supervised machine learning for microbiomics: bridging the gap between current and best practices

    [https://arxiv.org/abs/2402.17621](https://arxiv.org/abs/2402.17621)

    该研究通过分析大量期刊文章，总结了监督机器学习在微生物组学中的现有实践，探讨了实验设计方法的优缺点，并提出了如何避免常见实验设计缺陷的指导。

    

    机器学习（ML）将加速临床微生物组学创新，如疾病诊断和预后。这将需要高质量、可重现、可解释的工作流程，其预测能力达到或超过监管机构对临床工具设定的高门槛。我们通过深入分析2021-2022年发表的100篇同行评议的期刊文章，捕捉了当前将监督ML应用于微生物组学数据的实践的一个快照。我们采用数据驱动方法，引导讨论各种实验设计方法的优点，包括关键考虑因素，如如何减轻小数据集大小的影响同时避免数据泄漏。我们进一步提供关于如何避免可能损害模型性能、可信度和可重复性的常见实验设计缺陷的指南。讨论附有一个互动在线教程。

    arXiv:2402.17621v1 Announce Type: cross  Abstract: Machine learning (ML) is set to accelerate innovations in clinical microbiomics, such as in disease diagnostics and prognostics. This will require high-quality, reproducible, interpretable workflows whose predictive capabilities meet or exceed the high thresholds set for clinical tools by regulatory agencies. Here, we capture a snapshot of current practices in the application of supervised ML to microbiomics data, through an in-depth analysis of 100 peer-reviewed journal articles published in 2021-2022. We apply a data-driven approach to steer discussion of the merits of varied approaches to experimental design, including key considerations such as how to mitigate the effects of small dataset size while avoiding data leakage. We further provide guidance on how to avoid common experimental design pitfalls that can hurt model performance, trustworthiness, and reproducibility. Discussion is accompanied by an interactive online tutorial th
    
[^2]: TSIS: t-SMILES的补充算法用于基于片段的分子表示

    TSIS: A Supplementary Algorithm to t-SMILES for Fragment-based Molecular Representation

    [https://arxiv.org/abs/2402.02164](https://arxiv.org/abs/2402.02164)

    本研究引入了TSIS算法作为t-SMILES的补充，用于改进基于字符串的分子表示方法。实验证明，TSIS模型在处理语法中的长期依赖性方面表现优于其他模型。

    

    字符串基本的分子表示方法，如SMILES，在线性表示分子信息方面是事实上的标准。然而，必须使用配对符号和解析算法导致了长的语法依赖关系，使得即使是最先进的深度学习模型也难以准确理解语法和语义。尽管DeepSMILES和SELFIES已经解决了某些限制，但它们仍然在处理高级语法方面存在困难，使得一些字符串难以阅读。本研究引入了一个补充算法TSIS（TSID简化），用于t-SMILES家族。TSIS与另一个基于片段的线性解决方案SAFE进行了比较实验，结果表明SAFE在处理语法中的长期依赖性时存在挑战。TSIS继续使用t-SMILES中定义的树作为其基础数据结构，这使其与SAFE模型有所不同。TSIS模型的性能超过了SAFE模型，表明t-SMILES的树结构起到了重要作用。

    String-based molecular representations, such as SMILES, are a de facto standard for linearly representing molecular information. However, the must be paired symbols and the parsing algorithm result in long grammatical dependencies, making it difficult for even state-of-the-art deep learning models to accurately comprehend the syntax and semantics. Although DeepSMILES and SELFIES have addressed certain limitations, they still struggle with advanced grammar, which makes some strings difficult to read. This study introduces a supplementary algorithm, TSIS (TSID Simplified), to t-SMILES family. Comparative experiments between TSIS and another fragment-based linear solution, SAFE, indicate that SAFE presents challenges in managing long-term dependencies in grammar. TSIS continues to use the tree defined in t-SMILES as its foundational data structure, which sets it apart from the SAFE model. The performance of TSIS models surpasses that of SAFE models, indicating that the tree structure of t
    

