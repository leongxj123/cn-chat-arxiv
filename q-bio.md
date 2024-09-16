# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SpanSeq: Similarity-based sequence data splitting method for improved development and assessment of deep learning projects](https://arxiv.org/abs/2402.14482) | SpanSeq 是一种用于生物数据序列的数据库分区方法，能够避免训练集和测试集之间的数据泄漏。 |

# 详细

[^1]: SpanSeq：用于改进深度学习项目开发和评估的基于相似度的序列数据拆分方法

    SpanSeq: Similarity-based sequence data splitting method for improved development and assessment of deep learning projects

    [https://arxiv.org/abs/2402.14482](https://arxiv.org/abs/2402.14482)

    SpanSeq 是一种用于生物数据序列的数据库分区方法，能够避免训练集和测试集之间的数据泄漏。

    

    过去几年中，在计算生物学中使用深度学习模型的增加很大，并且随着诸如自然语言处理等领域的当前进展，预计将进一步增加。本文提出了SpanSeq，这是一种适用于大多数生物序列（基因、蛋白质和基因组）的机器学习数据库分区方法，旨在避免数据集之间的数据泄漏。

    arXiv:2402.14482v1 Announce Type: new  Abstract: The use of deep learning models in computational biology has increased massively in recent years, and is expected to do so further with the current advances in fields like Natural Language Processing. These models, although able to draw complex relations between input and target, are also largely inclined to learn noisy deviations from the pool of data used during their development. In order to assess their performance on unseen data (their capacity to generalize), it is common to randomly split the available data in development (train/validation) and test sets. This procedure, although standard, has lately been shown to produce dubious assessments of generalization due to the existing similarity between samples in the databases used. In this work, we present SpanSeq, a database partition method for machine learning that can scale to most biological sequences (genes, proteins and genomes) in order to avoid data leakage between sets. We a
    

