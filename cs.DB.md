# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Memory-Efficient Sequential Pattern Mining with Hybrid Tries](https://arxiv.org/abs/2202.06834) | 提出了一种基于混合trie的内存高效序列模式挖掘方法，在内存消耗和计算时间方面相比现有技术有显著改善，且是唯一一个能够处理256GB系统内存下大数据集的方法。 |
| [^2] | [Temporalising Unique Characterisability and Learnability of Ontology-Mediated Queries.](http://arxiv.org/abs/2306.07662) | 本文研究了在时间化本体中介查询中唯一可特征性和可学习性的问题，并提出了相应的传递结果。 |

# 详细

[^1]: 基于混合Tries的内存高效序列模式挖掘

    Memory-Efficient Sequential Pattern Mining with Hybrid Tries

    [https://arxiv.org/abs/2202.06834](https://arxiv.org/abs/2202.06834)

    提出了一种基于混合trie的内存高效序列模式挖掘方法，在内存消耗和计算时间方面相比现有技术有显著改善，且是唯一一个能够处理256GB系统内存下大数据集的方法。

    

    随着现代数据集的指数级增长，对于能够处理如此庞大数据集的高效挖掘算法的需求变得日益迫切。本文提出了一种内存高效的方法用于序列模式挖掘（SPM），这是知识发现中的一个基本主题，面临着针对大数据集的已知内存瓶颈。我们的方法涉及一种新颖的混合trie数据结构，利用重复模式紧凑地存储内存中的数据集; 以及一个相应的挖掘算法，旨在有效地从此紧凑表示中提取模式。对真实测试实例的数值结果显示，与最先进技术相比，对于小到中等大小的数据集，内存消耗平均提高了88％，计算时间提高了41％。此外，我们的算法是唯一一个在系统内存为256GB的情况下能够处理大数据集的SPM方法。

    arXiv:2202.06834v2 Announce Type: replace-cross  Abstract: As modern data sets continue to grow exponentially in size, the demand for efficient mining algorithms capable of handling such large data sets becomes increasingly imperative. This paper develops a memory-efficient approach for Sequential Pattern Mining (SPM), a fundamental topic in knowledge discovery that faces a well-known memory bottleneck for large data sets. Our methodology involves a novel hybrid trie data structure that exploits recurring patterns to compactly store the data set in memory; and a corresponding mining algorithm designed to effectively extract patterns from this compact representation. Numerical results on real-life test instances show an average improvement of 88% in memory consumption and 41% in computation time for small to medium-sized data sets compared to the state of the art. Furthermore, our algorithm stands out as the only capable SPM approach for large data sets within 256GB of system memory.
    
[^2]: 时间化本体中介查询的唯一可特征性和可学习性

    Temporalising Unique Characterisability and Learnability of Ontology-Mediated Queries. (arXiv:2306.07662v1 [cs.AI])

    [http://arxiv.org/abs/2306.07662](http://arxiv.org/abs/2306.07662)

    本文研究了在时间化本体中介查询中唯一可特征性和可学习性的问题，并提出了相应的传递结果。

    

    最近，通过示例来研究数据库查询的唯一可特征性和可学习性已经扩展到了本体中介查询。在这里，我们研究了获得的结果在多大程度上可以提升到时间化的本体中介查询。我们系统地介绍了非时间化情况下相关方法，然后展示了通用的传递结果，可以确定现有结果在何种条件下可以推广到时间化查询。

    Recently, the study of the unique characterisability and learnability of database queries by means of examples has been extended to ontology-mediated queries. Here, we study in how far the obtained results can be lifted to temporalised ontology-mediated queries. We provide a systematic introduction to the relevant approaches in the non-temporal case and then show general transfer results pinpointing under which conditions existing results can be lifted to temporalised queries.
    

