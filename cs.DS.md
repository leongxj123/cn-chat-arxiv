# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation](https://arxiv.org/abs/2311.09922) | 本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。 |
| [^2] | [Memory-Efficient Sequential Pattern Mining with Hybrid Tries](https://arxiv.org/abs/2202.06834) | 提出了一种基于混合trie的内存高效序列模式挖掘方法，在内存消耗和计算时间方面相比现有技术有显著改善，且是唯一一个能够处理256GB系统内存下大数据集的方法。 |
| [^3] | [Non-Clashing Teaching Maps for Balls in Graphs.](http://arxiv.org/abs/2309.02876) | 本文研究了在图中球的概念类中的非冲突教学图，并证明了相关决策问题{\sc B-NCTD$^+$}是NP完全的。 |

# 详细

[^1]: 通过采用整数列表作为多项式基数2指数的集合来实现快速乘法

    Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation

    [https://arxiv.org/abs/2311.09922](https://arxiv.org/abs/2311.09922)

    本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。

    

    我们演示了一种基于用整数列表表示的多项式基数2指数集合的乘法方法。该方法采用python代码实现了一组算法。我们展示了该方法在某一位数范围内比数论变换(NTT)和卡拉茨巴(Karatsuba)乘法更快。我们还实现了用python代码进行比较，与多项式基数2整数方法进行比较。我们展示了任何整数或实数都可以表示为整数索引列表，表示二进制中的有限级数。该数字的整数索引有限级数可以存储和分布在多个CPU / GPU上。我们展示了加法和乘法运算可以应用于作为索引整数表示的两个补码加法，并可以完全分布在给定的CPU / GPU架构上。我们展示了完全的分布性能。

    We demonstrate a multiplication method based on numbers represented as set of polynomial radix 2 indices stored as an integer list. The 'polynomial integer index multiplication' method is a set of algorithms implemented in python code. We demonstrate the method to be faster than both the Number Theoretic Transform (NTT) and Karatsuba for multiplication within a certain bit range. Also implemented in python code for comparison purposes with the polynomial radix 2 integer method. We demonstrate that it is possible to express any integer or real number as a list of integer indices, representing a finite series in base two. The finite series of integer index representation of a number can then be stored and distributed across multiple CPUs / GPUs. We show that operations of addition and multiplication can be applied as two's complement additions operating on the index integer representations and can be fully distributed across a given CPU / GPU architecture. We demonstrate fully distribute
    
[^2]: 基于混合Tries的内存高效序列模式挖掘

    Memory-Efficient Sequential Pattern Mining with Hybrid Tries

    [https://arxiv.org/abs/2202.06834](https://arxiv.org/abs/2202.06834)

    提出了一种基于混合trie的内存高效序列模式挖掘方法，在内存消耗和计算时间方面相比现有技术有显著改善，且是唯一一个能够处理256GB系统内存下大数据集的方法。

    

    随着现代数据集的指数级增长，对于能够处理如此庞大数据集的高效挖掘算法的需求变得日益迫切。本文提出了一种内存高效的方法用于序列模式挖掘（SPM），这是知识发现中的一个基本主题，面临着针对大数据集的已知内存瓶颈。我们的方法涉及一种新颖的混合trie数据结构，利用重复模式紧凑地存储内存中的数据集; 以及一个相应的挖掘算法，旨在有效地从此紧凑表示中提取模式。对真实测试实例的数值结果显示，与最先进技术相比，对于小到中等大小的数据集，内存消耗平均提高了88％，计算时间提高了41％。此外，我们的算法是唯一一个在系统内存为256GB的情况下能够处理大数据集的SPM方法。

    arXiv:2202.06834v2 Announce Type: replace-cross  Abstract: As modern data sets continue to grow exponentially in size, the demand for efficient mining algorithms capable of handling such large data sets becomes increasingly imperative. This paper develops a memory-efficient approach for Sequential Pattern Mining (SPM), a fundamental topic in knowledge discovery that faces a well-known memory bottleneck for large data sets. Our methodology involves a novel hybrid trie data structure that exploits recurring patterns to compactly store the data set in memory; and a corresponding mining algorithm designed to effectively extract patterns from this compact representation. Numerical results on real-life test instances show an average improvement of 88% in memory consumption and 41% in computation time for small to medium-sized data sets compared to the state of the art. Furthermore, our algorithm stands out as the only capable SPM approach for large data sets within 256GB of system memory.
    
[^3]: 非冲突教学图在图中球的应用研究

    Non-Clashing Teaching Maps for Balls in Graphs. (arXiv:2309.02876v1 [cs.CC])

    [http://arxiv.org/abs/2309.02876](http://arxiv.org/abs/2309.02876)

    本文研究了在图中球的概念类中的非冲突教学图，并证明了相关决策问题{\sc B-NCTD$^+$}是NP完全的。

    

    最近，Kirkpatrick等人[ALT 2019]和Fallat等人[JMLR 2023]引入了非冲突教学，并表明它是满足Goldman和Mathias提出的防止勾结基准的最高效的机器教学模型。对于一个概念类$\cal{C}$来说，教学图$T$将一个（教学）集合$T(C)$分配给每个概念$C \in \cal{C}$。如果没有一对概念与它们的教学集合的并一致，则教学图是非冲突的。非冲突教学图（NCTM）$T$的大小是$T(C)$中的最大大小，其中$C \in \cal{C}$。概念类$\mathcal{B}(G)$的非冲突教学维度NCTD$(\cal{C})$是$\cal{C}$的一个NCTM的最小大小。类似地，NCTM$^+$和NCTD$^+(\cal{C})$的定义是类似的，只是教师只能使用正例。我们研究了由图$G$的所有球组成的概念类$\mathcal{B}(G)$的NCTMs和NCTM$^+$s。我们证明了与NCTD$^+$的相关决策问题{\sc B-NCTD$^+$}是NP完全的。

    Recently, Kirkpatrick et al. [ALT 2019] and Fallat et al. [JMLR 2023] introduced non-clashing teaching and showed it to be the most efficient machine teaching model satisfying the benchmark for collusion-avoidance set by Goldman and Mathias. A teaching map $T$ for a concept class $\cal{C}$ assigns a (teaching) set $T(C)$ of examples to each concept $C \in \cal{C}$. A teaching map is non-clashing if no pair of concepts are consistent with the union of their teaching sets. The size of a non-clashing teaching map (NCTM) $T$ is the maximum size of a $T(C)$, $C \in \cal{C}$. The non-clashing teaching dimension NCTD$(\cal{C})$ of $\cal{C}$ is the minimum size of an NCTM for $\cal{C}$. NCTM$^+$ and NCTD$^+(\cal{C})$ are defined analogously, except the teacher may only use positive examples.  We study NCTMs and NCTM$^+$s for the concept class $\mathcal{B}(G)$ consisting of all balls of a graph $G$. We show that the associated decision problem {\sc B-NCTD$^+$} for NCTD$^+$ is NP-complete in spl
    

