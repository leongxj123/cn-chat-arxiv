# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HEAM : Hashed Embedding Acceleration using Processing-In-Memory](https://arxiv.org/abs/2402.04032) | HEAM是一种采用异构内存架构的方法，将3D堆叠DRAM与DIMM集成，用于加速处理大规模个性化推荐系统中的嵌入操作。 |

# 详细

[^1]: HEAM: 使用处理-内存进行散列嵌入加速的方法

    HEAM : Hashed Embedding Acceleration using Processing-In-Memory

    [https://arxiv.org/abs/2402.04032](https://arxiv.org/abs/2402.04032)

    HEAM是一种采用异构内存架构的方法，将3D堆叠DRAM与DIMM集成，用于加速处理大规模个性化推荐系统中的嵌入操作。

    

    在当今的数据中心中，个性化推荐系统面临着诸多挑战，特别是在执行嵌入操作时需要大容量的内存和高带宽。之前的方法依赖于DIMM-based近内存处理技术或引入3D堆叠DRAM来解决内存限制和扩展内存带宽的问题。然而，这些解决方案在处理日益扩大的个性化推荐系统大小时存在不足之处。推荐模型已经增长到超过数十TB的大小，导致在传统单节点推断服务器上高效运行变得困难。尽管已经提出了各种算法方法来减小嵌入表容量，但通常会导致内存访问增加或内存资源利用低效的问题。本文引入了HEAM，一种异构内存架构，将3D堆叠DRAM与DIMM集成在一起，以加速组合嵌入的推荐系统。

    In today's data centers, personalized recommendation systems face challenges such as the need for large memory capacity and high bandwidth, especially when performing embedding operations. Previous approaches have relied on DIMM-based near-memory processing techniques or introduced 3D-stacked DRAM to address memory-bound issues and expand memory bandwidth. However, these solutions fall short when dealing with the expanding size of personalized recommendation systems. Recommendation models have grown to sizes exceeding tens of terabytes, making them challenging to run efficiently on traditional single-node inference servers. Although various algorithmic methods have been proposed to reduce embedding table capacity, they often result in increased memory access or inefficient utilization of memory resources. This paper introduces HEAM, a heterogeneous memory architecture that integrates 3D-stacked DRAM with DIMM to accelerate recommendation systems in which compositional embedding is util
    

