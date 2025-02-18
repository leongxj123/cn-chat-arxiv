# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GreedyML: A Parallel Algorithm for Maximizing Submodular Functions](https://arxiv.org/abs/2403.10332) | 提出了一种用于在分布式存储多处理器上最大化子模函数的并行近似算法，以解决实际应用领域中海量数据集上的子模优化问题。 |

# 详细

[^1]: GreedyML：一种用于最大化子模函数的并行算法

    GreedyML: A Parallel Algorithm for Maximizing Submodular Functions

    [https://arxiv.org/abs/2403.10332](https://arxiv.org/abs/2403.10332)

    提出了一种用于在分布式存储多处理器上最大化子模函数的并行近似算法，以解决实际应用领域中海量数据集上的子模优化问题。

    

    我们描述了一种用于在分布式存储多处理器上最大化单调子模函数的并行近似算法。我们的工作受到在海量数据集上解决子模优化问题的需求的启发，用于实际应用领域，如数据摘要，机器学习和图稀疏化。我们的工作基于Barbosa、Ene、Nguyen和Ward（2015）提出的随机分布式RandGreedI算法。该算法通过将数据随机分区到所有处理器中，然后使用单个累积步骤计算分布式解决方案，其中所有处理器将它们的部分解决方案发送给一个处理器。然而，对于大问题，累积步骤可能超过处理器上可用的内存，并且执行累积的处理器可能成为计算瓶颈。

    arXiv:2403.10332v1 Announce Type: cross  Abstract: We describe a parallel approximation algorithm for maximizing monotone submodular functions subject to hereditary constraints on distributed memory multiprocessors. Our work is motivated by the need to solve submodular optimization problems on massive data sets, for practical applications in areas such as data summarization, machine learning, and graph sparsification. Our work builds on the randomized distributed RandGreedI algorithm, proposed by Barbosa, Ene, Nguyen, and Ward (2015). This algorithm computes a distributed solution by randomly partitioning the data among all the processors and then employing a single accumulation step in which all processors send their partial solutions to one processor. However, for large problems, the accumulation step could exceed the memory available on a processor, and the processor which performs the accumulation could become a computational bottleneck.   Here, we propose a generalization of the R
    

