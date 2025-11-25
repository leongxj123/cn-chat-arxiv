# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Limits of Assumption-free Tests for Algorithm Performance](https://arxiv.org/abs/2402.07388) | 这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。 |
| [^2] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |

# 详细

[^1]: 无假设测试算法性能的限制

    The Limits of Assumption-free Tests for Algorithm Performance

    [https://arxiv.org/abs/2402.07388](https://arxiv.org/abs/2402.07388)

    这项研究探讨了使用有限数据量回答算法性能问题的基本限制，证明了黑盒测试方法无法准确回答算法在不同训练集上的整体性能和特定模型的性能问题。

    

    算法评价和比较是机器学习和统计学中基本的问题，一个算法在给定的建模任务中表现如何，哪个算法表现最佳？许多方法已经开发出来评估算法性能，通常基于交叉验证策略，将感兴趣的算法在不同的数据子集上重新训练，并评估其在留出数据点上的性能。尽管广泛使用这些程序，但对于这些方法的理论性质尚未完全理解。在这项工作中，我们探讨了在有限的数据量下回答这些问题的一些基本限制。特别地，我们区分了两个问题: 算法$A$在大小为$n$的训练集上学习问题有多好，以及在特定大小为$n$的训练数据集上运行$A$所产生的特定拟合模型有多好？我们的主要结果证明，对于任何将算法视为黑盒的测试方法，无法准确地回答这两个问题。

    Algorithm evaluation and comparison are fundamental questions in machine learning and statistics -- how well does an algorithm perform at a given modeling task, and which algorithm performs best? Many methods have been developed to assess algorithm performance, often based around cross-validation type strategies, retraining the algorithm of interest on different subsets of the data and assessing its performance on the held-out data points. Despite the broad use of such procedures, the theoretical properties of these methods are not yet fully understood. In this work, we explore some fundamental limits for answering these questions with limited amounts of data. In particular, we make a distinction between two questions: how good is an algorithm $A$ at the problem of learning from a training set of size $n$, versus, how good is a particular fitted model produced by running $A$ on a particular training data set of size $n$?   Our main results prove that, for any test that treats the algor
    
[^2]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    

