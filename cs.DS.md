# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Capacity Provisioning Motivated Online Non-Convex Optimization Problem with Memory and Switching Cost](https://arxiv.org/abs/2403.17480) | 该论文考虑了一种在线非凸优化问题，目标是通过调节活动服务器数量最小化作业延迟，引入了切换成本，提出了竞争算法。 |
| [^2] | [A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation](https://arxiv.org/abs/2402.03358) | 这篇综述调研了图缩减方法，包括稀疏化、粗化和浓缩，在解决大型图形数据分析和计算复杂性方面起到了重要作用。调研对这些方法的技术细节进行了系统的回顾，并强调了它们在实际应用中的关键性。同时，调研还提出了保证图缩减技术持续有效性的关键研究方向。 |
| [^3] | [Connectivity Oracles for Predictable Vertex Failures](https://arxiv.org/abs/2312.08489) | 论文研究了在预测算法范式下设计支持顶点失败的连通性预测器的问题，并提出了一种数据结构，能够以预处理时间和查询时间的多项式关系来处理失败顶点集合。 |
| [^4] | [Accelerated Algorithms for Constrained Nonconvex-Nonconcave Min-Max Optimization and Comonotone Inclusion](https://arxiv.org/abs/2206.05248) | 本论文提出了针对约束共单调极小-极大优化和共单调包含问题的加速算法，扩展了现有算法并实现了较优的收敛速率，同时证明了算法的收敛性。 |
| [^5] | [Autumn: A Scalable Read Optimized LSM-tree based Key-Value Stores with Fast Point and Range Read Speed.](http://arxiv.org/abs/2305.05074) | Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎，其创新之处在于通过动态调整相邻两层之间的容量比来不断提高读性能，使得点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$。 |

# 详细

[^1]: 具有内存和切换成本的在线非凸优化问题的容量调配动机

    Capacity Provisioning Motivated Online Non-Convex Optimization Problem with Memory and Switching Cost

    [https://arxiv.org/abs/2403.17480](https://arxiv.org/abs/2403.17480)

    该论文考虑了一种在线非凸优化问题，目标是通过调节活动服务器数量最小化作业延迟，引入了切换成本，提出了竞争算法。

    

    考虑了一种在线非凸优化问题，其目标是通过调节活动服务器的数量来最小化一组作业的流量时间（总延迟），但在时间变化时改变活动服务器数量会产生切换成本。每个作业在任何时间内最多可以由一个固定速度的服务器处理。与通常具有切换成本的在线凸优化（OCO）问题相比，所考虑的目标函数是非凸的，并且更重要的是，在每个时间点，它取决于所有过去的决策，而不仅仅是当前的决策。考虑了最坏情况和随机输入；对于这两种情况，提出了竞争算法。

    arXiv:2403.17480v1 Announce Type: cross  Abstract: An online non-convex optimization problem is considered where the goal is to minimize the flow time (total delay) of a set of jobs by modulating the number of active servers, but with a switching cost associated with changing the number of active servers over time. Each job can be processed by at most one fixed speed server at any time. Compared to the usual online convex optimization (OCO) problem with switching cost, the objective function considered is non-convex and more importantly, at each time, it depends on all past decisions and not just the present one. Both worst-case and stochastic inputs are considered; for both cases, competitive algorithms are derived.
    
[^2]: 图缩减的综合调研：稀疏化、粗化和浓缩

    A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation

    [https://arxiv.org/abs/2402.03358](https://arxiv.org/abs/2402.03358)

    这篇综述调研了图缩减方法，包括稀疏化、粗化和浓缩，在解决大型图形数据分析和计算复杂性方面起到了重要作用。调研对这些方法的技术细节进行了系统的回顾，并强调了它们在实际应用中的关键性。同时，调研还提出了保证图缩减技术持续有效性的关键研究方向。

    

    许多真实世界的数据集可以自然地表示为图，涵盖了广泛的领域。然而，图数据集的复杂性和规模的增加为分析和计算带来了显著的挑战。为此，图缩减技术在保留关键属性的同时简化大型图形数据变得越来越受关注。在本调研中，我们旨在提供对图缩减方法的全面理解，包括图稀疏化、图粗化和图浓缩。具体而言，我们建立了这些方法的统一定义，并引入了一个分层分类法来分类这些方法所解决的挑战。我们的调研系统地回顾了这些方法的技术细节，并强调了它们在各种场景中的实际应用。此外，我们还概述了保证图缩减技术持续有效性的关键研究方向，并提供了一个详细的论文列表链接。

    Many real-world datasets can be naturally represented as graphs, spanning a wide range of domains. However, the increasing complexity and size of graph datasets present significant challenges for analysis and computation. In response, graph reduction techniques have gained prominence for simplifying large graphs while preserving essential properties. In this survey, we aim to provide a comprehensive understanding of graph reduction methods, including graph sparsification, graph coarsening, and graph condensation. Specifically, we establish a unified definition for these methods and introduce a hierarchical taxonomy to categorize the challenges they address. Our survey then systematically reviews the technical details of these methods and emphasizes their practical applications across diverse scenarios. Furthermore, we outline critical research directions to ensure the continued effectiveness of graph reduction techniques, as well as provide a comprehensive paper list at https://github.
    
[^3]: 预测顶点失败的连通性预测器

    Connectivity Oracles for Predictable Vertex Failures

    [https://arxiv.org/abs/2312.08489](https://arxiv.org/abs/2312.08489)

    论文研究了在预测算法范式下设计支持顶点失败的连通性预测器的问题，并提出了一种数据结构，能够以预处理时间和查询时间的多项式关系来处理失败顶点集合。

    

    设计支持顶点失败的连通性预测器是针对无向图的基本数据结构问题之一。已有的研究在查询时间方面已经有了很好的理解：以前的作品[Duan-Pettie STOC'10; Long-Saranurak FOCS'22]实现了与失败顶点数量成线性关系的查询时间，并且在需要多项式时间的预处理和多项式时间的更新的条件下是有条件最优的。我们在预测算法的范式下重新审视了这个问题：我们问，如果可以预测到失败顶点集合，查询时间是否可以提高。更具体地说，我们设计了一个数据结构，给定一个图G=(V,E)和一个预测会失败的顶点集合\widehat{D} \subseteq V（其中d=|\widehat{D}|），将其预处理时间为$\tilde{O}(d|E|)$，然后可以接收一个更新，该更新以对称差分形式给出。

    arXiv:2312.08489v2 Announce Type: replace-cross  Abstract: The problem of designing connectivity oracles supporting vertex failures is one of the basic data structures problems for undirected graphs. It is already well understood: previous works [Duan--Pettie STOC'10; Long--Saranurak FOCS'22] achieve query time linear in the number of failed vertices, and it is conditionally optimal as long as we require preprocessing time polynomial in the size of the graph and update time polynomial in the number of failed vertices.   We revisit this problem in the paradigm of algorithms with predictions: we ask if the query time can be improved if the set of failed vertices can be predicted beforehand up to a small number of errors. More specifically, we design a data structure that, given a graph $G=(V,E)$ and a set of vertices predicted to fail $\widehat{D} \subseteq V$ of size $d=|\widehat{D}|$, preprocesses it in time $\tilde{O}(d|E|)$ and then can receive an update given as the symmetric differ
    
[^4]: 加速算法用于约束非凸-非凹极小-极大优化和共单调包含

    Accelerated Algorithms for Constrained Nonconvex-Nonconcave Min-Max Optimization and Comonotone Inclusion

    [https://arxiv.org/abs/2206.05248](https://arxiv.org/abs/2206.05248)

    本论文提出了针对约束共单调极小-极大优化和共单调包含问题的加速算法，扩展了现有算法并实现了较优的收敛速率，同时证明了算法的收敛性。

    

    我们研究了约束共单调极小-极大优化，一类结构化的非凸-非凹极小-极大优化问题以及它们对共单调包含的推广。在我们的第一个贡献中，我们将最初由Yoon和Ryu（2021）提出的无约束极小-极大优化的Extra Anchored Gradient（EAG）算法扩展到约束共单调极小-极大优化和共单调包含问题，并实现了所有一阶方法中的最优收敛速率$O\left(\frac{1}{T}\right)$。此外，我们证明了算法的迭代收敛到解集中的一个点。在我们的第二个贡献中，我们将由Lee和Kim（2021）开发的快速额外梯度（FEG）算法扩展到约束共单调极小-极大优化和共单调包含，并实现了相同的$O\left(\frac{1}{T}\right)$收敛速率。这个速率适用于文献中研究过的最广泛的共单调包含问题集合。我们的分析基于s的内容。

    We study constrained comonotone min-max optimization, a structured class of nonconvex-nonconcave min-max optimization problems, and their generalization to comonotone inclusion. In our first contribution, we extend the Extra Anchored Gradient (EAG) algorithm, originally proposed by Yoon and Ryu (2021) for unconstrained min-max optimization, to constrained comonotone min-max optimization and comonotone inclusion, achieving an optimal convergence rate of $O\left(\frac{1}{T}\right)$ among all first-order methods. Additionally, we prove that the algorithm's iterations converge to a point in the solution set. In our second contribution, we extend the Fast Extra Gradient (FEG) algorithm, as developed by Lee and Kim (2021), to constrained comonotone min-max optimization and comonotone inclusion, achieving the same $O\left(\frac{1}{T}\right)$ convergence rate. This rate is applicable to the broadest set of comonotone inclusion problems yet studied in the literature. Our analyses are based on s
    
[^5]: Autumn：基于LSM-tree的可扩展的面向读操作优化的键值存储引擎

    Autumn: A Scalable Read Optimized LSM-tree based Key-Value Stores with Fast Point and Range Read Speed. (arXiv:2305.05074v1 [cs.DB])

    [http://arxiv.org/abs/2305.05074](http://arxiv.org/abs/2305.05074)

    Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎，其创新之处在于通过动态调整相邻两层之间的容量比来不断提高读性能，使得点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$。

    

    基于Log Structured Merge Trees (LSM-tree)的键值存储引擎被广泛应用于许多存储系统中，以支持更新、点读和区间读等各种操作。本文中，我们提出了一个名为Autumn的可扩展的、面向读操作优化的基于LSM-tree的键值存储引擎，它具有最少的点读和区间读成本。通过动态调整相邻两层之间的容量比来不断提高读性能，点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$，并应用了新的Garnering合并策略。Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎。

    The Log Structured Merge Trees (LSM-tree) based key-value stores are widely used in many storage systems to support a variety of operations such as updates, point reads, and range reads. Traditionally, LSM-tree's merge policy organizes data into multiple levels of exponentially increasing capacity to support high-speed writes. However, we contend that the traditional merge policies are not optimized for reads. In this work, we present Autumn, a scalable and read optimized LSM-tree based key-value stores with minimal point and range read cost. The key idea in improving the read performance is to dynamically adjust the capacity ratio between two adjacent levels as more data are stored. As a result, smaller levels gradually increase their capacities and merge more often. In particular, the point and range read cost improves from the previous best known $O(logN)$ complexity to $O(\sqrt{logN})$ in Autumn by applying the new novel Garnering merge policy. While Garnering merge policy optimize
    

