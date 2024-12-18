# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extremal graphical modeling with latent variables](https://arxiv.org/abs/2403.09604) | 提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。 |
| [^2] | [Sample-Efficient Clustering and Conquer Procedures for Parallel Large-Scale Ranking and Selection](https://arxiv.org/abs/2402.02196) | 我们提出了一种新颖的并行大规模排序和选择问题的聚类及征服方法，通过利用相关信息进行聚类以提高样本效率，在大规模AI应用中表现优异。 |

# 详细

[^1]: 混合变量的极端图模型

    Extremal graphical modeling with latent variables

    [https://arxiv.org/abs/2403.09604](https://arxiv.org/abs/2403.09604)

    提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。

    

    极端图模型编码多变量极端条件独立结构，并为量化罕见事件风险提供强大工具。我们提出了面向潜变量的可延伸图模型的可行凸规划方法，将 H\"usler-Reiss 精度矩阵分解为编码观察变量之间的图结构的稀疏部分和编码少量潜变量对观察变量的影响的低秩部分。我们提供了\texttt{eglatent}的有限样本保证，并展示它能一致地恢复条件图以及潜变量的数量。

    arXiv:2403.09604v1 Announce Type: cross  Abstract: Extremal graphical models encode the conditional independence structure of multivariate extremes and provide a powerful tool for quantifying the risk of rare events. Prior work on learning these graphs from data has focused on the setting where all relevant variables are observed. For the popular class of H\"usler-Reiss models, we propose the \texttt{eglatent} method, a tractable convex program for learning extremal graphical models in the presence of latent variables. Our approach decomposes the H\"usler-Reiss precision matrix into a sparse component encoding the graphical structure among the observed variables after conditioning on the latent variables, and a low-rank component encoding the effect of a few latent variables on the observed variables. We provide finite-sample guarantees of \texttt{eglatent} and show that it consistently recovers the conditional graph as well as the number of latent variables. We highlight the improved 
    
[^2]: 并行大规模排序和选择问题的样本高效聚类及征服方法

    Sample-Efficient Clustering and Conquer Procedures for Parallel Large-Scale Ranking and Selection

    [https://arxiv.org/abs/2402.02196](https://arxiv.org/abs/2402.02196)

    我们提出了一种新颖的并行大规模排序和选择问题的聚类及征服方法，通过利用相关信息进行聚类以提高样本效率，在大规模AI应用中表现优异。

    

    我们提出了一种新颖的"聚类和征服"方法，用于解决并行大规模排序和选择问题，通过利用相关信息进行聚类，以打破样本效率的瓶颈。在并行计算环境中，基于相关性的聚类可以实现O(p)的样本复杂度减少速度，这是理论上可达到的最佳减少速度。我们提出的框架是通用的，在固定预算和固定精度的范式下，可以无缝集成各种常见的排序和选择方法。它可以在无需高精确度相关估计和精确聚类的情况下实现改进。在大规模人工智能应用中，如神经结构搜索，我们的无筛选版本的方法惊人地超过了完全顺序化的基准，表现出更高的样本效率。这表明利用有价值的结构信息，如相关性，是绕过传统方法的一条可行路径。

    We propose novel "clustering and conquer" procedures for the parallel large-scale ranking and selection (R&S) problem, which leverage correlation information for clustering to break the bottleneck of sample efficiency. In parallel computing environments, correlation-based clustering can achieve an $\mathcal{O}(p)$ sample complexity reduction rate, which is the optimal reduction rate theoretically attainable. Our proposed framework is versatile, allowing for seamless integration of various prevalent R&S methods under both fixed-budget and fixed-precision paradigms. It can achieve improvements without the necessity of highly accurate correlation estimation and precise clustering. In large-scale AI applications such as neural architecture search, a screening-free version of our procedure surprisingly surpasses fully-sequential benchmarks in terms of sample efficiency. This suggests that leveraging valuable structural information, such as correlation, is a viable path to bypassing the trad
    

