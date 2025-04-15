# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extremal graphical modeling with latent variables](https://arxiv.org/abs/2403.09604) | 提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。 |
| [^2] | [Understanding Best Subset Selection: A Tale of Two C(omplex)ities.](http://arxiv.org/abs/2301.06259) | 本文研究了最佳子集选择在高维稀疏线性回归设置中的变量选择性质，通过研究残差化特征和虚假投影的复杂性来揭示模型一致性的边界条件。 |

# 详细

[^1]: 混合变量的极端图模型

    Extremal graphical modeling with latent variables

    [https://arxiv.org/abs/2403.09604](https://arxiv.org/abs/2403.09604)

    提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。

    

    极端图模型编码多变量极端条件独立结构，并为量化罕见事件风险提供强大工具。我们提出了面向潜变量的可延伸图模型的可行凸规划方法，将 H\"usler-Reiss 精度矩阵分解为编码观察变量之间的图结构的稀疏部分和编码少量潜变量对观察变量的影响的低秩部分。我们提供了\texttt{eglatent}的有限样本保证，并展示它能一致地恢复条件图以及潜变量的数量。

    arXiv:2403.09604v1 Announce Type: cross  Abstract: Extremal graphical models encode the conditional independence structure of multivariate extremes and provide a powerful tool for quantifying the risk of rare events. Prior work on learning these graphs from data has focused on the setting where all relevant variables are observed. For the popular class of H\"usler-Reiss models, we propose the \texttt{eglatent} method, a tractable convex program for learning extremal graphical models in the presence of latent variables. Our approach decomposes the H\"usler-Reiss precision matrix into a sparse component encoding the graphical structure among the observed variables after conditioning on the latent variables, and a low-rank component encoding the effect of a few latent variables on the observed variables. We provide finite-sample guarantees of \texttt{eglatent} and show that it consistently recovers the conditional graph as well as the number of latent variables. We highlight the improved 
    
[^2]: 了解最佳子集选择: 两种复杂性的故事

    Understanding Best Subset Selection: A Tale of Two C(omplex)ities. (arXiv:2301.06259v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2301.06259](http://arxiv.org/abs/2301.06259)

    本文研究了最佳子集选择在高维稀疏线性回归设置中的变量选择性质，通过研究残差化特征和虚假投影的复杂性来揭示模型一致性的边界条件。

    

    几十年来，最佳子集选择(BSS)主要由于计算瓶颈而困扰统计学家。然而，直到最近，现代计算突破重新点燃了对BSS的理论兴趣并导致了新的发现。最近，Guo等人表明，BSS的模型选择性能受到了鲁棒性设计依赖的边界量的控制，不像LASSO、SCAD、MCP等现代方法。在他们的理论结果的激励下，本文还研究了高维稀疏线性回归设置下最佳子集选择的变量选择性质。我们展示了除了可辨识性边界以外，下列两种复杂性度量在表征模型一致性边界条件中起着基本的作用：(a)“残差化特征”的复杂性，(b)“虚假投影”的复杂性。特别地，我们建立了一个仅依赖于可辨识性边界的简单边界条件。

    For decades, best subset selection (BSS) has eluded statisticians mainly due to its computational bottleneck. However, until recently, modern computational breakthroughs have rekindled theoretical interest in BSS and have led to new findings. Recently, \cite{guo2020best} showed that the model selection performance of BSS is governed by a margin quantity that is robust to the design dependence, unlike modern methods such as LASSO, SCAD, MCP, etc. Motivated by their theoretical results, in this paper, we also study the variable selection properties of best subset selection for high-dimensional sparse linear regression setup. We show that apart from the identifiability margin, the following two complexity measures play a fundamental role in characterizing the margin condition for model consistency: (a) complexity of \emph{residualized features}, (b) complexity of \emph{spurious projections}. In particular, we establish a simple margin condition that depends only on the identifiability mar
    

