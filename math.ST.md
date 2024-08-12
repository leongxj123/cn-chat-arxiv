# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Matching via convex relaxation to the simplex.](http://arxiv.org/abs/2310.20609) | 本文提出了一种新的图匹配方法，通过对单位单纯形进行凸松弛，并开发了高效的镜像下降方案来解决该问题。在相关高斯Wigner模型下，单纯形松弛法具有唯一解，并且能够精确恢复地面真实排列。 |

# 详细

[^1]: 通过对单纯形进行凸松弛解决图匹配问题

    Graph Matching via convex relaxation to the simplex. (arXiv:2310.20609v1 [stat.ML])

    [http://arxiv.org/abs/2310.20609](http://arxiv.org/abs/2310.20609)

    本文提出了一种新的图匹配方法，通过对单位单纯形进行凸松弛，并开发了高效的镜像下降方案来解决该问题。在相关高斯Wigner模型下，单纯形松弛法具有唯一解，并且能够精确恢复地面真实排列。

    

    本文针对图匹配问题进行研究，该问题包括在两个输入图之间找到最佳对齐，并在计算机视觉、网络去匿名化和蛋白质对齐等领域有许多应用。解决这个问题的常见方法是通过对NP难问题“二次分配问题”（QAP）进行凸松弛。本文引入了一种新的凸松弛方法，即对单位单纯形进行松弛，并开发了一种具有闭合迭代形式的高效镜像下降方案来解决该问题。在相关高斯Wigner模型下，我们证明了单纯形松弛法在高概率下具有唯一解。在无噪声情况下，这被证明可以精确恢复地面真实排列。此外，我们建立了一种新的输入矩阵假设条件，用于标准贪心取整方法，并且这个条件比常用的“对角线优势”条件更宽松。我们使用这个条件证明了地面真实排列的精确一步恢复。

    This paper addresses the Graph Matching problem, which consists of finding the best possible alignment between two input graphs, and has many applications in computer vision, network deanonymization and protein alignment. A common approach to tackle this problem is through convex relaxations of the NP-hard \emph{Quadratic Assignment Problem} (QAP).  Here, we introduce a new convex relaxation onto the unit simplex and develop an efficient mirror descent scheme with closed-form iterations for solving this problem. Under the correlated Gaussian Wigner model, we show that the simplex relaxation admits a unique solution with high probability. In the noiseless case, this is shown to imply exact recovery of the ground truth permutation. Additionally, we establish a novel sufficiency condition for the input matrix in standard greedy rounding methods, which is less restrictive than the commonly used `diagonal dominance' condition. We use this condition to show exact one-step recovery of the gro
    

