# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SAT Encoding of Partial Ordering Models for Graph Coloring Problems](https://arxiv.org/abs/2403.15961) | 该研究提出了新的SAT编码的偏序模型用于图着色问题，实验结果显示在一些情况下超越了现有的最先进方法，并对带宽着色问题进行了理论分析。 |

# 详细

[^1]: SAT编码的偏序模型用于图着色问题

    SAT Encoding of Partial Ordering Models for Graph Coloring Problems

    [https://arxiv.org/abs/2403.15961](https://arxiv.org/abs/2403.15961)

    该研究提出了新的SAT编码的偏序模型用于图着色问题，实验结果显示在一些情况下超越了现有的最先进方法，并对带宽着色问题进行了理论分析。

    

    在本文中，我们提出了基于偏序的整数线性规划（ILP）模型的图着色问题（GCP）和带宽着色问题（BCP）的新SAT编码。 GCP要求给定图的顶点分配最少数量的颜色，以便每两个相邻的顶点得到不同的颜色。 BCP是一个泛化问题，其中每条边都有一个权重，要求分配的颜色之间有最小的“距离”，目标是最小化使用的“最大”颜色。 对于被广泛研究的GCP，我们在DIMACS基准集上实验比较了我们新的SAT编码与现有最先进方法。 我们的评估证实，这种SAT编码对于稀疏图是有效的，并且甚至在一些DIMACS示例上胜过了现有最先进方法。 对于BCP，我们的理论分析表明，基于偏序的SAT和ILP公式的大小在渐近意义下小于经典的解法。

    arXiv:2403.15961v1 Announce Type: new  Abstract: In this paper, we suggest new SAT encodings of the partial-ordering based ILP model for the graph coloring problem (GCP) and the bandwidth coloring problem (BCP). The GCP asks for the minimum number of colors that can be assigned to the vertices of a given graph such that each two adjacent vertices get different colors. The BCP is a generalization, where each edge has a weight that enforces a minimal "distance" between the assigned colors, and the goal is to minimize the "largest" color used. For the widely studied GCP, we experimentally compare our new SAT encoding to the state-of-the-art approaches on the DIMACS benchmark set. Our evaluation confirms that this SAT encoding is effective for sparse graphs and even outperforms the state-of-the-art on some DIMACS instances. For the BCP, our theoretical analysis shows that the partial-ordering based SAT and ILP formulations have an asymptotically smaller size than that of the classical assi
    

