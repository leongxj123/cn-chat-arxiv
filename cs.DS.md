# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SAT Encoding of Partial Ordering Models for Graph Coloring Problems](https://arxiv.org/abs/2403.15961) | 该研究提出了新的SAT编码的偏序模型用于图着色问题，实验结果显示在一些情况下超越了现有的最先进方法，并对带宽着色问题进行了理论分析。 |
| [^2] | [Chasing Convex Functions with Long-term Constraints](https://arxiv.org/abs/2402.14012) | 引入并研究了一类带有长期约束的在线度量问题，提出了在可持续能源和计算系统中在线资源分配应用中的最优竞争算法和学习增强算法，并通过数值实验表现良好。 |
| [^3] | [Intrinsic Data Constraints and Upper Bounds in Binary Classification Performance.](http://arxiv.org/abs/2401.17036) | 我们研究了二元分类性能的内在数据限制和上界，提供了一个理论框架并进行了理论推理和实证检验，发现理论上限是可以被达到的，并计算出了三个常用评估指标的精确上限。 |

# 详细

[^1]: SAT编码的偏序模型用于图着色问题

    SAT Encoding of Partial Ordering Models for Graph Coloring Problems

    [https://arxiv.org/abs/2403.15961](https://arxiv.org/abs/2403.15961)

    该研究提出了新的SAT编码的偏序模型用于图着色问题，实验结果显示在一些情况下超越了现有的最先进方法，并对带宽着色问题进行了理论分析。

    

    在本文中，我们提出了基于偏序的整数线性规划（ILP）模型的图着色问题（GCP）和带宽着色问题（BCP）的新SAT编码。 GCP要求给定图的顶点分配最少数量的颜色，以便每两个相邻的顶点得到不同的颜色。 BCP是一个泛化问题，其中每条边都有一个权重，要求分配的颜色之间有最小的“距离”，目标是最小化使用的“最大”颜色。 对于被广泛研究的GCP，我们在DIMACS基准集上实验比较了我们新的SAT编码与现有最先进方法。 我们的评估证实，这种SAT编码对于稀疏图是有效的，并且甚至在一些DIMACS示例上胜过了现有最先进方法。 对于BCP，我们的理论分析表明，基于偏序的SAT和ILP公式的大小在渐近意义下小于经典的解法。

    arXiv:2403.15961v1 Announce Type: new  Abstract: In this paper, we suggest new SAT encodings of the partial-ordering based ILP model for the graph coloring problem (GCP) and the bandwidth coloring problem (BCP). The GCP asks for the minimum number of colors that can be assigned to the vertices of a given graph such that each two adjacent vertices get different colors. The BCP is a generalization, where each edge has a weight that enforces a minimal "distance" between the assigned colors, and the goal is to minimize the "largest" color used. For the widely studied GCP, we experimentally compare our new SAT encoding to the state-of-the-art approaches on the DIMACS benchmark set. Our evaluation confirms that this SAT encoding is effective for sparse graphs and even outperforms the state-of-the-art on some DIMACS instances. For the BCP, our theoretical analysis shows that the partial-ordering based SAT and ILP formulations have an asymptotically smaller size than that of the classical assi
    
[^2]: 在满足长期约束条件下追逐凸函数

    Chasing Convex Functions with Long-term Constraints

    [https://arxiv.org/abs/2402.14012](https://arxiv.org/abs/2402.14012)

    引入并研究了一类带有长期约束的在线度量问题，提出了在可持续能源和计算系统中在线资源分配应用中的最优竞争算法和学习增强算法，并通过数值实验表现良好。

    

    我们引入并研究了一类带有长期约束的在线度量问题。在这些问题中，一个在线玩家在度量空间$(X,d)$中做出决策$\mathbf{x}_t$，同时最小化他们的命中成本$f_t(\mathbf{x}_t)$和由度量确定的切换成本。在时间跨度$T$内，玩家必须满足长期需求约束$\sum_{t} c(\mathbf{x}_t) \geq 1$，其中$c(\mathbf{x}_t)$表示时间$t$时满足的需求比例。这类问题在可持续能源和计算系统中的在线资源分配中有着广泛的应用。我们为这些问题的具体实例设计了最优的竞争算法和学习增强算法，并进一步展示了我们提出的算法在数值实验中表现良好。

    arXiv:2402.14012v1 Announce Type: cross  Abstract: We introduce and study a family of online metric problems with long-term constraints. In these problems, an online player makes decisions $\mathbf{x}_t$ in a metric space $(X,d)$ to simultaneously minimize their hitting cost $f_t(\mathbf{x}_t)$ and switching cost as determined by the metric. Over the time horizon $T$, the player must satisfy a long-term demand constraint $\sum_{t} c(\mathbf{x}_t) \geq 1$, where $c(\mathbf{x}_t)$ denotes the fraction of demand satisfied at time $t$. Such problems can find a wide array of applications to online resource allocation in sustainable energy and computing systems. We devise optimal competitive and learning-augmented algorithms for specific instantiations of these problems, and further show that our proposed algorithms perform well in numerical experiments.
    
[^3]: 二元分类性能中的内在数据限制和上界

    Intrinsic Data Constraints and Upper Bounds in Binary Classification Performance. (arXiv:2401.17036v1 [cs.LG])

    [http://arxiv.org/abs/2401.17036](http://arxiv.org/abs/2401.17036)

    我们研究了二元分类性能的内在数据限制和上界，提供了一个理论框架并进行了理论推理和实证检验，发现理论上限是可以被达到的，并计算出了三个常用评估指标的精确上限。

    

    数据组织的结构被广泛认为对机器学习算法的有效性有重要影响，尤其是在二元分类任务中。我们的研究提供了一个理论框架，认为给定数据集上二元分类器的最大潜力主要受到数据的内在特性的限制。通过理论推理和实证检验，我们使用了标准目标函数、评估指标和二元分类器，得出了两个主要结论。首先，我们证明了在实际数据集上二元分类性能的理论上限是可以被理论上达到的。这个上限代表了学习损失和评估指标之间的可计算平衡。其次，我们计算了三个常用评估指标的精确上限，揭示了与我们总体论点的根本一致性：上界与内在数据限制密切相关。

    The structure of data organization is widely recognized as having a substantial influence on the efficacy of machine learning algorithms, particularly in binary classification tasks. Our research provides a theoretical framework suggesting that the maximum potential of binary classifiers on a given dataset is primarily constrained by the inherent qualities of the data. Through both theoretical reasoning and empirical examination, we employed standard objective functions, evaluative metrics, and binary classifiers to arrive at two principal conclusions. Firstly, we show that the theoretical upper bound of binary classification performance on actual datasets can be theoretically attained. This upper boundary represents a calculable equilibrium between the learning loss and the metric of evaluation. Secondly, we have computed the precise upper bounds for three commonly used evaluation metrics, uncovering a fundamental uniformity with our overarching thesis: the upper bound is intricately 
    

