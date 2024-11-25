# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep ReLU Networks Have Surprisingly Simple Polytopes.](http://arxiv.org/abs/2305.09145) | 本文通过计算和分析ReLU网络多面体的单纯形直方图，发现在初始化和梯度下降时它们结构相对简单，这说明了一种新的隐式偏见。 |

# 详细

[^1]: 深层ReLU网络的多面体异常简单

    Deep ReLU Networks Have Surprisingly Simple Polytopes. (arXiv:2305.09145v1 [cs.LG])

    [http://arxiv.org/abs/2305.09145](http://arxiv.org/abs/2305.09145)

    本文通过计算和分析ReLU网络多面体的单纯形直方图，发现在初始化和梯度下降时它们结构相对简单，这说明了一种新的隐式偏见。

    

    ReLU网络是一种多面体上的分段线性函数。研究这种多面体的性质对于神经网络的研究和发展至关重要。目前，对于多面体的理论和实证研究仅停留在计算数量的水平，这远远不能完整地描述多面体。为了将特征提升到一个新的水平，我们提出通过三角剖分多面体得出多面体的形状。通过计算和分析不同多面体的单纯形直方图，我们发现ReLU网络在初始化和梯度下降时具有相对简单的多面体结构，尽管这些多面体从理论上来说可以非常丰富和复杂。这一发现可以被认为是一种新的隐式偏见。随后，我们使用非平凡的组合推导来理论上解释为什么增加深度不会创建更复杂的多面体，通过限制每个维度的平均单纯形数量。

    A ReLU network is a piecewise linear function over polytopes. Figuring out the properties of such polytopes is of fundamental importance for the research and development of neural networks. So far, either theoretical or empirical studies on polytopes only stay at the level of counting their number, which is far from a complete characterization of polytopes. To upgrade the characterization to a new level, here we propose to study the shapes of polytopes via the number of simplices obtained by triangulating the polytope. Then, by computing and analyzing the histogram of simplices across polytopes, we find that a ReLU network has relatively simple polytopes under both initialization and gradient descent, although these polytopes theoretically can be rather diverse and complicated. This finding can be appreciated as a novel implicit bias. Next, we use nontrivial combinatorial derivation to theoretically explain why adding depth does not create a more complicated polytope by bounding the av
    

