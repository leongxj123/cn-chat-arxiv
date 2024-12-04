# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Descriptive Complexity of Graph Neural Networks.](http://arxiv.org/abs/2303.04613) | 研究分析了图神经网络（GNN）在布尔电路复杂性和描述性复杂性方面的能力，证明了多项式规模有界深度的GNN族族可以计算的图查询正是带计数和内置关系的一阶逻辑受保护的片断GFO+C所定义的，这将GNN放在电路复杂性类TC^0中。 |

# 详细

[^1]: 图神经网络的描述性复杂性

    The Descriptive Complexity of Graph Neural Networks. (arXiv:2303.04613v2 [cs.LO] UPDATED)

    [http://arxiv.org/abs/2303.04613](http://arxiv.org/abs/2303.04613)

    研究分析了图神经网络（GNN）在布尔电路复杂性和描述性复杂性方面的能力，证明了多项式规模有界深度的GNN族族可以计算的图查询正是带计数和内置关系的一阶逻辑受保护的片断GFO+C所定义的，这将GNN放在电路复杂性类TC^0中。

    

    我们分析了图神经网络（GNN）的布尔电路复杂性和描述性复杂性的能力。我们证明了多项式规模有界深度的GNN族族可以计算的图查询正是那些用带计数和内置关系的一阶逻辑受保护的片断GFO+C定义的。这将GNN放在电路复杂性类TC^0中。值得注意的是，GNN家族可以使用任意实数权值和包括标准ReLU、Logistic“sigmod”和双曲正切函数在内的广泛激活函数类。如果GNN被允许使用随机初始化和全局读取（这些都是GNN在实践中广泛使用的标准功能），它们可以计算与阈门的有界深度布尔电路完全相同的查询，即在TC^0中的查询。此外，我们展示了一个带分段线性激活和有理权重的单个GNN可以在不建造内部关系的情况下由GFO+C定义。

    We analyse the power of graph neural networks (GNNs) in terms of Boolean circuit complexity and descriptive complexity.  We prove that the graph queries that can be computed by a polynomial-size bounded-depth family of GNNs are exactly those definable in the guarded fragment GFO+C of first-order logic with counting and with built-in relations. This puts GNNs in the circuit complexity class TC^0. Remarkably, the GNN families may use arbitrary real weights and a wide class of activation functions that includes the standard ReLU, logistic "sigmod", and hyperbolic tangent functions. If the GNNs are allowed to use random initialisation and global readout (both standard features of GNNs widely used in practice), they can compute exactly the same queries as bounded depth Boolean circuits with threshold gates, that is, exactly the queries in TC^0.  Moreover, we show that queries computable by a single GNN with piecewise linear activations and rational weights are definable in GFO+C without bui
    

