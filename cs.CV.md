# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonlinearity Enhanced Adaptive Activation Function](https://arxiv.org/abs/2403.19896) | 引入了一种带有甚至立方非线性的简单实现激活函数，通过引入可优化参数使得激活函数具有更大的自由度，可以提高神经网络的准确性，同时不需要太多额外的计算资源。 |
| [^2] | [Equipping Sketch Patches with Context-Aware Positional Encoding for Graphic Sketch Representation](https://arxiv.org/abs/2403.17525) | 提出了一种通过为素描补丁配备上下文感知的位置编码来保护不同绘图版本的方法，将绘图顺序信息嵌入图节点中，以更好地学习图形素描表示。 |

# 详细

[^1]: 非线性增强自适应激活函数

    Nonlinearity Enhanced Adaptive Activation Function

    [https://arxiv.org/abs/2403.19896](https://arxiv.org/abs/2403.19896)

    引入了一种带有甚至立方非线性的简单实现激活函数，通过引入可优化参数使得激活函数具有更大的自由度，可以提高神经网络的准确性，同时不需要太多额外的计算资源。

    

    引入一种简单实现的激活函数，具有甚至立方非线性，可以提高神经网络的准确性，而不需要太多额外的计算资源。通过一种明显的收敛与准确性之间的权衡来实现。该激活函数通过引入可优化参数来泛化标准RELU函数，从而增加了额外的自由度，使得非线性程度可以被调整。通过与标准技术进行比较，将在MNIST数字数据集的背景下量化相关准确性的提升。

    arXiv:2403.19896v1 Announce Type: new  Abstract: A simply implemented activation function with even cubic nonlinearity is introduced that increases the accuracy of neural networks without substantial additional computational resources. This is partially enabled through an apparent tradeoff between convergence and accuracy. The activation function generalizes the standard RELU function by introducing additional degrees of freedom through optimizable parameters that enable the degree of nonlinearity to be adjusted. The associated accuracy enhancement is quantified in the context of the MNIST digit data set through a comparison with standard techniques.
    
[^2]: 为图形素描表示装备具有上下文感知的位置编码的素描补丁

    Equipping Sketch Patches with Context-Aware Positional Encoding for Graphic Sketch Representation

    [https://arxiv.org/abs/2403.17525](https://arxiv.org/abs/2403.17525)

    提出了一种通过为素描补丁配备上下文感知的位置编码来保护不同绘图版本的方法，将绘图顺序信息嵌入图节点中，以更好地学习图形素描表示。

    

    一幅素描的绘制顺序记录了它是如何逐笔由人类创建的。对于图形素描表示学习，最近的研究通过根据基于时间的最近邻策略将每个补丁与另一个相连，将素描绘图顺序注入到图边构建中。然而，这样构建的图边可能不可靠，因为素描可能有不同版本的绘图。在本文中，我们提出了一种经过变体绘制保护的方法，通过为素描补丁配备具有上下文感知的位置编码(PE)，以更好地利用绘图顺序来学习图形素描表示。我们没有将素描绘制注入到图边中，而是仅将这些顺序信息嵌入到图节点中。具体来说，每个补丁嵌入都配备有正弦绝对PE，以突出绘图顺序中的顺序位置。它的相邻补丁按self-att的价值排序

    arXiv:2403.17525v1 Announce Type: cross  Abstract: The drawing order of a sketch records how it is created stroke-by-stroke by a human being. For graphic sketch representation learning, recent studies have injected sketch drawing orders into graph edge construction by linking each patch to another in accordance to a temporal-based nearest neighboring strategy. However, such constructed graph edges may be unreliable, since a sketch could have variants of drawings. In this paper, we propose a variant-drawing-protected method by equipping sketch patches with context-aware positional encoding (PE) to make better use of drawing orders for learning graphic sketch representation. Instead of injecting sketch drawings into graph edges, we embed these sequential information into graph nodes only. More specifically, each patch embedding is equipped with a sinusoidal absolute PE to highlight the sequential position in the drawing order. And its neighboring patches, ranked by the values of self-att
    

