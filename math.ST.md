# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Memorization with neural nets: going beyond the worst case.](http://arxiv.org/abs/2310.00327) | 本文研究了神经网络的插值问题，提出了一种简单的随机算法，在给定的数据集和两个类的情况下，能够以很高的概率构建一个插值的神经网络。这些结果与训练数据规模无关。 |

# 详细

[^1]: 神经网络的记忆化：超越最坏情况

    Memorization with neural nets: going beyond the worst case. (arXiv:2310.00327v1 [stat.ML])

    [http://arxiv.org/abs/2310.00327](http://arxiv.org/abs/2310.00327)

    本文研究了神经网络的插值问题，提出了一种简单的随机算法，在给定的数据集和两个类的情况下，能够以很高的概率构建一个插值的神经网络。这些结果与训练数据规模无关。

    

    在实践中，深度神经网络通常能够轻松地插值其训练数据。为了理解这一现象，许多研究都旨在量化神经网络架构的记忆能力：即在任意放置这些点并任意分配标签的情况下，架构能够插值的最大点数。然而，对于实际数据，人们直觉地期望存在一种良性结构，使得插值在比记忆能力建议的较小网络尺寸上已经发生。在本文中，我们通过采用实例特定的观点来研究插值。我们引入了一个简单的随机算法，它可以在多项式时间内给定一个固定的有限数据集和两个类的情况下，以很高的概率构建出一个插值三层神经网络。所需的参数数量与这两个类的几何特性及其相互排列有关。因此，我们获得了与训练数据规模无关的保证。

    In practice, deep neural networks are often able to easily interpolate their training data. To understand this phenomenon, many works have aimed to quantify the memorization capacity of a neural network architecture: the largest number of points such that the architecture can interpolate any placement of these points with any assignment of labels. For real-world data, however, one intuitively expects the presence of a benign structure so that interpolation already occurs at a smaller network size than suggested by memorization capacity. In this paper, we investigate interpolation by adopting an instance-specific viewpoint. We introduce a simple randomized algorithm that, given a fixed finite dataset with two classes, with high probability constructs an interpolating three-layer neural network in polynomial time. The required number of parameters is linked to geometric properties of the two classes and their mutual arrangement. As a result, we obtain guarantees that are independent of t
    

