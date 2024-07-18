# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mode Connectivity in Auction Design.](http://arxiv.org/abs/2305.11005) | 该论文研究了拍卖设计领域的一个基本问题，即最优拍卖设计。在研究中，作者证明了神经网络在一定条件下可以通过简单的分段线性路径连接不同的局部最优解，并取得了成功。 |
| [^2] | [Towards Lower Bounds on the Depth of ReLU Neural Networks.](http://arxiv.org/abs/2105.14835) | 该研究运用数学和优化理论方法，就 ReLU 神经网络的深度下界做了探究，有助于更好地理解这种网络所能表示的函数类的性质。此外，该研究还肯定了一项旧的分段线性函数猜想。 |

# 详细

[^1]: 拍卖设计中的模式连通性

    Mode Connectivity in Auction Design. (arXiv:2305.11005v1 [cs.GT])

    [http://arxiv.org/abs/2305.11005](http://arxiv.org/abs/2305.11005)

    该论文研究了拍卖设计领域的一个基本问题，即最优拍卖设计。在研究中，作者证明了神经网络在一定条件下可以通过简单的分段线性路径连接不同的局部最优解，并取得了成功。

    

    最优拍卖设计是算法博弈论中的一个基本问题，即使在非常简单的情况下，这个问题也很难。最近不同的经济学可微分理论表明，神经网络可以有效地学习已知的最优拍卖机制，发现有趣的新机制。为了理论上证明它们的实证成功，我们聚焦于第一个这样的网络，RochetNet，并研究所谓的仿射极大化拍卖的广义版本。我们证明它们满足模式连通性，即局部最优解通过一个简单的分段线性路径连接，路径上的每个解都几乎和两个局部最优解之一一样好。模式连通性最近被证明是神经网络用于预测问题的一个有趣的经验和理论的属性。我们的结果是对可微分经济学领域中神经网络用于解决非线性设计问题的第一个这样的分析。

    Optimal auction design is a fundamental problem in algorithmic game theory. This problem is notoriously difficult already in very simple settings. Recent work in differentiable economics showed that neural networks can efficiently learn known optimal auction mechanisms and discover interesting new ones. In an attempt to theoretically justify their empirical success, we focus on one of the first such networks, RochetNet, and a generalized version for affine maximizer auctions. We prove that they satisfy mode connectivity, i.e., locally optimal solutions are connected by a simple, piecewise linear path such that every solution on the path is almost as good as one of the two local optima. Mode connectivity has been recently investigated as an intriguing empirical and theoretically justifiable property of neural networks used for prediction problems. Our results give the first such analysis in the context of differentiable economics, where neural networks are used directly for solving non-
    
[^2]: 关于 ReLU 神经网络深度下界的探究

    Towards Lower Bounds on the Depth of ReLU Neural Networks. (arXiv:2105.14835v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2105.14835](http://arxiv.org/abs/2105.14835)

    该研究运用数学和优化理论方法，就 ReLU 神经网络的深度下界做了探究，有助于更好地理解这种网络所能表示的函数类的性质。此外，该研究还肯定了一项旧的分段线性函数猜想。

    

    我们运用混合整数优化、多面体理论和热带几何学等技术，为理解具有 ReLU 激活和给定结构的神经网络所能表示的函数类做出了更好的贡献。尽管普适逼近定理认为单层隐藏层就足以学习任何函数，但我们提供了一个数学的对称性，并详细探讨了添加更多层（无大小限制）时是否严格增加了可表示函数的类。作为研究副产品，我们肯定了 Wang 和 Sun（2005）有关分段线性函数的一个旧猜想。我们还给出了表示具有对数深度函数所需的神经网络大小上界。

    We contribute to a better understanding of the class of functions that can be represented by a neural network with ReLU activations and a given architecture. Using techniques from mixed-integer optimization, polyhedral theory, and tropical geometry, we provide a mathematical counterbalance to the universal approximation theorems which suggest that a single hidden layer is sufficient for learning any function. In particular, we investigate whether the class of exactly representable functions strictly increases by adding more layers (with no restrictions on size). As a by-product of our investigations, we settle an old conjecture about piecewise linear functions by Wang and Sun (2005) in the affirmative. We also present upper bounds on the sizes of neural networks required to represent functions with logarithmic depth.
    

