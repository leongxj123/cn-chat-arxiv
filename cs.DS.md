# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional Linear Bandits with Knapsacks.](http://arxiv.org/abs/2311.01327) | 本文研究了具有背包约束的高维线性赌臂问题，利用稀疏结构实现改进遗憾。通过开发在线硬阈值算法和原始-对偶框架结合的方法，实现了对特征维度的对数改进的次线性遗憾。 |

# 详细

[^1]: 具有背包约束的高维线性赌臂问题研究

    High-dimensional Linear Bandits with Knapsacks. (arXiv:2311.01327v1 [cs.LG])

    [http://arxiv.org/abs/2311.01327](http://arxiv.org/abs/2311.01327)

    本文研究了具有背包约束的高维线性赌臂问题，利用稀疏结构实现改进遗憾。通过开发在线硬阈值算法和原始-对偶框架结合的方法，实现了对特征维度的对数改进的次线性遗憾。

    

    我们研究了在特征维度较大的高维设置下的具有背包约束的上下文赌臂问题。每个手臂拉动的奖励等于稀疏高维权重向量与当前到达的特征的乘积，加上额外的随机噪声。在本文中，我们研究如何利用这种稀疏结构来实现CBwK问题的改进遗憾。为此，我们首先开发了一种在线的硬阈值算法的变体，以在线方式进行稀疏估计。我们进一步将我们的在线估计器与原始-对偶框架结合起来，在每个背包约束上分配一个对偶变量，并利用在线学习算法来更新对偶变量，从而控制背包容量的消耗。我们证明了这种集成方法使我们能够实现对特征维度的对数改进的次线性遗憾，从而改进了多项式相关性。

    We study the contextual bandits with knapsack (CBwK) problem under the high-dimensional setting where the dimension of the feature is large. The reward of pulling each arm equals the multiplication of a sparse high-dimensional weight vector and the feature of the current arrival, with additional random noise. In this paper, we investigate how to exploit this sparsity structure to achieve improved regret for the CBwK problem. To this end, we first develop an online variant of the hard thresholding algorithm that performs the sparse estimation in an online manner. We further combine our online estimator with a primal-dual framework, where we assign a dual variable to each knapsack constraint and utilize an online learning algorithm to update the dual variable, thereby controlling the consumption of the knapsack capacity. We show that this integrated approach allows us to achieve a sublinear regret that depends logarithmically on the feature dimension, thus improving the polynomial depend
    

