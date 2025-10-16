# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph-Structured Kernel Design for Power Flow Learning using Gaussian Processes.](http://arxiv.org/abs/2308.07867) | 本文提出了一种图结构核设计，用于使用高斯过程进行功率流学习，通过顶点度核和网络扫描主动学习方案，实现了更高效的学习和样本复杂度降低。 |
| [^2] | [Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs.](http://arxiv.org/abs/2206.02346) | 本文研究了约束马尔可夫决策过程中优化问题的自然策略梯度原始-对偶方法。通过自然策略梯度上升和投影次梯度下降更新变量，我们的方法在全局收敛中实现了次线性速率，而且不受状态-动作空间大小限制。 |

# 详细

[^1]: 使用高斯过程进行功率流学习的图结构核设计

    Graph-Structured Kernel Design for Power Flow Learning using Gaussian Processes. (arXiv:2308.07867v1 [eess.SY])

    [http://arxiv.org/abs/2308.07867](http://arxiv.org/abs/2308.07867)

    本文提出了一种图结构核设计，用于使用高斯过程进行功率流学习，通过顶点度核和网络扫描主动学习方案，实现了更高效的学习和样本复杂度降低。

    

    本文提出了一种基于物理启发的图结构核设计，用于使用高斯过程进行功率流学习。该核被命名为顶点度核（VDK），它依赖于基于网络图或拓扑的电压注入关系的潜在分解。值得注意的是，VDK设计避免了需要解决核搜索的优化问题。为了提高效率，我们还探索了一种图缩减方法，以获得具有较少项的VDK表示。此外，我们提出了一种新颖的网络扫描主动学习方案，它智能地选择顺序训练输入，加速VDK的学习。利用VDK的可加性结构，主动学习算法对GP的预测方差进行了块下降类型的过程，作为信息增益的代理。仿真结果表明，所提出的VDK-GP与中等规模500个节点和大规模1354个节点的完整GP相比，实现了超过两倍的样本复杂度降低。

    This paper presents a physics-inspired graph-structured kernel designed for power flow learning using Gaussian Process (GP). The kernel, named the vertex-degree kernel (VDK), relies on latent decomposition of voltage-injection relationship based on the network graph or topology. Notably, VDK design avoids the need to solve optimization problems for kernel search. To enhance efficiency, we also explore a graph-reduction approach to obtain a VDK representation with lesser terms. Additionally, we propose a novel network-swipe active learning scheme, which intelligently selects sequential training inputs to accelerate the learning of VDK. Leveraging the additive structure of VDK, the active learning algorithm performs a block-descent type procedure on GP's predictive variance, serving as a proxy for information gain. Simulations demonstrate that the proposed VDK-GP achieves more than two fold sample complexity reduction, compared to full GP on medium scale 500-Bus and large scale 1354-Bus 
    
[^2]: 自然策略梯度原始-对偶方法在约束MDP中的收敛性和样本复杂度研究

    Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs. (arXiv:2206.02346v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2206.02346](http://arxiv.org/abs/2206.02346)

    本文研究了约束马尔可夫决策过程中优化问题的自然策略梯度原始-对偶方法。通过自然策略梯度上升和投影次梯度下降更新变量，我们的方法在全局收敛中实现了次线性速率，而且不受状态-动作空间大小限制。

    

    我们研究了顺序决策问题，旨在最大化预期总奖励，同时满足对预期总效用的约束。我们使用自然策略梯度方法来解决约束马尔可夫决策过程（约束MDP）的折扣无限时序优化控制问题。具体地，我们提出了一种新的自然策略梯度原始-对偶（NPG-PD）方法，该方法通过自然策略梯度上升更新原始变量，通过投影次梯度下降更新对偶变量。尽管底层最大化涉及非凸目标函数和非凸约束集，但在softmax策略参数化下，我们证明了我们的方法在优化间隙和约束违规方面实现全局收敛，并具有次线性速率。此类收敛与状态-动作空间的大小无关，即无维度限制。此外，对于对数线性和一般平滑策略参数化，我们确立了收敛性和样本复杂度界限。

    We study sequential decision making problems aimed at maximizing the expected total reward while satisfying a constraint on the expected total utility. We employ the natural policy gradient method to solve the discounted infinite-horizon optimal control problem for Constrained Markov Decision Processes (constrained MDPs). Specifically, we propose a new Natural Policy Gradient Primal-Dual (NPG-PD) method that updates the primal variable via natural policy gradient ascent and the dual variable via projected sub-gradient descent. Although the underlying maximization involves a nonconcave objective function and a nonconvex constraint set, under the softmax policy parametrization we prove that our method achieves global convergence with sublinear rates regarding both the optimality gap and the constraint violation. Such convergence is independent of the size of the state-action space, i.e., it is~dimension-free. Furthermore, for log-linear and general smooth policy parametrizations, we esta
    

